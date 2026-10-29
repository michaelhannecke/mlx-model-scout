"""Local MCP client: input parsing, HITL guard, server mode and agent mode (in-memory, offline)."""

import io
import json

import pytest
from fastmcp import Client
from rich.console import Console
from sqlmodel import select

import local_client
from conftest import tool_call
from local_client import (
    mcp_tools_to_openai,
    needs_confirmation,
    parse_selection,
    run_agent_mode,
    run_server_mode,
)
from mlx_model_scout.db.models import HITLDecision, LLMCall, ReportRow
from mlx_model_scout.db.session import session_scope
from mlx_model_scout.server import create_app
from test_orchestrator import CARD_JSON, PROFILE, _advisor, _plan, _scout, fake_hub  # noqa: F401 (fixture, used via usefixtures)

CODER = "mlx-community/Coder-8B-Instruct-4bit"
CHAT = "mlx-community/Chat-8B-Instruct-4bit"


def _console() -> Console:
    return Console(file=io.StringIO(), width=200)


def _answers(*answers: str):
    queue = list(answers)
    asked: list[str] = []

    def ask(question: str) -> str:
        asked.append(question)
        return queue.pop(0)

    ask.asked = asked
    return ask


def test_parse_selection():
    ids = ["a/one", "a/two", "a/three"]
    assert parse_selection("", ids) == {}
    assert parse_selection("1, 3", ids) == {"keep": ["a/one", "a/three"]}
    assert parse_selection("2 +org/new ctx=16384", ids) == {
        "keep": ["a/two"],
        "add": ["org/new"],
        "context_tokens": 16384,
    }
    assert parse_selection("+org/new", ids) == {"add": ["org/new"], "keep": []}
    assert parse_selection("1 abort", ids) == {"abort": True}
    with pytest.raises(ValueError):
        parse_selection("7", ids)


def test_needs_confirmation_guards_committing_calls_only():
    assert needs_confirmation("save_report", {"approved": True})
    assert needs_confirmation("continue_recommendation", {"run_id": "x", "approve": True})
    assert needs_confirmation("download_model", {"repo_id": "a/b", "confirmed": True})
    assert needs_confirmation("save_report", {"approved": False}) is None
    assert needs_confirmation("continue_recommendation", {"run_id": "x", "keep": ["a/b"]}) is None
    assert needs_confirmation("estimate_fit", {"repo_id": "a/b"}) is None


async def test_mcp_tools_convert_to_openai_format():
    async with Client(create_app()) as c:
        tools = mcp_tools_to_openai(await c.list_tools())
    by_name = {t["function"]["name"]: t for t in tools}
    assert by_name["estimate_fit"]["type"] == "function"
    assert "repo_id" in by_name["estimate_fit"]["function"]["parameters"]["properties"]


@pytest.mark.usefixtures("fake_hub")
async def test_server_mode_select_and_approve(fake_llm):
    fake_llm.replies = [_plan("instruct"), _scout(CODER, CHAT), CARD_JSON, _advisor(CODER)]
    ask = _answers("1", "a")
    console = _console()
    async with Client(create_app()) as c:
        run = await run_server_mode(c, console, ask, "Python coding", PROFILE, 8192, 4)

    assert run["status"] == "completed" and run["report_id"]
    output = console.file.getvalue()
    assert "Checkpoint 1" in output and "Checkpoint 2" in output and CODER in output
    with session_scope() as s:
        assert s.exec(select(ReportRow)).one().approved is True


@pytest.mark.usefixtures("fake_hub")
async def test_server_mode_reprompts_on_bad_input_and_aborts(fake_llm):
    fake_llm.replies = [_plan("instruct"), _scout(CODER)]
    ask = _answers("99", "abort")
    async with Client(create_app()) as c:
        run = await run_server_mode(c, _console(), ask, "chat", PROFILE, 8192, 4)
    assert run["status"] == "aborted" and len(ask.asked) == 2


async def test_agent_mode_uses_curated_tools_and_asks_user(fake_llm):
    fake_llm.replies = [tool_call("list_hardware_profiles", {}), "Which profile should I use?"]
    ask = _answers("quit")
    async with Client(create_app()) as c:
        messages = await run_agent_mode(c, _console(), ask, "chat", PROFILE, 8192)

    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert PROFILE in tool_msg["content"]
    assert messages[-1] == {"role": "assistant", "content": "Which profile should I use?"}
    first_request = fake_llm.requests[0]
    assert {t["function"]["name"] for t in first_request["tools"]} == local_client.AGENT_TOOLS
    assert "Use case: chat" in first_request["messages"][1]["content"]
    with session_scope() as s:
        responses = [r.response for r in s.exec(select(LLMCall).where(LLMCall.agent == "client"))]
    assert "[tool_calls] list_hardware_profiles" in responses[0]


@pytest.mark.usefixtures("fake_hub")
async def test_agent_mode_drives_the_two_checkpoint_workflow(fake_llm):
    async with Client(create_app()) as c:
        # the queue interleaves the client agent and the server-side agents (same fake LLM)
        fake_llm.replies = [
            tool_call("run_recommendation", {"use_case": "Python coding", "profile_id": PROFILE}),
            _plan("instruct"),
            _scout(CODER, CHAT),
            "Which candidates do you want to keep?",
        ]
        user = _answers("Keep the coder.", "Yes, approve.", "y", "quit")

        def scripted_ask(question: str) -> str:
            # The run_id only exists at runtime, so the agent's next moves are queued after each answer.
            answer = user(question)
            if answer == "Keep the coder.":
                run_id = _run_id_from(fake_llm)
                fake_llm.replies += [
                    tool_call("continue_recommendation", {"run_id": run_id, "keep": [CODER]}, "c2"),
                    CARD_JSON,
                    _advisor(CODER),
                    "Here is the report. Approve?",
                ]
            elif answer == "Yes, approve.":
                run_id = _run_id_from(fake_llm)
                fake_llm.replies += [
                    tool_call("continue_recommendation", {"run_id": run_id, "approve": True}, "c3"),
                    "Saved.",
                ]
            return answer

        await run_agent_mode(c, _console(), scripted_ask, "Python coding", PROFILE, 8192)

    with session_scope() as s:
        reports = s.exec(select(ReportRow)).all()
        approved = [r.approved for r in reports]
        steps = [h.step for h in s.exec(select(HITLDecision))]
    assert approved == [True]
    assert steps == ["select_candidates", "client_confirm:continue_recommendation", "approve_report"]


def _run_id_from(fake_llm) -> str:
    """run_id from the first workflow tool result the client sent back to the LLM."""
    for request in reversed(fake_llm.requests):
        for m in request["messages"]:
            if m["role"] == "tool" and '"run_id"' in m["content"]:
                return json.loads(m["content"])["run_id"]
    raise AssertionError("no run_id in tool results")


async def test_agent_mode_hitl_guard_blocks_unconfirmed_approval(fake_llm):
    fake_llm.replies = [
        tool_call("continue_recommendation", {"run_id": "r1", "approve": True}),
        "Okay, what should change?",
    ]
    ask = _answers("n", "quit")
    async with Client(create_app()) as c:
        messages = await run_agent_mode(c, _console(), ask, "chat", PROFILE, 8192)

    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert "did NOT confirm" in tool_msg["content"]  # the tool was never called (r1 would error)
    with session_scope() as s:
        decision = s.exec(select(HITLDecision)).one()
        step, answer, run_id = decision.step, json.loads(decision.decision_json)["answer"], decision.run_id
    assert step == "client_confirm:continue_recommendation" and answer == "n"
    assert run_id == "r1"  # the confirmation shows up in the run timeline of the audit log


async def test_agent_mode_refuses_tools_outside_the_curated_set(fake_llm):
    fake_llm.replies = [tool_call("save_report", {"report": {"summary": "x"}, "approved": False}), "Ok."]
    async with Client(create_app()) as c:
        messages = await run_agent_mode(c, _console(), _answers("quit"), "chat", PROFILE, 8192)
    assert "not available" in next(m for m in messages if m["role"] == "tool")["content"]
    with session_scope() as s:
        assert s.exec(select(ReportRow)).first() is None


async def test_agent_mode_reports_invalid_tool_arguments(fake_llm):
    bad = {"tool_calls": [{"id": "c1", "name": "estimate_fit", "arguments": "{not json"}]}
    fake_llm.replies = [bad, "Sorry."]
    async with Client(create_app()) as c:
        messages = await run_agent_mode(c, _console(), _answers("quit"), "chat", PROFILE, 8192)
    assert "not valid JSON" in next(m for m in messages if m["role"] == "tool")["content"]


async def test_agent_mode_stops_when_llm_is_down(fake_llm, llm_down):
    fake_llm.replies = [llm_down]
    console = _console()
    async with Client(create_app()) as c:
        messages = await run_agent_mode(c, console, _answers(), "chat", PROFILE, 8192)
    assert len(messages) == 2 and "Local LLM call failed" in console.file.getvalue()


def test_server_env_forwards_scout_vars_only():
    env = local_client.server_env({"SCOUT_DB_PATH": "/tmp/x.db", "HOME": "/h", "SCOUT_LLM_API_KEY": "k"})
    assert env == {
        "SCOUT_DB_PATH": "/tmp/x.db",
        "SCOUT_LLM_API_KEY": "k",
        "SCOUT_LOG_LEVEL": "WARNING",
        "FASTMCP_LOG_LEVEL": "WARNING",
    }
    assert local_client.server_env({"SCOUT_LOG_LEVEL": "DEBUG"})["SCOUT_LOG_LEVEL"] == "DEBUG"


def test_module_has_no_own_openai_client():
    # decision 4 (README): the OpenAI client is only created in agents/llm.py (audit logging)
    with open(local_client.__file__, encoding="utf-8") as fh:
        source = fh.read()
    assert "import openai" not in source and "from openai" not in source and "AsyncOpenAI" not in source


def test_llm_text_is_escaped_for_rich():
    # Qwen card caveat "No [think][/think] blocks" was rendered as "No  blocks" (rich markup)
    console = _console()
    run = {
        "report": {
            "generated_by": "llm",
            "recommendations": [
                {
                    "rank": 1,
                    "repo_id": "a/b",
                    "fit": {"verdict": "fits", "est_decode_tps": 1.0, "required_gb": 1.0},
                    "rationale": "Uses [bold]tags[/bold].",
                    "trade_offs": ["No [think][/think] blocks"],
                }
            ],
            "summary": "[red]x[/red]",
            "excluded": {},
        }
    }
    local_client.render_report(console, run)
    out = console.file.getvalue()
    assert "[think][/think]" in out and "[bold]tags[/bold]" in out and "[red]x[/red]" in out
