"""Orchestrator: two-call HITL protocol end to end (fake HF + fake LLM), no-fit branch, fallbacks."""

import json

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from sqlmodel import select

from conftest import LLAMA3_8B_CONFIG
from mlx_model_scout.agents import orchestrator
from mlx_model_scout.agents.orchestrator import fallback_terms, validate_plan
from mlx_model_scout.app.alternatives import base_name, max_fitting_context, smaller_siblings
from mlx_model_scout.app.hf_client import HFClient
from mlx_model_scout.app.model_spec import build_model_spec
from mlx_model_scout.app.profiles import load_chips, load_profiles
from mlx_model_scout.config.settings import get_settings
from mlx_model_scout.db.models import HITLDecision, RunStateRow, ToolCall, WorkflowRun
from mlx_model_scout.db.session import session_scope
from mlx_model_scout.schemas.agents import ExecutionPlan
from mlx_model_scout.server import create_app
from mlx_model_scout.tools import hub_tools

PROFILE = "studio-m4max-64"  # 48 GiB available; detection-independent (CI runs on Linux)

LLAMA_70B = {
    **LLAMA3_8B_CONFIG,
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "intermediate_size": 28672,
}
# "Big-*" (70B) carry no size in the name on purpose: with "70B" the scout's name-based
# pre-filter would drop the 8-bit repo before the calculator, and the no-fit branch never runs.
CONFIGS = {
    "mlx-community/Coder-8B-Instruct-4bit": LLAMA3_8B_CONFIG,
    "mlx-community/Chat-8B-Instruct-4bit": LLAMA3_8B_CONFIG,
    "mlx-community/Big-Instruct-8bit": {**LLAMA_70B, "quantization": {"bits": 8, "group_size": 64}},
    "mlx-community/Big-Instruct-4bit": LLAMA_70B,
}
CARDS = {rid: f"# {rid}\nA model for Python coding and chat. HumanEval 80.1." for rid in CONFIGS}


def _hf_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/models":
        term = request.url.params.get("search", "").lower()
        items = [
            {"id": rid, "pipeline_tag": "text-generation", "downloads": 10}
            for rid in CONFIGS
            if term in rid.lower()
        ]
        return httpx.Response(200, json=items)
    for rid in CONFIGS:
        if path == f"/{rid}/resolve/main/config.json":
            return httpx.Response(200, json=CONFIGS[rid])
        if path == f"/{rid}/raw/main/README.md":
            return httpx.Response(200, text=CARDS[rid])
    return httpx.Response(404)


@pytest.fixture
def fake_hub(monkeypatch):
    transport = httpx.MockTransport(_hf_handler)
    monkeypatch.setattr(hub_tools, "HFClient", lambda: HFClient(transport=transport))
    monkeypatch.setattr(orchestrator, "HFClient", lambda: HFClient(transport=transport))


def _plan(*terms: str) -> str:
    return json.dumps({"search_terms": list(terms), "needs_vision": False, "rationale": "test"})


def _scout(*repo_ids: str) -> str:
    return json.dumps({"candidates": [{"repo_id": r, "reason": "fits the use case"} for r in repo_ids]})


CARD_JSON = json.dumps({"summary": "Python coder.", "strengths": ["code"], "reported_benchmarks": {}})


def _advisor(*repo_ids: str) -> str:
    ranking = [{"repo_id": r, "rationale": "Good for Python.", "trade_offs": []} for r in repo_ids]
    return json.dumps({"ranking": ranking, "summary": "Coder first."})


async def _call(client: Client, tool: str, **args) -> dict:
    return (await client.call_tool(tool, args)).structured_content


async def test_happy_path_two_checkpoints(fake_hub, fake_llm):
    coder, chat = "mlx-community/Coder-8B-Instruct-4bit", "mlx-community/Chat-8B-Instruct-4bit"
    fake_llm.replies = [_plan("instruct"), _scout(coder, chat)]
    async with Client(create_app()) as c:
        run = await _call(c, "run_recommendation", use_case="Python coding", profile_id=PROFILE)
        assert run["status"] == "needs_user_input" and run["checkpoint"] == "select_candidates"
        assert [x["repo_id"] for x in run["candidates"]] == [coder, chat]
        assert run["candidates"][0]["fit"]["verdict"] == "fits"
        assert "CHECKPOINT 1" in run["message"]

        fake_llm.replies = [CARD_JSON, _advisor(coder)]
        draft = await _call(c, "continue_recommendation", run_id=run["run_id"], keep=[coder])
        assert draft["checkpoint"] == "approve_report"
        assert [r["repo_id"] for r in draft["report"]["recommendations"]] == [coder]
        assert draft["report"]["generated_by"] == "llm"

        done = await _call(c, "continue_recommendation", run_id=run["run_id"], approve=True)
        assert done["status"] == "completed" and done["report_id"]

        with pytest.raises(ToolError, match="already finished"):
            await c.call_tool("continue_recommendation", {"run_id": run["run_id"], "approve": True})

    run_id = run["run_id"]
    with session_scope() as s:
        assert s.get(WorkflowRun, run_id).status == "succeeded"
        assert s.get(RunStateRow, run_id).checkpoint is None
        steps = [h.step for h in s.exec(select(HITLDecision).where(HITLDecision.run_id == run_id))]
        tools = {t.tool for t in s.exec(select(ToolCall).where(ToolCall.run_id == run_id))}
    assert steps == ["select_candidates", "approve_report"]
    assert {"scan_mlx_models", "estimate_fit", "analyze_model_card", "save_report"} <= tools


async def test_no_fit_branch_offers_smaller_variant(fake_hub, fake_llm):
    big8 = "mlx-community/Big-Instruct-8bit"
    big4 = "mlx-community/Big-Instruct-4bit"
    fake_llm.replies = [_plan("big"), _scout(big8)]
    run = await orchestrator.start("general chat", PROFILE)

    assert run.checkpoint == "select_candidates"
    assert [c.fit.verdict for c in run.candidates] == ["no_fit"]
    assert [a.repo_id for a in run.alternatives] == [big4]
    assert run.alternatives[0].fit.verdict in ("fits", "tight")
    assert "None of the candidates fits" in run.message

    fake_llm.replies = [CARD_JSON, _advisor(big4)]
    draft = await orchestrator.resume(run.run_id, keep=[big4])
    assert [r.repo_id for r in draft.report.recommendations] == [big4]


async def test_selection_without_fitting_model_stays_at_checkpoint_1(fake_hub, fake_llm):
    fake_llm.replies = [_plan("big"), _scout("mlx-community/Big-Instruct-8bit")]
    run = await orchestrator.start("general chat", PROFILE)
    again = await orchestrator.resume(run.run_id, keep=["mlx-community/Big-Instruct-8bit"])
    assert again.checkpoint == "select_candidates" and again.alternatives


async def test_llm_down_whole_flow_still_completes(fake_hub, fake_llm, llm_down):
    fake_llm.replies = [llm_down] * 10
    run = await orchestrator.start("Python coding assistant", PROFILE)
    assert run.plan.search_terms == ["coder", "instruct"]  # keyword fallback
    assert run.candidates  # scout fallback: most downloaded

    draft = await orchestrator.resume(run.run_id)  # keep = all that fit
    assert draft.report.generated_by == "deterministic"
    assert draft.report.recommendations


async def test_feedback_revises_then_abort(fake_hub, fake_llm):
    coder = "mlx-community/Coder-8B-Instruct-4bit"
    fake_llm.replies = [_plan("coder"), _scout(coder), CARD_JSON, _advisor(coder)]
    run = await orchestrator.start("Python coding", PROFILE)
    await orchestrator.resume(run.run_id)

    fake_llm.replies = [_advisor(coder)]  # card is cached; only the advisor runs again
    revised = await orchestrator.resume(run.run_id, feedback="Prefer smaller models.")
    assert revised.checkpoint == "approve_report"
    assert "Prefer smaller models" in fake_llm.requests[-1]["messages"][1]["content"]
    assert revised.report.use_case == "Python coding"  # feedback is not baked into the report

    aborted = await orchestrator.resume(run.run_id, abort=True)
    assert aborted.status == "aborted"
    with session_scope() as s:
        assert s.get(WorkflowRun, run.run_id).status == "aborted"


async def test_reject_stores_unapproved_report(fake_hub, fake_llm):
    coder = "mlx-community/Coder-8B-Instruct-4bit"
    fake_llm.replies = [_plan("coder"), _scout(coder), CARD_JSON, _advisor(coder)]
    run = await orchestrator.start("Python coding", PROFILE)
    await orchestrator.resume(run.run_id)
    rejected = await orchestrator.resume(run.run_id, approve=False)
    assert rejected.status == "aborted" and rejected.report_id


async def test_decisions_for_the_wrong_checkpoint_are_rejected(fake_hub, fake_llm):
    coder = "mlx-community/Coder-8B-Instruct-4bit"
    fake_llm.replies = [_plan("coder"), _scout(coder), CARD_JSON, _advisor(coder)]
    run = await orchestrator.start("Python coding", PROFILE)
    # a client model tried to skip checkpoint 1 by approving right away
    with pytest.raises(ValueError, match="checkpoint 'select_candidates'"):
        await orchestrator.resume(run.run_id, approve=True)
    draft = await orchestrator.resume(run.run_id, keep=[coder])  # run is still usable
    with pytest.raises(ValueError, match="checkpoint 'approve_report'"):
        await orchestrator.resume(draft.run_id, keep=[coder])
    with session_scope() as s:
        assert s.get(WorkflowRun, run.run_id).status == "running"  # misuse does not fail the run


async def test_unknown_run_id_raises():
    with pytest.raises(ValueError, match="Unknown run_id"):
        await orchestrator.resume("nope")


def test_validate_plan_normalizes_and_falls_back():
    plan, notes = validate_plan(
        ExecutionPlan(search_terms=["Coder", "a full sentence here please", "VL"]), "x"
    )
    assert plan.search_terms == ["coder", "vl"] and notes
    plan, _ = validate_plan(ExecutionPlan(search_terms=[]), "analyze images of invoices")
    assert plan.search_terms == ["vl", "instruct"]
    assert fallback_terms("chat") == ["instruct"]


def test_alternatives_helpers():
    assert base_name("mlx-community/Qwen3-Coder-Next-6bit") == "Qwen3-Coder-Next"
    assert base_name("mlx-community/gpt-oss-20b-MXFP4-Q8") == "gpt-oss-20b"
    repos = [
        "mlx-community/M-70B-Instruct-8bit",
        "mlx-community/M-70B-Instruct-4bit",
        "mlx-community/M-70B-Instruct-6bit",
        "mlx-community/M-70B-Instruct-bf16",
        "mlx-community/Other-70B-4bit",
        "someone/M-70B-Instruct-4bit",
    ]
    assert smaller_siblings("mlx-community/M-70B-Instruct-8bit", repos) == [
        "mlx-community/M-70B-Instruct-6bit",
        "mlx-community/M-70B-Instruct-4bit",
    ]


def test_max_fitting_context():
    spec = build_model_spec("x/llama-8b", LLAMA3_8B_CONFIG)
    small = load_profiles()[PROFILE].model_copy(update={"unified_memory_gb": 12, "reserved_gb": 0})
    chip = load_chips()[small.chip]
    # 12 GiB * 0.75 = 9 GiB: 4.2 GiB weights + 1.5 overhead leave ~3.3 GiB for KV (128 KiB/token)
    assert max_fitting_context(spec, small, chip, 131072, get_settings()) == 16384
