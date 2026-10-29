"""Fully local MCP client (custom feature: "MCP client application").

Connects to the MLX Model Scout server over stdio and drives the recommendation
workflow with the LOCAL LLM (omlx / mlx_lm.server), so no prompt or tool result
leaves the machine (unlike Cursor/Claude Desktop, whose agent model may run in
the cloud). Human-in-the-loop happens in the terminal.

Modes:
- server (default): calls `run_recommendation`, renders the checkpoints as tables,
  asks the user and resumes with `continue_recommendation`. Deterministic and robust.
- agent: the local LLM is a conversational front end. It talks to the user and turns
  free-text answers into tool calls (OpenAI tool-calling format) on a curated tool set:
  the server-side workflow (`run_recommendation` / `continue_recommendation`) plus
  read-only lookups. Reports therefore always come from the validated advisor.
  The client enforces HITL: an approving call needs a y/n confirmation first.

Why not give the local model the `recommend_models` prompt and all tools? Tested with
Qwen3-30B-A3B: it invented a run_id, skipped checkpoint 1 and wrote and saved its own
report via `save_report`, bypassing the advisor's validation. The prompt flow stays
for capable client models (Cursor, Claude Desktop).

Run:  uv run python client/local_client.py "German RAG over contracts" --profile local
      uv run python client/local_client.py "Python coding assistant" --mode agent
"""

import argparse
import asyncio
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from mlx_model_scout.agents.llm import LLMError, chat_with_tools
from mlx_model_scout.config.settings import get_settings
from mlx_model_scout.db.audit import log_hitl

PROJECT_DIR = Path(__file__).resolve().parents[1]
USABLE = ("fits", "tight")
MAX_AGENT_STEPS = 30
MAX_TOOL_RESULT_CHARS = 6000  # keep the local model's context small

Ask = Callable[[str], str]


# --- rendering -------------------------------------------------------------------------


def _fmt(value: Any, suffix: str = "") -> str:
    return "-" if value is None else f"{value}{suffix}"


def render_candidates(console: Console, run: dict) -> list[str]:
    """Table of candidates (+ no-fit alternatives). Returns repo ids in display order (1-based)."""
    rows = [(c, "") for c in run["candidates"]] + [(a, "alt") for a in run["alternatives"]]
    table = Table(title=f"Checkpoint 1 - candidates (run {run['run_id'][:8]})", show_lines=False)
    for col in ("#", "repo", "verdict", "req / avail GiB", "tok/s", "conf", "reason"):
        table.add_column(col, overflow="fold")
    for i, (row, kind) in enumerate(rows, 1):
        fit = row["fit"]
        style = "green" if fit["verdict"] == "fits" else "yellow" if fit["verdict"] == "tight" else "red"
        table.add_row(
            f"{i}{' ' + kind if kind else ''}",
            row["repo_id"],
            f"[{style}]{fit['verdict']}[/{style}]",
            f"{_fmt(fit['required_gb'])} / {fit['available_gb']}",
            _fmt(fit["est_decode_tps"]),
            fit["confidence"],
            escape(row["reason"]),
        )
    console.print(table)
    return [row["repo_id"] for row, _ in rows]


def render_report(console: Console, run: dict) -> None:
    report = run["report"]
    table = Table(title=f"Checkpoint 2 - recommendation ({report['generated_by']})", show_lines=True)
    for col in ("rank", "repo", "verdict", "tok/s", "req GiB", "rationale", "trade-offs"):
        table.add_column(col, overflow="fold")
    for rec in report["recommendations"]:
        fit = rec["fit"]
        table.add_row(
            str(rec["rank"]),
            rec["repo_id"],
            fit["verdict"],
            _fmt(fit["est_decode_tps"]),
            _fmt(fit["required_gb"]),
            escape(rec["rationale"]),
            escape("\n".join(rec["trade_offs"])) or "-",
        )
    console.print(table)
    console.print(Panel(escape(report["summary"]), title="Summary"))
    if report["excluded"]:
        console.print(
            "[dim]Excluded: "
            + escape("; ".join(f"{k}: {v}" for k, v in report["excluded"].items()))
            + "[/dim]"
        )


def render_notes(console: Console, notes: list[str]) -> None:
    for note in notes:
        console.print(f"[dim]- {escape(note)}[/dim]")


# --- server mode ---------------------------------------------------------------------------


def parse_selection(text: str, repo_ids: list[str]) -> dict[str, Any]:
    """Parse the checkpoint-1 answer into continue_recommendation arguments.

    Tokens (space/comma separated): numbers from the table = keep, `+org/repo` = add,
    `ctx=16384` = new context length, `a`/`abort` = abort. Empty input = keep all that fit.
    """
    args: dict[str, Any] = {}
    keep: list[str] = []
    for token in text.replace(",", " ").split():
        if token.lower() in ("a", "abort"):
            return {"abort": True}
        if token.startswith("+") and "/" in token:
            args.setdefault("add", []).append(token[1:])
        elif token.lower().startswith("ctx="):
            args["context_tokens"] = int(token[4:])
        elif token.isdigit() and 1 <= int(token) <= len(repo_ids):
            keep.append(repo_ids[int(token) - 1])
        else:
            raise ValueError(f"Cannot parse '{token}'")
    if keep or "add" in args:
        args["keep"] = keep
    return args


async def run_server_mode(
    client: Client, console: Console, ask: Ask, use_case: str, profile: str, context: int, max_candidates: int
) -> dict:
    """Drive run_recommendation / continue_recommendation with terminal checkpoints."""
    with console.status("Planning, scanning and estimating (local LLM)..."):
        run = await _call(
            client,
            "run_recommendation",
            use_case=use_case,
            profile_id=profile,
            context_tokens=context,
            max_candidates=max_candidates,
        )
    while run["status"] == "needs_user_input":
        if run["checkpoint"] == "select_candidates":
            repo_ids = render_candidates(console, run)
            render_notes(console, run["notes"])
            args = _ask_selection(console, ask, repo_ids)
        else:
            render_report(console, run)
            args = _ask_approval(console, ask)
        with console.status("Working (local LLM)..."):
            run = await _call(client, "continue_recommendation", run_id=run["run_id"], **args)
    style = "green" if run["status"] == "completed" else "yellow"
    console.print(
        f"[{style}]{escape(run['message'])}[/{style}]"
        + (f" (report id {run['report_id']})" if run["report_id"] else "")
    )
    return run


def _ask_selection(console: Console, ask: Ask, repo_ids: list[str]) -> dict[str, Any]:
    hint = "Keep: numbers (e.g. 1 3), +org/repo to add, ctx=N, Enter = all that fit, a = abort"
    while True:
        try:
            return parse_selection(ask(hint), repo_ids)
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")


def _ask_approval(console: Console, ask: Ask) -> dict[str, Any]:
    while True:
        answer = ask("[a]pprove and save / [f]eedback / [r]eject / a[b]ort").strip().lower()
        if answer in ("a", "approve"):
            return {"approve": True}
        if answer in ("f", "feedback"):
            return {"feedback": ask("What should change?")}
        if answer in ("r", "reject"):
            return {"approve": False}
        if answer in ("b", "abort"):
            return {"abort": True}
        console.print("[red]Please answer a, f, r or b.[/red]")


async def _call(client: Client, tool: str, **args: Any) -> dict:
    return (await client.call_tool(tool, args)).structured_content


# --- agent mode ---------------------------------------------------------------------------

# Tools the agent may use: the server-side workflow plus read-only lookups (no save_report,
# no download_model: approval goes through continue_recommendation and its checkpoint).
AGENT_TOOLS = {
    "list_hardware_profiles",
    "run_recommendation",
    "continue_recommendation",
    "estimate_fit",
    "analyze_model_card",
}

AGENT_SYSTEM = (
    "You help the user choose a local Apple-MLX model by driving a recommendation workflow:\n"
    "1. Call run_recommendation once. It returns status 'needs_user_input' at checkpoint "
    "'select_candidates' with candidates (and alternatives if nothing fits).\n"
    "2. Show the candidates as a short table (repo, verdict, required/available GiB, tok/s) and "
    "ask which to keep. Then STOP and wait for the user's answer (reply without a tool call).\n"
    "3. Call continue_recommendation with the SAME run_id and keep=[exact repo ids the user chose] "
    "(or add / context_tokens / abort). It returns checkpoint 'approve_report' with a report.\n"
    "4. Show the report (rank, repo, rationale, trade-offs, tok/s) and ask for approval. STOP.\n"
    "5. Only if the user approves: continue_recommendation(run_id, approve=true). For changes: "
    "feedback='...'. To reject: approve=false.\n"
    "Rules: copy run_id and repo ids exactly from tool results, never invent them. Never invent or "
    "recompute numbers; quote them from tool results. Use estimate_fit / analyze_model_card only "
    "to answer follow-up questions."
)


def mcp_tools_to_openai(tools: list[Any]) -> list[dict]:
    """Convert MCP tool definitions to the OpenAI `tools` format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "")[:1000],
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def needs_confirmation(name: str, args: dict) -> str | None:
    """Client-side HITL guard: a question for the user if the call would commit something."""
    if name == "save_report" and args.get("approved") is True:
        return "The model wants to SAVE the report as APPROVED. Do you approve it?"
    if name == "continue_recommendation" and args.get("approve") is True:
        return "The model wants to APPROVE the report of this run. Do you approve it?"
    if name == "download_model" and args.get("confirmed") is True:
        return f"The model wants to DOWNLOAD {args.get('repo_id')}. Proceed?"
    return None


async def _execute_tool(client: Client, console: Console, ask: Ask, name: str, raw_args: str) -> str:
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as exc:
        return f"Error: arguments are not valid JSON ({exc}). Retry with valid JSON."
    if name not in AGENT_TOOLS:
        return f"Error: tool '{name}' is not available. Available: {', '.join(sorted(AGENT_TOOLS))}."
    if question := needs_confirmation(name, args):
        answer = ask(f"{question} [y/n]").strip().lower()
        log_hitl(
            args.get("run_id"),
            f"client_confirm:{name}",
            options=["y", "n"],
            decision={"answer": answer, "args": args},
        )
        if answer not in ("y", "yes"):
            return "The user did NOT confirm. Do not retry; ask the user what to change."
    console.print(f"[cyan]-> {escape(name)}[/cyan] {escape(json.dumps(args)[:200])}")
    try:
        result = await client.call_tool(name, args)
    except ToolError as exc:
        return f"Tool error: {exc}"
    payload = result.structured_content if result.structured_content is not None else result.data
    return json.dumps(payload, default=str)[:MAX_TOOL_RESULT_CHARS]


async def run_agent_mode(
    client: Client,
    console: Console,
    ask: Ask,
    use_case: str,
    profile: str,
    context: int,
    llm_client: Any = None,
) -> list[dict]:
    """Agent loop: the local LLM drives the workflow tools; the terminal user answers questions."""
    tools = mcp_tools_to_openai([t for t in await client.list_tools() if t.name in AGENT_TOOLS])
    task = f"Use case: {use_case}\nHardware profile: {profile}\nContext length: {context} tokens"
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": task},
    ]
    for _ in range(MAX_AGENT_STEPS):
        try:
            with console.status("Local LLM thinking..."):
                msg = await chat_with_tools("client", messages, tools, client=llm_client)
        except LLMError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            break
        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in msg.tool_calls
                    ],
                }
            )
            for call in msg.tool_calls:
                result = await _execute_tool(
                    client, console, ask, call.function.name, call.function.arguments
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            continue
        # the model answers in Markdown (tables, lists); rich renders it, no markup escaping needed
        console.print(Panel(Markdown(msg.content or "(no answer)"), title="Assistant"))
        messages.append({"role": "assistant", "content": msg.content or ""})
        answer = ask("Your answer (Enter or 'quit' to end)")
        if answer.strip().lower() in ("", "quit", "exit"):
            break
        messages.append({"role": "user", "content": answer})
    else:
        console.print(f"[yellow]Stopped after {MAX_AGENT_STEPS} steps.[/yellow]")
    return messages


# --- entry point ----------------------------------------------------------------------------


def server_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Environment for the server subprocess.

    The MCP SDK starts stdio servers with a minimal environment, so SCOUT_* / FASTMCP_* overrides
    from the shell (e.g. SCOUT_DB_PATH) are forwarded explicitly; the server still reads `.env` for the rest.
    Server logs go to stderr, which would clutter the terminal UI: WARNING unless set.
    """
    env = {k: v for k, v in environ.items() if k.startswith(("SCOUT_", "FASTMCP_"))}
    env.setdefault("SCOUT_LOG_LEVEL", "WARNING")
    env.setdefault("FASTMCP_LOG_LEVEL", "WARNING")  # fastmcp's own "Starting MCP server" line
    return env


async def main(
    use_case: str, profile: str, context: int, mode: str, max_candidates: int, record: Path | None = None
) -> None:
    console = Console(record=record is not None)
    server_log = get_settings().db_path.parent / "server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    transport = StdioTransport(
        command="uv",
        args=["--directory", str(PROJECT_DIR), "run", "mlx-model-scout"],
        env=server_env(os.environ),
        log_file=server_log,  # server stderr (incl. tool tracebacks) stays out of the terminal UI
    )

    def ask(question: str) -> str:
        # escape: "[a]pprove" would otherwise be parsed as rich markup and disappear
        answer = Prompt.ask(escape(question), default="", show_default=False, console=console)
        if record:  # typed input is echoed by the terminal, not by rich: add it to the recording
            console.print(f"[bold cyan]> {escape(answer)}[/bold cyan]")
        return answer

    async with Client(transport) as client:
        console.print(f"[bold]MLX Model Scout[/bold] - {mode} mode, profile '{profile}', context {context}")
        console.print(f"[dim]Server log: {server_log}[/dim]")
        if mode == "server":
            await run_server_mode(client, console, ask, use_case, profile, context, max_candidates)
        else:
            await run_agent_mode(client, console, ask, use_case, profile, context)
    if record:
        console.save_svg(str(record), title=f"mlx-model-scout client ({mode} mode)")
        print(f"Session recorded to {record}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("use_case", help="What the model should do, e.g. 'Python coding assistant'")
    parser.add_argument("--profile", default="local", help="Hardware profile id (default: this Mac)")
    parser.add_argument("--context", type=int, default=8192, help="Target context length in tokens")
    parser.add_argument("--mode", choices=["server", "agent"], default="server")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--record", type=Path, help="Save the terminal session as SVG (for docs)")
    a = parser.parse_args()
    asyncio.run(main(a.use_case, a.profile, a.context, a.mode, a.max_candidates, a.record))
