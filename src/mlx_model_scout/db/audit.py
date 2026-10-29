"""Audit logging helpers (replacement for an external observability stack).

Everything relevant ends up in SQLite and can be browsed with Datasette
(see compose.yaml). Logging must never break the main flow: all helpers
swallow and report their own errors to stderr.
"""

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlmodel import delete

from mlx_model_scout.config.settings import get_settings
from mlx_model_scout.db.models import HITLDecision, LLMCall, ToolCall, WorkflowRun, utcnow
from mlx_model_scout.db.session import session_scope

log = logging.getLogger(__name__)


def new_run_id() -> str:
    return uuid.uuid4().hex


def _dump(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def redact_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (sha256, stored_text) honoring SCOUT_LOG_FULL_MESSAGES."""
    settings = get_settings()
    full = _dump(messages)
    digest = hashlib.sha256(full.encode()).hexdigest()
    if settings.log_full_messages:
        return digest, full
    limit = settings.log_truncate_chars
    truncated = [{**m, "content": str(m.get("content", ""))[:limit]} for m in messages]
    return digest, _dump(truncated)


def log_llm_call(**fields: Any) -> None:
    """Persist one LLM call. Expects LLMCall field names; `messages` as list."""
    try:
        messages = fields.pop("messages", [])
        digest, stored = redact_messages(messages)
        response = fields.pop("response", None)
        if response is not None and not get_settings().log_full_messages:
            response = str(response)[: get_settings().log_truncate_chars]
        with session_scope() as s:
            s.add(LLMCall(messages_sha256=digest, messages=stored, response=response, **fields))
    except Exception:  # pragma: no cover - logging must not break the flow
        log.exception("Failed to log LLM call")


@contextmanager
def tool_audit(tool: str, arguments: dict[str, Any], run_id: str | None = None) -> Iterator[dict]:
    """Context manager that logs a tool invocation with duration and status.

    Usage:
        with tool_audit("estimate_fit", args) as audit:
            result = ...
            audit["summary"] = "fits, 12.3 GiB"
    """
    audit: dict[str, Any] = {"summary": None}
    start = time.perf_counter()
    status, error = "ok", None
    try:
        yield audit
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            with session_scope() as s:
                s.add(
                    ToolCall(
                        run_id=run_id,
                        tool=tool,
                        arguments_json=_dump(arguments),
                        result_summary=audit.get("summary"),
                        duration_ms=int((time.perf_counter() - start) * 1000),
                        status=status,
                        error=error,
                    )
                )
        except Exception:  # pragma: no cover
            log.exception("Failed to log tool call")


def start_run(workflow: str, inputs: dict[str, Any]) -> str:
    run_id = new_run_id()
    with session_scope() as s:
        s.add(WorkflowRun(id=run_id, workflow=workflow, input_json=_dump(inputs)))
    return run_id


def finish_run(run_id: str, status: str = "succeeded", error: str | None = None) -> None:
    with session_scope() as s:
        run = s.get(WorkflowRun, run_id)
        if run:
            run.status, run.error, run.finished_at = status, error, utcnow()
            s.add(run)


def log_hitl(run_id: str | None, step: str, options: Any, decision: Any) -> None:
    with session_scope() as s:
        s.add(
            HITLDecision(run_id=run_id, step=step, options_json=_dump(options), decision_json=_dump(decision))
        )


def purge_old_logs(days: int | None = None) -> int:
    """Delete audit rows older than the retention window. Returns rows deleted."""
    cutoff = datetime.now(UTC) - timedelta(days=days or get_settings().log_retention_days)
    deleted = 0
    with session_scope() as s:
        for table in (LLMCall, ToolCall):
            result = s.exec(delete(table).where(table.created_at < cutoff))
            deleted += result.rowcount or 0
    return deleted
