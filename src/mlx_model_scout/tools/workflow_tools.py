"""Tool implementations: the server-side recommend workflow (orchestrator, two-call HITL)."""

from mlx_model_scout.agents import orchestrator
from mlx_model_scout.db.audit import tool_audit
from mlx_model_scout.schemas.agents import RecommendationRun


async def run_recommendation(
    use_case: str, profile_id: str = "local", context_tokens: int | None = None, max_candidates: int = 6
) -> RecommendationRun:
    """Start a run; returns at checkpoint 1 (candidate selection)."""
    args = {"use_case": use_case, "profile_id": profile_id, "context_tokens": context_tokens}
    with tool_audit("run_recommendation", args) as audit:
        result = await orchestrator.start(use_case, profile_id, context_tokens, max_candidates)
        audit["summary"] = f"run {result.run_id}: {result.status} at {result.checkpoint}"
        return result


async def continue_recommendation(
    run_id: str,
    keep: list[str] | None = None,
    add: list[str] | None = None,
    context_tokens: int | None = None,
    approve: bool | None = None,
    feedback: str | None = None,
    abort: bool = False,
) -> RecommendationRun:
    """Resume a run with the user's decision at its current checkpoint."""
    args = {
        "keep": keep,
        "add": add,
        "context_tokens": context_tokens,
        "approve": approve,
        "feedback": feedback,
        "abort": abort,
    }
    with tool_audit("continue_recommendation", args, run_id) as audit:
        result = await orchestrator.resume(run_id, keep, add, context_tokens, approve, feedback, abort)
        audit["summary"] = f"{result.status} at {result.checkpoint}"
        return result
