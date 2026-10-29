"""Scout agent: selects candidate repos for a use case.

Two stages:
1. Deterministic pre-filter (no LLM): drop non-LLM pipelines and repos whose
   name-based size clearly exceeds the available memory; cap the pool so the
   prompt stays small for a local model.
2. LLM shortlist (structured output `CandidateList`), validated against the pool:
   unknown repo ids are dropped, duplicates removed, the list capped. If the LLM
   fails or proposes nothing valid, a deterministic fallback (most downloaded) is used.
"""

import logging

from openai import AsyncOpenAI

from mlx_model_scout.agents.llm import LLMError, structured_call
from mlx_model_scout.app.model_spec import size_gb_from_name
from mlx_model_scout.config.settings import Settings
from mlx_model_scout.schemas.agents import Candidate, CandidateList, ScoutResult
from mlx_model_scout.schemas.model import HubModel

log = logging.getLogger(__name__)

MAX_POOL = 40
# Name-based sizes are rough (no KV cache, no overhead): only drop repos far beyond the budget.
SIZE_SLACK = 1.1

NON_LLM_PIPELINES = {
    "automatic-speech-recognition",
    "audio-classification",
    "feature-extraction",
    "image-classification",
    "image-to-image",
    "object-detection",
    "sentence-similarity",
    "text-to-audio",
    "text-to-image",
    "text-to-speech",
    "zero-shot-image-classification",
}

SYSTEM = (
    "You select local LLM candidates for a user's use case from a list of Hugging Face MLX repos. "
    "Judge only from repo names, pipeline tags and popularity. Prefer instruct/chat models for "
    "assistants, coder models for coding, vision-language models only if images are needed. "
    "Copy repo ids EXACTLY from the list; never invent ids. Do not estimate memory or speed: "
    "that is computed later by a calculator."
)


def prefilter(models: list[HubModel], available_gb: float | None) -> tuple[list[HubModel], list[str]]:
    """Cheap, deterministic narrowing of the scan result. Returns (pool, notes)."""
    notes: list[str] = []
    pool = [m for m in models if m.pipeline_tag not in NON_LLM_PIPELINES]
    if dropped := len(models) - len(pool):
        notes.append(f"Pre-filter: {dropped} non-LLM repos dropped (pipeline tag).")
    if available_gb:
        fitting = []
        for m in pool:
            size = size_gb_from_name(m.repo_id)
            if size is None or size <= available_gb * SIZE_SLACK:
                fitting.append(m)
        if dropped := len(pool) - len(fitting):
            notes.append(f"Pre-filter: {dropped} repos dropped as clearly too large (size from name).")
        pool = fitting
    if len(pool) > MAX_POOL:
        pool = sorted(pool, key=lambda m: m.downloads or 0, reverse=True)[:MAX_POOL]
        notes.append(f"Pre-filter: pool capped at the {MAX_POOL} most downloaded repos.")
    return pool, notes


def _fallback(pool: list[HubModel], max_candidates: int) -> list[Candidate]:
    ranked = sorted(pool, key=lambda m: m.downloads or 0, reverse=True)[:max_candidates]
    return [Candidate(repo_id=m.repo_id, reason="Fallback: most downloaded in the pool.") for m in ranked]


def validate_candidates(
    proposed: list[Candidate], pool: list[HubModel], max_candidates: int
) -> tuple[list[Candidate], list[str]]:
    """Keep only repo ids from the pool, without duplicates, capped. Returns (valid, notes)."""
    known = {m.repo_id for m in pool}
    valid: list[Candidate] = []
    unknown: list[str] = []
    for c in proposed:
        if c.repo_id not in known:
            unknown.append(c.repo_id)
        elif all(v.repo_id != c.repo_id for v in valid):
            valid.append(c)
    notes = [f"Dropped repo ids not in the scan: {', '.join(unknown)}."] if unknown else []
    if len(valid) > max_candidates:
        notes.append(f"Kept the first {max_candidates} of {len(valid)} proposed candidates.")
    return valid[:max_candidates], notes


async def select_candidates(
    use_case: str,
    models: list[HubModel],
    available_gb: float | None = None,
    max_candidates: int = 8,
    run_id: str | None = None,
    client: AsyncOpenAI | None = None,
    settings: Settings | None = None,
) -> ScoutResult:
    """Shortlist up to `max_candidates` repos for `use_case` from a scan result."""
    pool, notes = prefilter(models, available_gb)
    if not pool:
        return ScoutResult(candidates=[], pool_size=0, llm_used=False, notes=[*notes, "No repos left."])

    listing = "\n".join(
        f"- {m.repo_id} | {m.pipeline_tag or '?'} | downloads {m.downloads or 0}" for m in pool
    )
    user = (
        f"Use case: {use_case}\n\nPick up to {max_candidates} candidates, best first, from these repos "
        f"(repo id | pipeline | downloads):\n{listing}"
    )
    try:
        draft = await structured_call(
            "scout", CandidateList, SYSTEM, user, run_id=run_id, client=client, settings=settings
        )
    except LLMError as exc:
        log.warning("Scout LLM call failed: %s", exc)
        notes.append("LLM unavailable; deterministic fallback (most downloaded).")
        return ScoutResult(
            candidates=_fallback(pool, max_candidates), pool_size=len(pool), llm_used=False, notes=notes
        )

    valid, validation_notes = validate_candidates(draft.candidates, pool, max_candidates)
    notes.extend(validation_notes)
    if not valid:
        notes.append("LLM proposed no valid candidates; deterministic fallback (most downloaded).")
        return ScoutResult(
            candidates=_fallback(pool, max_candidates), pool_size=len(pool), llm_used=False, notes=notes
        )
    return ScoutResult(candidates=valid, pool_size=len(pool), llm_used=True, notes=notes)
