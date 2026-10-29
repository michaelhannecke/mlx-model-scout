"""Advisor agent: ranks fitting models for a use case and writes the rationale.

The LLM receives the computed FitEstimate numbers and card insights as FACTS and
returns only an order plus text (`AdvisorDraft`, no numeric fields). Code then
builds the RecommendationReport and attaches each FitEstimate unchanged, so the
numbers cannot be altered. Afterwards:
- repo ids not among the usable candidates are dropped, duplicates removed, max 5;
- any number+unit in the text (GiB, tok/s, ...) must match the facts, else the
  text is replaced by a deterministic one;
- if the LLM fails or ranks nothing valid, `rank_without_llm` is used.
"""

import logging
import re

from openai import AsyncOpenAI

from mlx_model_scout.agents.analyst import clip, clip_list
from mlx_model_scout.agents.llm import LLMError, structured_call
from mlx_model_scout.config.settings import Settings
from mlx_model_scout.schemas.agents import AdvisorDraft
from mlx_model_scout.schemas.fit import FitEstimate, Recommendation, RecommendationReport
from mlx_model_scout.schemas.model import ModelCardInsights

log = logging.getLogger(__name__)

MAX_RECOMMENDATIONS = 5
MAX_RATIONALE_CHARS = 400
MAX_SUMMARY_CHARS = 600
USABLE = {"fits": 0, "tight": 1}
_NUMBER_WITH_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:GiB|GB|tok/s|t/s|tokens?/s(?:ec)?|tokens per second)", re.IGNORECASE
)

SYSTEM = (
    "You are an advisor for running local LLMs on Apple Silicon with MLX. Rank the candidates "
    "for the user's use case. The memory and speed facts were computed by a calculator and are "
    "correct: use them for your judgement, but do NOT repeat or recompute any numbers in your "
    "text (they are shown to the user in a table). Weigh suitability for the use case first "
    "(card summary, use cases, strengths, caveats), then speed and memory headroom. 'tight' means "
    "little memory headroom. The 'memory headroom' and 'speed' labels are computed and correct: "
    "use them for any statement about memory or speed. A 'warning' line means the requested "
    "context exceeds what the model supports: rank it lower and mention it as a trade-off. "
    "Copy repo ids EXACTLY; leave out candidates that do not suit the use case."
)


def _exclusion_reason(fit: FitEstimate) -> str:
    if fit.verdict == "no_fit":
        return f"no_fit: requires {fit.required_gb} GiB, available {fit.available_gb} GiB"
    return f"{fit.verdict}: memory could not be estimated"


def rank_without_llm(
    use_case: str, profile_id: str, ctx: int, fits: list[FitEstimate], notes: list[str] | None = None
) -> RecommendationReport:
    """Deterministic fallback ranking (also useful as an eval baseline).

    Order: verdict (fits > tight), then estimated decode speed, then headroom.
    """
    usable = [f for f in fits if f.verdict in USABLE]
    usable.sort(key=lambda f: (USABLE[f.verdict], -(f.est_decode_tps or 0), -(f.headroom_gb or 0)))
    recs = [
        Recommendation(
            rank=i + 1,
            repo_id=f.repo_id,
            fit=f,
            rationale=f"{f.verdict}; ~{f.est_decode_tps or '?'} tok/s; headroom {f.headroom_gb} GiB",
        )
        for i, f in enumerate(usable[:MAX_RECOMMENDATIONS])
    ]
    excluded = {f.repo_id: _exclusion_reason(f) for f in fits if f.verdict not in USABLE}
    excluded |= {f.repo_id: "lower in deterministic ranking" for f in usable[MAX_RECOMMENDATIONS:]}
    return RecommendationReport(
        use_case=use_case,
        profile_id=profile_id,
        context_tokens=ctx,
        recommendations=recs,
        excluded=excluded,
        summary=f"{len(recs)} usable of {len(fits)} candidates (deterministic ranking by fit and speed).",
        generated_by="deterministic",
        notes=notes or [],
    )


def _fact_values(fits: list[FitEstimate]) -> set[float]:
    values: set[float] = set()
    for f in fits:
        for v in (
            f.weights_gb,
            f.kv_cache_gb,
            f.required_gb,
            f.available_gb,
            f.headroom_gb,
            f.est_decode_tps,
        ):
            if v is not None:
                values |= {round(v), round(v, 1), round(v, 2)}
    return values


def numbers_match_facts(text: str, fits: list[FitEstimate]) -> bool:
    """True if every number+unit in `text` equals a (rounded) value from `fits`."""
    allowed = _fact_values(fits)
    return all(float(m.group(1)) in allowed for m in _NUMBER_WITH_UNIT.finditer(text))


def memory_label(fit: FitEstimate) -> str:
    """Qualitative memory headroom computed by code (local models misjudge raw GiB numbers)."""
    if fit.headroom_gb is None or not fit.available_gb:
        return "unknown"
    share = fit.headroom_gb / fit.available_gb
    return "ample" if share >= 0.5 else "moderate" if share >= 0.2 else "tight"


def speed_labels(fits: list[FitEstimate]) -> dict[str, str]:
    """'speed rank i of n' per repo (1 = fastest), computed by code; unknown speeds are not ranked."""
    known = sorted((f for f in fits if f.est_decode_tps), key=lambda f: -f.est_decode_tps)
    labels = {}
    for i, f in enumerate(known, 1):
        suffix = " (fastest)" if i == 1 else " (slowest)" if i == len(known) and i > 1 else ""
        labels[f.repo_id] = f"rank {i} of {len(known)}{suffix}"
    return labels


def _facts_block(fits: list[FitEstimate], insights: dict[str, ModelCardInsights]) -> str:
    blocks = []
    speed = speed_labels(fits)
    for f in fits:
        lines = [
            f"### {f.repo_id}",
            f"fit: {f.verdict}, est. decode {f.est_decode_tps or 'unknown'} tok/s, "
            f"headroom {f.headroom_gb} GiB, confidence {f.confidence}",
            f"memory headroom: {memory_label(f)}; speed: {speed.get(f.repo_id, 'unknown')}",
        ]
        lines += [f"warning: {a}" for a in f.assumptions if a.startswith("Requested context")]
        card = insights.get(f.repo_id)
        if card and card.llm_used:
            lines += [
                f"summary: {card.summary or '-'}",
                f"use cases: {', '.join(card.primary_use_cases) or '-'}",
                f"strengths: {', '.join(card.strengths) or '-'}",
                f"caveats: {', '.join(card.caveats) or '-'}",
            ]
        else:
            lines.append("card: not analyzed (judge from the repo name)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def draft_report(
    use_case: str,
    profile_id: str,
    ctx: int,
    fits: list[FitEstimate],
    insights: dict[str, ModelCardInsights] | None = None,
    run_id: str | None = None,
    client: AsyncOpenAI | None = None,
    settings: Settings | None = None,
) -> RecommendationReport:
    """Rank usable candidates with the LLM; validate; fall back to deterministic ranking."""
    insights = insights or {}
    usable = {f.repo_id: f for f in fits if f.verdict in USABLE}
    excluded = {f.repo_id: _exclusion_reason(f) for f in fits if f.verdict not in USABLE}
    if not usable:
        return rank_without_llm(
            use_case, profile_id, ctx, fits, notes=["No candidate fits; nothing to rank."]
        )

    user = (
        f"Use case: {use_case}\nContext length: {ctx} tokens\n\n"
        f"Candidates (max {MAX_RECOMMENDATIONS} in your ranking):\n\n"
        f"{_facts_block(list(usable.values()), insights)}"
    )
    try:
        draft = await structured_call(
            "advisor", AdvisorDraft, SYSTEM, user, run_id=run_id, client=client, settings=settings
        )
    except LLMError as exc:
        log.warning("Advisor LLM call failed: %s", exc)
        return rank_without_llm(
            use_case, profile_id, ctx, fits, notes=["LLM unavailable; deterministic ranking."]
        )

    notes: list[str] = []
    recs: list[Recommendation] = []
    unknown: list[str] = []
    for item in draft.ranking:
        fit = usable.get(item.repo_id)
        if fit is None:
            unknown.append(item.repo_id)
            continue
        if any(r.repo_id == item.repo_id for r in recs) or len(recs) >= MAX_RECOMMENDATIONS:
            continue
        rationale, trade_offs = (
            clip(item.rationale, MAX_RATIONALE_CHARS),
            clip_list(item.trade_offs, max_items=3),
        )
        if not numbers_match_facts(" ".join([rationale, *trade_offs]), [fit]):
            notes.append(f"{item.repo_id}: rationale contained numbers not matching the facts; replaced.")
            rationale = f"{fit.verdict}; ~{fit.est_decode_tps or '?'} tok/s; headroom {fit.headroom_gb} GiB"
            trade_offs = []
        recs.append(
            Recommendation(
                rank=len(recs) + 1, repo_id=item.repo_id, fit=fit, rationale=rationale, trade_offs=trade_offs
            )
        )
    if unknown:
        notes.append(f"Dropped repo ids not among the usable candidates: {', '.join(unknown)}.")
    if not recs:
        return rank_without_llm(
            use_case,
            profile_id,
            ctx,
            fits,
            notes=[*notes, "LLM ranked no valid candidate; deterministic ranking."],
        )

    ranked = {r.repo_id for r in recs}
    excluded |= {rid: "not ranked by the advisor for this use case" for rid in usable if rid not in ranked}
    summary = clip(draft.summary, MAX_SUMMARY_CHARS)
    if not numbers_match_facts(summary, list(usable.values())):
        notes.append("Summary contained numbers not matching the facts; replaced.")
        summary = f"{len(recs)} recommended of {len(fits)} candidates."
    return RecommendationReport(
        use_case=use_case,
        profile_id=profile_id,
        context_tokens=ctx,
        recommendations=recs,
        excluded=excluded,
        summary=summary,
        generated_by="llm",
        notes=notes,
    )
