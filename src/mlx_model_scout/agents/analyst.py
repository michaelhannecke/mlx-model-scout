"""Analyst agent: turns a model card into structured insights.

Facts that exist as structured data (config.json, card front matter) are derived
deterministically (app.model_spec, app.model_card); the LLM only handles the
unstructured prose of the card. MLX conversion cards are boilerplate, so the
base model's card is preferred when available.
"""

import logging

from openai import AsyncOpenAI

from mlx_model_scout.agents.llm import LLMError, structured_call
from mlx_model_scout.app.model_card import card_metadata, clean_card_text, is_boilerplate, verify_benchmarks
from mlx_model_scout.config.settings import Settings
from mlx_model_scout.schemas.model import CardExtraction, ModelCardInsights

log = logging.getLogger(__name__)

MAX_CARD_CHARS = 12_000  # keep prompts small for local models (~3-4k tokens)
MAX_BENCHMARKS = 10
# Length caps for LLM text derived from third-party cards (prompt-injection surface towards client
# models; local models ignore "max N" in the schema, so limits are enforced in code).
MAX_SUMMARY_CHARS = 300
MAX_LIST_ITEMS = 5
MAX_ITEM_CHARS = 120
MAX_BENCHMARK_NAME_CHARS = 60


def clip(text: str | None, limit: int) -> str | None:
    """Truncate `text` to `limit` characters (with an ellipsis); None stays None."""
    if text is None or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def clip_list(
    items: list[str], max_items: int = MAX_LIST_ITEMS, max_chars: int = MAX_ITEM_CHARS
) -> list[str]:
    """At most `max_items` entries, each clipped to `max_chars`."""
    return [clip(i, max_chars) for i in items[:max_items]]


SYSTEM = (
    "You are a precise ML model analyst. Extract facts from a Hugging Face model card. "
    "Only report what the card states; use null or empty lists when information is missing. "
    "Keep list items short (max 8 words). "
    "For reported_benchmarks, copy scores of the named model only, exactly as printed; "
    "comparison tables often list other models in other columns - ignore those. "
    "Never invent or compute numbers."
)


async def extract_card_insights(
    repo_id: str,
    repo_card: str,
    base_card: str = "",
    run_id: str | None = None,
    client: AsyncOpenAI | None = None,
    settings: Settings | None = None,
) -> ModelCardInsights:
    """Combine deterministic card metadata with an LLM extraction of the card prose.

    Uses the base model's card if given, else the repo card unless it is conversion
    boilerplate. If the LLM is unavailable, the deterministic part is still returned
    (`llm_used=False`) so callers can degrade gracefully.
    """
    meta = card_metadata(repo_card)
    if base_card and not meta["license"]:
        meta["license"] = card_metadata(base_card)["license"]
    insights = ModelCardInsights(repo_id=repo_id, **meta)

    if base_card.strip():
        source, text = "base_model", clean_card_text(base_card, MAX_CARD_CHARS)
    elif repo_card.strip() and not is_boilerplate(repo_card):
        source, text = "repo", clean_card_text(repo_card, MAX_CARD_CHARS)
    else:
        insights.notes.append("No informative model card available; metadata only.")
        return insights

    subject = meta["base_model"] if source == "base_model" and meta["base_model"] else repo_id
    user = f"Model: {subject}\n\nModel card:\n{text}"
    try:
        extraction = await structured_call(
            "analyst", CardExtraction, SYSTEM, user, run_id=run_id, client=client, settings=settings
        )
    except LLMError as exc:
        log.warning("Analyst LLM call failed for %s: %s", repo_id, exc)
        insights.card_source = source
        insights.notes.append("LLM unavailable; card prose not analyzed.")
        return insights

    kept, dropped = verify_benchmarks(extraction.reported_benchmarks, text)
    if dropped:
        insights.notes.append(f"Dropped benchmarks not found verbatim in the card: {', '.join(dropped)}.")
    if len(kept) > MAX_BENCHMARKS:  # local models ignore "max N" in the schema; enforce it here
        insights.notes.append(f"Kept the first {MAX_BENCHMARKS} of {len(kept)} reported benchmarks.")
        kept = dict(list(kept.items())[:MAX_BENCHMARKS])
    return insights.model_copy(
        update={
            "summary": clip(extraction.summary, MAX_SUMMARY_CHARS),
            "primary_use_cases": clip_list(extraction.primary_use_cases),
            "languages": clip_list(extraction.languages, max_items=10, max_chars=40),
            "strengths": clip_list(extraction.strengths),
            "caveats": clip_list(extraction.caveats),
            "reported_benchmarks": {clip(k, MAX_BENCHMARK_NAME_CHARS): v for k, v in kept.items()},
            "card_source": source,
            "llm_used": True,
        }
    )
