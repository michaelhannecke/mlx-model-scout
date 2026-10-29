"""Deterministic model-card handling (no LLM).

MLX conversion repos (e.g. mlx-community) ship boilerplate cards: YAML front
matter, one "converted to MLX format from <base>" line and a usage snippet.
The useful prose lives in the BASE model's card. This module parses the front
matter, finds the base model, cleans card text for a small LLM prompt and
verifies LLM-reported benchmark numbers against the card text.
"""

import re
from typing import Any

import yaml

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_REPO_ID = r"[\w.-]+/[\w.-]+"
# Body line written by mlx-lm / mlx-vlm conversion; more reliable than front-matter base_model
# (e.g. gemma-3-27b-it-4bit lists the -pt model in its front matter but was converted from -it).
_CONVERTED_FROM = re.compile(
    rf"converted to MLX format from\s+\[`?({_REPO_ID})`?\]"
    rf"|converted to MLX format from\s+.*?huggingface\.co/({_REPO_ID})",
    re.IGNORECASE,
)
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
# Only real HTML layout tags: cards also mention pseudo-tags like <think> that carry meaning.
_HTML_TAG = re.compile(
    r"</?(?:a|b|br|center|details|div|em|figure|h[1-6]|hr|i|img|li|ol|p|picture|source|span|strong"
    r"|summary|sup|sub|table|tbody|td|th|thead|tr|u|ul)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# "<think>" is a special token for reasoning models (Qwen3): inside a prompt it is tokenized as a
# control token and cannot be echoed back as text, which garbles extracted caveats. Neutralize it.
_SPECIAL_TAG = re.compile(r"<(/?)(think|tool_call|tool_response)>", re.IGNORECASE)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BLANK_LINES = re.compile(r"\n{3,}")


def split_front_matter(card: str) -> tuple[dict[str, Any], str]:
    """Return (front matter as dict, body). Malformed YAML yields an empty dict."""
    match = _FRONT_MATTER.match(card)
    if not match:
        return {}, card
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), card[match.end() :]


def _first_str(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value) if value else None


def card_metadata(card: str) -> dict[str, str | None]:
    """Structured facts from a card: license, base_model, pipeline_tag."""
    meta, body = split_front_matter(card)
    match = _CONVERTED_FROM.search(body)
    converted_from = (match.group(1) or match.group(2)) if match else None
    return {
        "license": _first_str(meta.get("license")),
        "base_model": converted_from or _first_str(meta.get("base_model")),
        "pipeline_tag": _first_str(meta.get("pipeline_tag")),
    }


def clean_card_text(card: str, max_chars: int) -> str:
    """Strip front matter, code blocks, HTML and images; truncate for a small prompt."""
    _, body = split_front_matter(card)
    for pattern in (_CODE_BLOCK, _HTML_COMMENT, _IMAGE, _HTML_TAG):
        body = pattern.sub("", body)
    body = _SPECIAL_TAG.sub(r"[\1\2]", body)
    body = _BLANK_LINES.sub("\n\n", body).strip()
    return body[:max_chars]


def is_boilerplate(card: str) -> bool:
    """True for conversion-only cards that carry no information beyond metadata."""
    body = clean_card_text(card, 10_000)
    return len(body) < 1_500 and "converted to mlx format" in body.lower()


def _number_in_text(value: float, text: str) -> bool:
    candidates = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    return any(re.search(rf"(?<![\d.]){re.escape(c)}(?!\d)", text) for c in candidates)


def verify_benchmarks(benchmarks: dict[str, float], card_text: str) -> tuple[dict[str, float], list[str]]:
    """Keep only benchmark scores whose number literally appears in the card text.

    Returns (kept, dropped names). Guards against hallucinated numbers; it cannot
    detect a real number attributed to the wrong benchmark or model column.
    """
    kept: dict[str, float] = {}
    dropped: list[str] = []
    for name, value in benchmarks.items():
        if _number_in_text(value, card_text):
            kept[name] = value
        else:
            dropped.append(name)
    return kept, dropped
