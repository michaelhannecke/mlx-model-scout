"""Deterministic helpers for the "nothing fits" branch of the recommend workflow.

Two kinds of alternatives:
- a shorter context for the same model (only the KV cache shrinks), and
- smaller quantizations of the same model (sibling repos such as `-8bit` -> `-4bit`).
"""

import re

from mlx_model_scout.app.fit_calculator import estimate_fit
from mlx_model_scout.app.model_spec import size_gb_from_name
from mlx_model_scout.config.settings import Settings
from mlx_model_scout.schemas.hardware import Chip, HardwareProfile
from mlx_model_scout.schemas.model import ModelSpec

MIN_CONTEXT = 2048
# Trailing name parts that describe the quantization/format, not the model.
_QUANT_SUFFIX = re.compile(
    r"[-_](?:\d{1,2}[-_]?bits?|bf16|fp16|fp8|mxfp\d|nvfp\d|q\d|dwq|qat|awq|mlx|optiq)$", re.IGNORECASE
)


def base_name(repo_id: str) -> str:
    """Repo name without quantization suffixes, e.g. 'Qwen3-Coder-Next-6bit' -> 'Qwen3-Coder-Next'."""
    name = repo_id.split("/")[-1]
    while (stripped := _QUANT_SUFFIX.sub("", name)) != name:
        name = stripped
    return name


def smaller_siblings(repo_id: str, repo_ids: list[str]) -> list[str]:
    """Repos of the same author and base name whose name-based size is smaller, smallest last.

    Sorted largest first: the biggest variant that still fits usually has the best quality.
    """
    author = repo_id.split("/")[0]
    base = base_name(repo_id).lower()
    own = size_gb_from_name(repo_id)
    siblings = []
    for other in repo_ids:
        if other == repo_id or other.split("/")[0] != author or base_name(other).lower() != base:
            continue
        size = size_gb_from_name(other)
        if own is None or (size is not None and size < own):
            siblings.append((size or 0.0, other))
    return [rid for _, rid in sorted(siblings, reverse=True)]


def max_fitting_context(
    spec: ModelSpec, profile: HardwareProfile, chip: Chip | None, context_tokens: int, settings: Settings
) -> int | None:
    """Largest context (halving from `context_tokens`, >= MIN_CONTEXT) at which the model fits.

    None if it does not fit even at MIN_CONTEXT (weights alone too large) or memory is unknown.
    """
    ctx = context_tokens // 2
    while ctx >= MIN_CONTEXT:
        if estimate_fit(spec, profile, chip, ctx, settings).verdict in ("fits", "tight"):
            return ctx
        ctx //= 2
    return None
