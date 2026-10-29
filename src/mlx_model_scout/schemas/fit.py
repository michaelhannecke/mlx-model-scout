"""Fit estimation and recommendation Pydantic models."""

from typing import Literal

from pydantic import BaseModel, Field

from mlx_model_scout.schemas.model import Confidence

FitVerdict = Literal["fits", "tight", "no_fit", "unknown"]


class FitEstimate(BaseModel):
    """Deterministic memory/speed estimate of one model on one hardware profile."""

    repo_id: str
    profile_id: str
    context_tokens: int
    weights_gb: float | None
    kv_cache_gb: float | None
    runtime_overhead_gb: float
    required_gb: float | None
    available_gb: float
    headroom_gb: float | None
    verdict: FitVerdict
    est_decode_tps: float | None = Field(
        default=None, description="Bandwidth-bound decode estimate (tokens/s); heuristic."
    )
    confidence: Confidence
    assumptions: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """One ranked entry in the Advisor report."""

    rank: int = Field(ge=1)
    repo_id: str
    fit: FitEstimate
    rationale: str
    trade_offs: list[str] = Field(default_factory=list)


class RecommendationReport(BaseModel):
    """Final structured output of the recommend workflow."""

    use_case: str
    profile_id: str
    context_tokens: int
    recommendations: list[Recommendation]
    excluded: dict[str, str] = Field(default_factory=dict, description="repo_id -> reason it was excluded")
    summary: str
    generated_by: Literal["llm", "deterministic"] = Field(
        default="deterministic", description="Who ranked: the Advisor LLM or the fallback ranking."
    )
    notes: list[str] = Field(default_factory=list, description="Validation corrections and fallbacks.")
