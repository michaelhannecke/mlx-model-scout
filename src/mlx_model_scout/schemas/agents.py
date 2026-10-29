"""Structured outputs of the Scout and Advisor agents.

Two layers per agent:
- `*Draft` / `CandidateList`: what the LLM returns. Small, flat, and WITHOUT any
  memory/speed numbers (the LLM never produces them).
- Result models: the validated outcome after code has checked repo ids against the
  input and attached the deterministic numbers.
"""

from typing import Literal

from pydantic import BaseModel, Field

from mlx_model_scout.schemas.fit import FitEstimate, RecommendationReport


class Candidate(BaseModel):
    """One model the Scout proposes for the use case."""

    repo_id: str = Field(description="Exact repo id copied from the provided list.")
    reason: str = Field(description="Why it fits the use case, max 15 words.")


class CandidateList(BaseModel):
    """LLM output of the Scout agent."""

    candidates: list[Candidate] = Field(default_factory=list, description="Best first.")


class ScoutResult(BaseModel):
    """Validated Scout outcome."""

    candidates: list[Candidate]
    pool_size: int = Field(description="Repos left after the deterministic pre-filter.")
    llm_used: bool
    notes: list[str] = Field(default_factory=list)


class RankedItem(BaseModel):
    """One entry of the Advisor's ranking (no numbers: they are attached by code)."""

    repo_id: str = Field(description="Exact repo id copied from the provided candidates.")
    rationale: str = Field(description="Why this rank for the use case, max 30 words, no numbers.")
    trade_offs: list[str] = Field(default_factory=list, description="Max 3 short phrases.")


class AdvisorDraft(BaseModel):
    """LLM output of the Advisor agent."""

    ranking: list[RankedItem] = Field(default_factory=list, description="Best first, max 5.")
    summary: str = Field(description="Two sentences for the user, no numbers.")


# --- Orchestrator (Plan-and-Execute + human-in-the-loop) ------------------------


class ExecutionPlan(BaseModel):
    """LLM output of the planning step: how to search the Hub for this use case."""

    search_terms: list[str] = Field(
        default_factory=list, description="1-3 short Hub search terms, e.g. 'coder', 'instruct', 'vl'."
    )
    needs_vision: bool = Field(default=False, description="True only if the use case involves images.")
    rationale: str = Field(default="", description="One sentence why these terms.")


Checkpoint = Literal["select_candidates", "approve_report"]
RunStatus = Literal["needs_user_input", "completed", "aborted"]


class CandidateFit(BaseModel):
    """One row of the candidate table shown at checkpoint 1 (numbers from the calculator)."""

    repo_id: str
    reason: str = Field(description="Why it was proposed (scout, user or no-fit alternative).")
    fit: FitEstimate


class RecommendationRun(BaseModel):
    """Response of `run_recommendation` / `continue_recommendation`.

    When `status == "needs_user_input"`, show `message` and the data to the user, collect a
    decision and call `continue_recommendation` with the same `run_id`.
    """

    run_id: str
    status: RunStatus
    checkpoint: Checkpoint | None = None
    message: str
    plan: ExecutionPlan | None = None
    candidates: list[CandidateFit] = Field(default_factory=list)
    alternatives: list[CandidateFit] = Field(
        default_factory=list, description="No-fit branch: smaller variants or shorter context that fit."
    )
    report: RecommendationReport | None = None
    report_id: int | None = None
    notes: list[str] = Field(default_factory=list)


class RunState(BaseModel):
    """Server-side state of a recommendation run between the two tool calls (persisted as JSON)."""

    use_case: str
    profile_id: str
    context_tokens: int
    checkpoint: Checkpoint | None = None
    plan: ExecutionPlan | None = None
    candidates: list[CandidateFit] = Field(default_factory=list)
    alternatives: list[CandidateFit] = Field(default_factory=list)
    report: RecommendationReport | None = None
    revisions: int = 0
    notes: list[str] = Field(default_factory=list)
