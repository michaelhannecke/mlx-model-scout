"""SQLModel tables.

Two groups:
- Domain data: scans, hub models, derived specs, fit estimates, reports, (mock) benchmarks
- Audit/observability: workflow runs, LLM calls, tool calls, human-in-the-loop decisions

JSON payloads are stored as TEXT (serialized Pydantic models) to keep the
schema stable while the Pydantic models evolve.
"""

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- Domain ---------------------------------------------------------------


class Scan(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    author: str = Field(index=True)
    started_at: datetime = Field(default_factory=utcnow)
    n_models: int = 0
    n_new: int = 0


class HubModelRow(SQLModel, table=True):
    __tablename__ = "hub_models"

    repo_id: str = Field(primary_key=True)
    author: str | None = Field(default=None, index=True)
    pipeline_tag: str | None = None
    last_modified: datetime | None = None
    downloads: int | None = None
    first_seen_at: datetime = Field(default_factory=utcnow)
    last_seen_at: datetime = Field(default_factory=utcnow)


class ModelSpecRow(SQLModel, table=True):
    __tablename__ = "model_specs"

    repo_id: str = Field(primary_key=True)
    spec_json: str
    card_insights_json: str | None = None
    updated_at: datetime = Field(default_factory=utcnow)


class FitEstimateRow(SQLModel, table=True):
    __tablename__ = "fit_estimates"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str | None = Field(default=None, index=True)
    repo_id: str = Field(index=True)
    profile_id: str = Field(index=True)
    context_tokens: int
    verdict: str
    estimate_json: str
    created_at: datetime = Field(default_factory=utcnow)


class ReportRow(SQLModel, table=True):
    __tablename__ = "reports"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str | None = Field(default=None, index=True)
    report_json: str
    approved: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class RunStateRow(SQLModel, table=True):
    """State of a `run_recommendation` run between its tool calls (HITL checkpoints)."""

    __tablename__ = "run_states"

    run_id: str = Field(primary_key=True)
    checkpoint: str | None = None
    state_json: str
    updated_at: datetime = Field(default_factory=utcnow)


class BenchmarkRow(SQLModel, table=True):
    __tablename__ = "benchmarks"

    id: int | None = Field(default=None, primary_key=True)
    repo_id: str = Field(index=True)
    profile_id: str
    source: str = "mock"  # "mock" until real mlx-lm benchmarking is implemented
    result_json: str
    created_at: datetime = Field(default_factory=utcnow)


# --- Audit / observability -----------------------------------------------


class WorkflowRun(SQLModel, table=True):
    __tablename__ = "workflow_runs"

    id: str = Field(primary_key=True)  # uuid4 hex
    workflow: str
    input_json: str | None = None
    status: str = "running"  # running | succeeded | failed | aborted
    error: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class LLMCall(SQLModel, table=True):
    __tablename__ = "llm_calls"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str | None = Field(default=None, index=True)
    agent: str = Field(index=True)
    model: str
    messages_sha256: str
    messages: str  # full or truncated, depending on SCOUT_LOG_FULL_MESSAGES
    response: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    attempt: int = 1
    status: str = "ok"  # ok | validation_error | error
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow, index=True)


class ToolCall(SQLModel, table=True):
    __tablename__ = "tool_calls"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str | None = Field(default=None, index=True)
    tool: str = Field(index=True)
    arguments_json: str
    result_summary: str | None = None
    duration_ms: int | None = None
    status: str = "ok"
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow, index=True)


class HITLDecision(SQLModel, table=True):
    __tablename__ = "hitl_decisions"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str | None = Field(default=None, index=True)
    step: str
    options_json: str
    decision_json: str
    created_at: datetime = Field(default_factory=utcnow)
