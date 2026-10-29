"""Resource `runs://recent`: recent workflow runs from the audit log."""

from sqlmodel import select

from mlx_model_scout.db.models import WorkflowRun
from mlx_model_scout.db.session import session_scope


def recent_runs(limit: int = 20) -> list[dict]:
    """Most recent workflow runs (status, inputs, timestamps), newest first."""
    with session_scope() as s:
        rows = s.exec(select(WorkflowRun).order_by(WorkflowRun.started_at.desc()).limit(limit)).all()
        return [r.model_dump(mode="json") for r in rows]
