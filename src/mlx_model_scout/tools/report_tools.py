"""Tool implementations: persisting approved reports."""

import json

from mlx_model_scout.db.audit import log_hitl, tool_audit
from mlx_model_scout.db.models import ReportRow
from mlx_model_scout.db.session import session_scope


def save_report(report: dict, approved: bool, run_id: str | None = None) -> dict:
    """Persist a report; `approved` must reflect an explicit user decision."""
    with tool_audit("save_report", {"approved": approved}, run_id) as audit:
        log_hitl(run_id, "approve_report", options=["approve", "reject"], decision={"approved": approved})
        with session_scope() as s:
            row = ReportRow(run_id=run_id, report_json=json.dumps(report, default=str), approved=approved)
            s.add(row)
            s.flush()
            report_id = row.id
        audit["summary"] = f"report {report_id} approved={approved}"
        return {"report_id": report_id, "approved": approved}
