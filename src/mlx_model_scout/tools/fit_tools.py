"""Tool implementations: fit estimation and (mocked) benchmark/download."""

import json
import random

from mlx_model_scout.app.fit_calculator import estimate_fit
from mlx_model_scout.app.profiles import ProfileError, load_chips
from mlx_model_scout.db.audit import tool_audit
from mlx_model_scout.db.models import BenchmarkRow, FitEstimateRow
from mlx_model_scout.db.session import session_scope
from mlx_model_scout.schemas.fit import FitEstimate
from mlx_model_scout.tools.hardware_tools import resolve_profile
from mlx_model_scout.tools.hub_tools import analyze_model


async def estimate_model_fit(
    repo_id: str, profile_id: str, context_tokens: int | None = None, run_id: str | None = None
) -> FitEstimate:
    """Analyze a repo and estimate its fit on a profile."""
    args = {"repo_id": repo_id, "profile_id": profile_id, "context_tokens": context_tokens}
    with tool_audit("estimate_fit", args, run_id) as audit:
        profile = resolve_profile(profile_id)
        spec = await analyze_model(repo_id)
        chip = load_chips().get(profile.chip)
        if chip is None and profile.chip != "unknown":
            raise ProfileError(f"Chip '{profile.chip}' missing in chips.yaml")
        fit = estimate_fit(spec, profile, chip, context_tokens)
        with session_scope() as s:
            s.add(
                FitEstimateRow(
                    run_id=run_id,
                    repo_id=repo_id,
                    profile_id=profile.id,
                    context_tokens=fit.context_tokens,
                    verdict=fit.verdict,
                    estimate_json=fit.model_dump_json(),
                )
            )
        audit["summary"] = f"{fit.verdict}, req {fit.required_gb} / avail {fit.available_gb} GiB"
        return fit


async def run_benchmark_mock(repo_id: str, profile_id: str, context_tokens: int | None = None) -> dict:
    """MOCK benchmark: returns the estimate with jitter, clearly labelled.

    Interface is final; replace the body with a subprocess call to
    `mlx_lm.generate`/a benchmark script on the HOST (never inside a container,
    no Metal there) once real benchmarking is in scope.
    """
    fit = await estimate_model_fit(repo_id, profile_id, context_tokens)
    tps = fit.est_decode_tps
    result = {
        "source": "mock",
        "repo_id": repo_id,
        "profile_id": profile_id,
        "decode_tps": round(tps * random.uniform(0.85, 1.05), 1) if tps else None,
        "peak_memory_gb": fit.required_gb,
        "note": "Simulated from the fit estimate. Not a real measurement.",
    }
    with session_scope() as s:
        s.add(
            BenchmarkRow(
                repo_id=repo_id, profile_id=profile_id, source="mock", result_json=json.dumps(result)
            )
        )
    return result


def download_model_mock(repo_id: str, confirmed: bool = False) -> dict:
    """MOCK download. Requires explicit confirmation (HITL)."""
    if not confirmed:
        return {
            "status": "confirmation_required",
            "message": (
                f"Downloading {repo_id} may need tens of GB. "
                "Ask the user, then call again with confirmed=true."
            ),
        }
    return {"status": "mocked", "repo_id": repo_id, "note": "No files downloaded (advisory scope)."}
