"""Tool implementations: Hugging Face scanning and model analysis."""

import asyncio

from mlx_model_scout.agents.analyst import extract_card_insights
from mlx_model_scout.app.hf_client import HFClient, HubError
from mlx_model_scout.app.model_card import card_metadata
from mlx_model_scout.app.model_spec import build_model_spec
from mlx_model_scout.config.settings import get_settings
from mlx_model_scout.db.audit import tool_audit
from mlx_model_scout.db.models import HubModelRow, ModelSpecRow, Scan, utcnow
from mlx_model_scout.db.session import session_scope
from mlx_model_scout.schemas.model import HubModel, ModelCardInsights, ModelSpec


async def scan_mlx_models(
    search: str | None = None,
    limit: int | None = None,
    authors: list[str] | None = None,
    run_id: str | None = None,
) -> dict:
    """Scan HF authors (parallel) for recent MLX repos and record new ones."""
    authors = authors or get_settings().hf_authors
    args = {"search": search, "limit": limit, "authors": authors}
    with tool_audit("scan_mlx_models", args, run_id) as audit:
        async with HFClient() as hf:
            results = await asyncio.gather(*(hf.list_models(a, search=search, limit=limit) for a in authors))
        models: list[HubModel] = [m for batch in results for m in batch]
        new_ids = _persist_scan(authors, models)
        audit["summary"] = f"{len(models)} models, {len(new_ids)} new"
        return {
            "total": len(models),
            "new_since_last_scan": new_ids,
            "models": [m.model_dump(mode="json") for m in models],
        }


def _persist_scan(authors: list[str], models: list[HubModel]) -> list[str]:
    new_ids: list[str] = []
    with session_scope() as s:
        for m in models:
            row = s.get(HubModelRow, m.repo_id)
            if row is None:
                new_ids.append(m.repo_id)
                row = HubModelRow(repo_id=m.repo_id, author=m.author)
            row.pipeline_tag = m.pipeline_tag
            row.last_modified = m.last_modified
            row.downloads = m.downloads
            row.last_seen_at = utcnow()
            s.add(row)
        for a in authors:
            n = sum(1 for m in models if m.author == a)
            s.add(Scan(author=a, n_models=n, n_new=sum(1 for i in new_ids if i.startswith(f"{a}/"))))
    return new_ids


async def analyze_model(repo_id: str) -> ModelSpec:
    """Fetch config.json and derive a ModelSpec deterministically (cached in DB).

    Also records the actual safetensors size, which the calculator prefers for memory.
    """
    with tool_audit("analyze_model", {"repo_id": repo_id}) as audit:
        async with HFClient() as hf:
            config, size = await asyncio.gather(hf.get_config(repo_id), hf.get_weights_size(repo_id))
        spec = build_model_spec(repo_id, config)
        if size:
            spec.weights_file_gb = round(size / 2**30, 2)
        with session_scope() as s:
            row = s.get(ModelSpecRow, repo_id) or ModelSpecRow(repo_id=repo_id, spec_json="")
            row.spec_json = spec.model_dump_json()
            row.updated_at = utcnow()
            s.add(row)
        audit["summary"] = f"{spec.total_params_b}B, {spec.quantization.bits}bit, {spec.confidence}"
        return spec


async def analyze_model_card(
    repo_id: str, refresh: bool = False, run_id: str | None = None
) -> ModelCardInsights:
    """Analyst agent on the model card (base model's card preferred); cached in DB."""
    with tool_audit("analyze_model_card", {"repo_id": repo_id, "refresh": refresh}, run_id) as audit:
        if not refresh:
            with session_scope() as s:
                row = s.get(ModelSpecRow, repo_id)
                if row and row.card_insights_json:
                    audit["summary"] = "cached"
                    return ModelCardInsights.model_validate_json(row.card_insights_json)

        notes: list[str] = []
        async with HFClient() as hf:
            repo_card = await hf.get_model_card(repo_id)
            base_model = card_metadata(repo_card)["base_model"]
            base_card = ""
            if base_model and base_model != repo_id:
                try:
                    base_card = await hf.get_model_card(base_model)
                except HubError as exc:  # e.g. gated base model without accepted license
                    notes.append(f"Base model card unavailable ({exc}).")

        insights = await extract_card_insights(repo_id, repo_card, base_card, run_id=run_id)
        insights.notes.extend(notes)
        if insights.llm_used:  # don't cache degraded results; retry when the LLM is back
            with session_scope() as s:
                row = s.get(ModelSpecRow, repo_id) or ModelSpecRow(repo_id=repo_id, spec_json="")
                row.card_insights_json = insights.model_dump_json()
                row.updated_at = utcnow()
                s.add(row)
        audit["summary"] = f"source={insights.card_source}, llm={insights.llm_used}"
        return insights
