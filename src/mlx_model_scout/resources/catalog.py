"""Resource `catalog://models`: models analyzed so far (local database)."""

from sqlmodel import select

from mlx_model_scout.db.models import ModelSpecRow
from mlx_model_scout.db.session import session_scope
from mlx_model_scout.schemas.model import ModelSpec


def cached_specs(limit: int = 100) -> list[ModelSpec]:
    """Analyzed model specs from the DB, newest first."""
    with session_scope() as s:
        query = select(ModelSpecRow).where(ModelSpecRow.spec_json != "")  # card-only rows have no spec
        rows = s.exec(query.order_by(ModelSpecRow.updated_at.desc()).limit(limit)).all()
        return [ModelSpec.model_validate_json(r.spec_json) for r in rows]


def model_catalog() -> list[dict]:
    """Previously analyzed models as plain dicts."""
    return [s.model_dump() for s in cached_specs()]
