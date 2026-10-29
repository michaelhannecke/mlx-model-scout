"""SQLite engine and session handling."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from mlx_model_scout.config.settings import get_settings
from mlx_model_scout.db import models  # noqa: F401  (registers tables)


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    """Create (once) the SQLite engine; creates the data directory and tables."""
    settings = get_settings()
    if url is None:
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        url = settings.db_url
    engine = create_engine(url, connect_args={"check_same_thread": False})
    use_wal = settings.db_wal

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
        cur = dbapi_conn.cursor()
        # WAL: readers (Datasette) don't block the server. WAL needs shared memory,
        # which does NOT work across separate VMs (Apple container = 1 VM per container).
        cur.execute(f"PRAGMA journal_mode={'WAL' if use_wal else 'DELETE'}")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    SQLModel.metadata.create_all(engine)
    if url == settings.db_url:  # only the configured file DB, not explicit URLs (e.g. in-memory)
        _restrict_permissions(settings.db_path)
    return engine


def _restrict_permissions(db_path: Path) -> None:
    """Make the DB owner-only (0600): the audit log holds prompts and use cases.

    Best effort: chmod is not supported on some mounts (e.g. virtiofs volumes in Apple container).
    """
    for path in (db_path, db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")):
        try:
            if path.is_file():
                path.chmod(0o600)
        except OSError:
            pass


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    with Session(engine or get_engine()) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
