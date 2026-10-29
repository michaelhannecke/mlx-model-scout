"""Central configuration via pydantic-settings.

All values can be overridden through environment variables or a `.env` file
(see `.env.sample`). Variables are prefixed with `SCOUT_`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime settings for the MLX Model Scout MCP server."""

    model_config = SettingsConfigDict(
        env_prefix="SCOUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Server -----------------------------------------------------------
    server_name: str = "mlx-model-scout"
    server_version: str = "0.1.0"
    transport: Literal["stdio", "http"] = "stdio"
    http_host: str = "127.0.0.1"
    http_port: int = 8765
    # DNS-rebinding protection for the HTTP transport: only these Host headers (fnmatch patterns)
    # are accepted, and browser Origins must match them. Add a name here only behind a proxy.
    http_allowed_hosts: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*", "[::1]", "[::1]:*"]
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Local LLM (omlx or mlx_lm.server, OpenAI-compatible) -------------
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "not-needed"  # omlx requires a real key; mlx_lm.server ignores it
    llm_model: str = "Qwen3-30B-A3B-Instruct-2507-4bit"  # exact id from GET /v1/models
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 2  # retries on Pydantic validation failure

    # --- Hugging Face Hub --------------------------------------------------
    hf_api_base: str = "https://huggingface.co"
    hf_token: str | None = None  # optional, only raises rate limits
    hf_authors: list[str] = Field(default_factory=lambda: ["mlx-community"])
    hf_scan_limit: int = 50
    hf_timeout_s: float = 30.0

    # --- Persistence & logging --------------------------------------------
    db_path: Path = PROJECT_ROOT / "data" / "scout.db"
    db_wal: bool = True  # disable when the DB file is shared across VMs (Apple container)
    log_full_messages: bool = False  # False: store truncated prompts + SHA-256 hash only
    log_truncate_chars: int = 500
    log_retention_days: int = 30

    # --- Hardware / fit heuristics ----------------------------------------
    hardware_profiles_path: Path = PROJECT_ROOT / "config" / "hardware_profiles.yaml"
    chips_path: Path = PROJECT_ROOT / "config" / "chips.yaml"
    default_context_tokens: int = 8192
    runtime_overhead_gb: float = 1.5  # Python/MLX runtime, activations, buffers
    default_wired_fraction: float = 0.75  # used when no explicit wired limit is known
    # Realistic share of peak memory bandwidth during decode, calibrated on an M5 Pro (omlx):
    # dense Qwen3.8-27B ~0.83, MoE Qwen3-30B-A3B ~0.59 (many small expert matmuls + routing).
    bandwidth_efficiency_dense: float = 0.8
    bandwidth_efficiency_moe: float = 0.6
    tight_headroom_gb: float = 4.0  # below this headroom a fit is rated "tight"

    @property
    def db_url(self) -> str:
        """SQLAlchemy URL for the SQLite database."""
        return f"sqlite:///{self.db_path}"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
