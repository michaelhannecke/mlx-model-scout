# OCI image for the MLX Model Scout MCP server (HTTP transport).
# Builds with Apple `container build` and with `docker build` alike.
#
# NOTE: The container never runs MLX. It has no Metal/GPU access (Linux VM).
# The local LLM (omlx / mlx_lm.server) runs on the macOS HOST; the server reaches it
# via SCOUT_LLM_BASE_URL. Hardware auto-detection is unavailable inside the
# container -> use configured profiles (config/hardware_profiles.yaml).

FROM python:3.14-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

# UV_PYTHON: use the base image's interpreter. It overrides .python-version, which pins
# exactly 3.14.0 (from `uv init --python 3.14.0`) while the slim image ships a newer 3.14.x.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (layer cache)
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Project
COPY src ./src
COPY config ./config
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 10001 scout && mkdir -p /data && chown scout /data
USER scout

ENV SCOUT_TRANSPORT=http \
    SCOUT_HTTP_HOST=0.0.0.0 \
    SCOUT_HTTP_PORT=8765 \
    SCOUT_DB_PATH=/data/scout.db \
    SCOUT_HARDWARE_PROFILES_PATH=/app/config/hardware_profiles.yaml \
    SCOUT_CHIPS_PATH=/app/config/chips.yaml \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8765
VOLUME ["/data"]

CMD ["mlx-model-scout"]
