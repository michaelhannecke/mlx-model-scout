"""FastMCP server entry point: creates the app and registers routers."""

from fastmcp import FastMCP

from mlx_model_scout.config.settings import Settings, get_settings
from mlx_model_scout.db.session import get_engine
from mlx_model_scout.routers.prompts import register_prompts
from mlx_model_scout.routers.resources import register_resources
from mlx_model_scout.routers.tools import register_tools
from mlx_model_scout.utils.logging import configure_logging

INSTRUCTIONS = """MLX Model Scout recommends Apple-MLX models for given hardware.
Numbers (memory, tok/s) come from deterministic tools; never invent or recompute them.
Use the `recommend_models` prompt for the full workflow, which requires user confirmation steps."""


def create_app() -> FastMCP:
    """Build the FastMCP app with all tools, resources and prompts registered."""
    settings = get_settings()
    mcp = FastMCP(name=settings.server_name, version=settings.server_version, instructions=INSTRUCTIONS)
    register_tools(mcp)
    register_resources(mcp)
    register_prompts(mcp)
    return mcp


def http_security(settings: Settings) -> dict:
    """Host/Origin validation for the HTTP transport (DNS-rebinding protection).

    The HTTP transport has no authentication and is meant for localhost only. Without this guard a
    web page could reach it through DNS rebinding (its own domain resolving to 127.0.0.1) and call
    tools. fastmcp's guard is off by default; strict mode answers 421 for foreign Host headers and
    403 for foreign browser Origins.
    """
    return {
        "host_origin_protection": True,
        "allowed_hosts": settings.http_allowed_hosts,
        "allowed_origins": [f"http://{h}" for h in settings.http_allowed_hosts],
    }


def main() -> None:
    """Console entry point (`uv run mlx-model-scout`)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    get_engine()  # create DB + tables early so failures surface at startup
    app = create_app()
    if settings.transport == "http":
        app.run(transport="http", host=settings.http_host, port=settings.http_port, **http_security(settings))
    else:
        app.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
