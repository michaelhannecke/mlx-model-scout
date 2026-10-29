"""Registers MCP resources (read-only context for agents). Keep thin: delegate to `resources/`."""

from fastmcp import FastMCP

from mlx_model_scout.resources import catalog, hardware, runs


def register_resources(mcp: FastMCP) -> None:
    @mcp.resource("hardware://profiles", mime_type="application/json")
    def hardware_profiles() -> list[dict]:
        """All hardware profiles (configured + detected 'local')."""
        return hardware.hardware_profiles()

    @mcp.resource("hardware://chips", mime_type="application/json")
    def chips() -> dict:
        """Apple Silicon reference data (memory bandwidth) used for speed estimates."""
        return hardware.chips()

    @mcp.resource("catalog://models", mime_type="application/json")
    def model_catalog() -> list[dict]:
        """Previously analyzed models (ModelSpec) from the local database."""
        return catalog.model_catalog()

    @mcp.resource("runs://recent", mime_type="application/json")
    def recent_runs() -> list[dict]:
        """Recent workflow runs (audit log)."""
        return runs.recent_runs()
