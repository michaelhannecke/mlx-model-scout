"""MLX Model Scout: an MCP server that recommends Apple-MLX models for given hardware."""

__version__ = "0.1.0"


def main() -> None:
    """Entry point used by `[project.scripts]`."""
    from mlx_model_scout.server import main as _main

    _main()
