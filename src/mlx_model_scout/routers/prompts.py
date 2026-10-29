"""Registers MCP prompts."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from mlx_model_scout.prompts.recommend import recommend_models_prompt


def register_prompts(mcp: FastMCP) -> None:
    @mcp.prompt(tags={"workflow", "hitl"})
    def recommend_models(
        use_case: Annotated[
            str, Field(description="What the model should do, e.g. 'German RAG over contracts'")
        ],
        profile_id: Annotated[str, Field(description="Hardware profile id or 'local'")] = "local",
        context_tokens: Annotated[int, Field(ge=256)] = 8192,
    ) -> str:
        """Agentic workflow: scan -> analyze -> fit -> user selects -> report -> user approves."""
        return recommend_models_prompt(use_case, profile_id, context_tokens)
