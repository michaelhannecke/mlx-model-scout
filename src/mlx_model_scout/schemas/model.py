"""Model metadata Pydantic models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


class HubModel(BaseModel):
    """Minimal listing entry from the Hugging Face Hub API."""

    repo_id: str
    author: str | None = None
    last_modified: datetime | None = None
    created_at: datetime | None = None
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    tags: list[str] = Field(default_factory=list)


class QuantizationInfo(BaseModel):
    """MLX quantization parameters (from config.json `quantization` block)."""

    bits: float = Field(description="Bits per weight; 16 for unquantized bf16/fp16.")
    group_size: int | None = Field(default=None, description="Affine quantization group size.")
    mode: str | None = Field(default=None, description="e.g. affine, mxfp4 (if present).")


class ModelSpec(BaseModel):
    """Architecture facts needed for fit estimation.

    Derived deterministically from config.json where possible. Fields the
    calculator cannot derive stay None and lower `confidence`.
    """

    repo_id: str
    architecture: str | None = None
    model_type: str | None = None
    is_moe: bool = False
    total_params_b: float | None = Field(default=None, description="Total params in billions.")
    active_params_b: float | None = Field(
        default=None,
        description=(
            "Params READ per decoded token in billions: active experts + attention + lm_head; "
            "excludes the input-embedding table (row lookup only). Drives the speed estimate."
        ),
    )
    quantization: QuantizationInfo
    weights_file_gb: float | None = Field(
        default=None,
        description="Actual size of the safetensors files in GiB (from the Hub); preferred for memory.",
    )
    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    sliding_window: int | None = Field(default=None, description="Window (tokens) of sliding-window layers.")
    num_sliding_layers: int = Field(default=0, description="Layers whose KV cache is capped at the window.")
    num_linear_layers: int = Field(
        default=0, description="Linear-attention layers (constant-size state, no growing KV cache)."
    )
    full_attention_kv_heads: int | None = Field(
        default=None, description="KV heads of full-attention layers if they differ (Gemma 4 global layers)."
    )
    full_attention_head_dim: int | None = Field(
        default=None, description="Head dim of full-attention layers if it differs (Gemma 4 global layers)."
    )
    max_context: int | None = None
    kv_cache_supported: bool = Field(
        default=True,
        description="False for architectures where the standard GQA KV formula does not apply.",
    )
    confidence: Confidence = "medium"
    notes: list[str] = Field(default_factory=list)


class CardExtraction(BaseModel):
    """LLM output of the Analyst agent: only what is NOT available as structured metadata.

    Kept small and flat on purpose: local models fail on deeply nested schemas.
    """

    summary: str | None = Field(default=None, description="One sentence: what the model is and is for.")
    primary_use_cases: list[str] = Field(default_factory=list, description="Max 5 short phrases.")
    languages: list[str] = Field(default_factory=list, description="Natural languages the card claims.")
    strengths: list[str] = Field(default_factory=list, description="Max 5, as stated in the card.")
    caveats: list[str] = Field(default_factory=list, description="Limitations/risks stated in the card.")
    reported_benchmarks: dict[str, float] = Field(
        default_factory=dict,
        description="Benchmark name -> score of THIS model exactly as printed in the card. Max 10.",
    )


class ModelCardInsights(CardExtraction):
    """Analyst result: deterministic card metadata plus the LLM extraction.

    `license`, `base_model` and `pipeline_tag` come from the card's YAML front matter,
    never from the LLM. Benchmarks whose value does not appear in the card are dropped.
    """

    repo_id: str
    base_model: str | None = None
    license: str | None = None
    pipeline_tag: str | None = None
    card_source: Literal["base_model", "repo", "none"] = Field(
        default="none", description="Which card the LLM read: the base model's (preferred) or the repo's."
    )
    llm_used: bool = Field(default=False, description="False if the LLM was unavailable or not needed.")
    notes: list[str] = Field(default_factory=list)
