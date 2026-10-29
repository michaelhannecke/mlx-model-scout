"""Deterministic derivation of a ModelSpec from a Hugging Face `config.json`.

Why not use the Hub's safetensors parameter count? For quantized MLX repos the
weights are packed into uint32 tensors plus scales/biases, so the reported
"parameter" count is wrong (roughly params * bits / 32 + overhead). We therefore
compute parameters from the architecture hyper-parameters.

Supported well: Llama/Mistral/Qwen/Gemma-style decoder-only transformers with
(G)QA attention and gated MLP, plus common MoE layouts (Mixtral, Qwen-MoE,
DeepSeek-style shared experts). Everything else gets `confidence="low"`.
"""

import re
from typing import Any

from mlx_model_scout.schemas.model import ModelSpec, QuantizationInfo

# Architectures whose KV cache is NOT described by the standard GQA formula.
_MLA_TYPES = {"deepseek_v2", "deepseek_v3", "kimi_k2", "glm4_moe_lite"}
_HYBRID_TYPES = {"jamba", "nemotron_h", "falcon_h1", "qwen3_next", "mamba", "mamba2", "lfm2"}

_EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
# Not preceded by a word char or dot: skips "8x7B" (Mixtral, 46.7B total) and "A3B" (active params).
_SIZE_IN_NAME = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
_ROUTER_PATH = re.compile(r"\.(gate|router|shared_expert_gate)$")

# transformers omits keys equal to the class default when saving config.json.
# Without these, e.g. Gemma 3 has no vocab_size and its tied lm_head is double-counted.
_TYPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "gemma3_text": {
        "vocab_size": 262_208,
        "tie_word_embeddings": True,
        "max_position_embeddings": 131_072,
        "sliding_window_pattern": 6,
    },
}


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Multimodal configs nest the LLM under `text_config`; merge it on top, then apply type defaults."""
    text = config.get("text_config")
    merged = {**config, **text} if isinstance(text, dict) else config
    return {**_TYPE_DEFAULTS.get(merged.get("model_type"), {}), **merged}


def _has_mixed_quantization(config: dict[str, Any], quant: QuantizationInfo) -> bool:
    """True if per-layer overrides (nested dicts in the quantization block) use other bits.

    MoE router overrides (e.g. `mlp.gate` at 8 bit) are ignored: they are too small to matter.
    """
    q = config.get("quantization") or {}
    return isinstance(q, dict) and any(
        isinstance(v, dict)
        and v.get("bits")
        and float(v["bits"]) != quant.bits
        and not _ROUTER_PATH.search(path)
        for path, v in q.items()
    )


def parse_quantization(config: dict[str, Any]) -> QuantizationInfo:
    """Read MLX quantization parameters; default to 16-bit if unquantized."""
    q = config.get("quantization") or config.get("quantization_config")
    if isinstance(q, dict) and q.get("bits"):
        return QuantizationInfo(bits=float(q["bits"]), group_size=q.get("group_size"), mode=q.get("mode"))
    dtype = str(config.get("torch_dtype") or config.get("dtype") or "bfloat16")
    return QuantizationInfo(bits=32.0 if "32" in dtype else 16.0)


def effective_bits(q: QuantizationInfo) -> float:
    """Bits per weight including quantization metadata overhead.

    Affine quantization stores an fp16 scale and bias per group (32 bits / group).
    MX formats store one 8-bit shared scale per group.
    """
    if q.bits >= 16 or not q.group_size:
        return q.bits
    if q.mode and q.mode.lower().startswith("mx"):
        return q.bits + 8 / q.group_size
    return q.bits + 32 / q.group_size


def _params_from_name(repo_id: str) -> float | None:
    match = _SIZE_IN_NAME.search(repo_id.split("/")[-1])
    return float(match.group(1)) if match else None


_BITS_IN_NAME = re.compile(r"(?<![a-z])(\d{1,2})[-_]?bit|mxfp(4)|(bf16|fp16)", re.IGNORECASE)


def size_gb_from_name(repo_id: str) -> float | None:
    """Rough weight size in GiB from the repo NAME only (e.g. 'Qwen3-30B-A3B-4bit' -> 15.7).

    Only for cheap pre-filtering before config.json is fetched; never for reported numbers.
    No bit suffix -> bf16 (16 bit). Quantized sizes include ~0.5 bit/weight group overhead.
    """
    params_b = _params_from_name(repo_id)
    if params_b is None:
        return None
    match = _BITS_IN_NAME.search(repo_id.split("/")[-1])
    bits = 16.0 if match is None or match.group(3) else float(match.group(1) or match.group(2))
    if bits < 16:
        bits += 0.5
    return params_b * 1e9 * bits / 8 / 2**30


def linear_attention_params(cfg: dict[str, Any]) -> int:
    """Params of one linear-attention (Gated DeltaNet) layer, Qwen3-Next / Qwen3.5 layout.

    Projections: qkv (h -> 2*key_dim + value_dim), z gate (h -> value_dim), a/b (h -> value
    heads each), out (value_dim -> h), plus a depthwise conv over the qkv channels.
    0 if the config lacks the `linear_*` keys.
    """
    h = cfg.get("hidden_size")
    keys = ("linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads", "linear_value_head_dim")
    if not isinstance(h, int) or not all(isinstance(cfg.get(k), int) for k in keys):
        return 0
    key_dim = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    value_dim = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    qkv = 2 * key_dim + value_dim
    conv = qkv * (cfg.get("linear_conv_kernel_dim") or 4)
    return h * qkv + h * value_dim + 2 * h * cfg["linear_num_value_heads"] + value_dim * h + conv


def attention_layout(cfg: dict[str, Any]) -> list[str] | None:
    """Per-layer attention kind: 'full', 'sliding', 'linear' or 'other'; None = all full (standard).

    Sources, in order: explicit `layer_types` (gpt-oss, Gemma 4), `sliding_window_pattern`
    (Gemma 3: every N-th layer is global) and `full_attention_interval` (Qwen3-Next: every
    N-th layer is full attention, the rest linear). A bare `sliding_window` without a pattern
    is ignored (e.g. Qwen2 sets it but disables it), so the estimate stays conservative.
    """
    layers = cfg.get("num_hidden_layers")
    if isinstance(cfg.get("layer_types"), list):
        mapping = {"full_attention": "full", "sliding_attention": "sliding", "linear_attention": "linear"}
        return [mapping.get(t, "other") for t in cfg["layer_types"]]
    if not isinstance(layers, int):
        return None
    pattern = cfg.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 1 and cfg.get("sliding_window"):
        return ["full" if (i + 1) % pattern == 0 else "sliding" for i in range(layers)]
    interval = cfg.get("full_attention_interval")
    if isinstance(interval, int) and interval > 1:
        return ["full" if (i + 1) % interval == 0 else "linear" for i in range(layers)]
    return None


def build_model_spec(repo_id: str, raw_config: dict[str, Any]) -> ModelSpec:
    """Derive a ModelSpec from config.json. Never raises on missing keys."""
    cfg = _text_config(raw_config)
    notes: list[str] = []
    model_type = cfg.get("model_type")
    archs = cfg.get("architectures") or []
    quant = parse_quantization(raw_config)

    h = cfg.get("hidden_size")
    layers = cfg.get("num_hidden_layers")
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim") or (h // n_heads if h and n_heads else None)
    vocab = cfg.get("vocab_size")
    inter = cfg.get("intermediate_size")
    max_ctx = cfg.get("max_position_embeddings")

    n_experts = next((cfg[k] for k in _EXPERT_KEYS if cfg.get(k)), None)
    is_moe = bool(n_experts and n_experts > 1)

    kv_supported = True
    confidence = "high"
    kinds = attention_layout(cfg)
    n_sliding = kinds.count("sliding") if kinds else 0
    n_linear = kinds.count("linear") if kinds else 0
    if model_type in _MLA_TYPES:
        kv_supported = False
        notes.append("MLA attention: standard KV formula not applicable.")
    if kinds and "other" in kinds:
        kv_supported = False
        notes.append("Unknown layer types (e.g. SSM/Mamba): KV cache not estimated.")
        confidence = "medium"
    elif model_type in _HYBRID_TYPES and not n_linear:
        kv_supported = False
        notes.append("Hybrid architecture (linear attention/SSM layers): KV cache not estimated.")
        confidence = "medium"
    if n_sliding:
        notes.append(
            f"{n_sliding} sliding-window layers: their KV cache is capped at {cfg['sliding_window']} tokens."
        )
    if n_linear:
        notes.append(f"{n_linear} linear-attention layers: constant-size state not counted (small).")
        confidence = "medium"
    if _has_mixed_quantization(raw_config, quant):
        notes.append(
            f"Per-layer quantization overrides: weights estimated at the base {quant.bits:g} bits; "
            "actual size may differ by up to ~10%."
        )
        confidence = "medium" if confidence == "high" else confidence

    total = active = None
    if all(isinstance(v, int) for v in (h, layers, n_heads, n_kv, head_dim, vocab)):
        embed = vocab * h
        lm_head = 0 if cfg.get("tie_word_embeddings", False) else vocab * h
        attn = h * n_heads * head_dim + 2 * h * n_kv * head_dim + n_heads * head_dim * h
        if cfg.get("attn_output_gate"):  # Qwen3-Next/3.5: q_proj also emits a gate (double width)
            attn += h * n_heads * head_dim
        # all attention/token-mixing params: full/sliding layers use `attn`, linear layers their own
        mixing = (layers - n_linear) * attn + n_linear * linear_attention_params(cfg)

        if is_moe:
            k = cfg.get("num_experts_per_tok") or 2
            moe_inter = cfg.get("moe_intermediate_size") or inter
            shared = 0
            if cfg.get("shared_expert_intermediate_size"):
                shared = 3 * h * cfg["shared_expert_intermediate_size"]
            elif cfg.get("n_shared_experts") and moe_inter:
                shared = 3 * h * moe_inter * cfg["n_shared_experts"]
            if moe_inter:
                expert = 3 * h * moe_inter
                router = h * n_experts
                dense_layers = int(cfg.get("first_k_dense_replace") or 0)
                moe_layers = layers - dense_layers
                dense_mlp = 3 * h * inter if inter else expert
                total = (
                    embed
                    + lm_head
                    + mixing
                    + dense_layers * dense_mlp
                    + moe_layers * (n_experts * expert + shared + router)
                )
                active = (
                    (lm_head or embed)
                    + mixing
                    + dense_layers * dense_mlp
                    + moe_layers * (k * expert + shared + router)
                )
                if dense_layers == 0 and (
                    cfg.get("mlp_only_layers") or cfg.get("decoder_sparse_step", 1) != 1
                ):
                    notes.append("Some layers may be dense; MoE param count approximated.")
                    confidence = "medium"
        elif inter:
            mlp = 3 * h * inter
            total = embed + lm_head + mixing + layers * mlp
            active = (lm_head or embed) + mixing + layers * mlp

    total_b = round(total / 1e9, 2) if total else None
    active_b = round(active / 1e9, 2) if active else None

    if total_b is None:
        total_b = _params_from_name(repo_id)
        active_b = None if is_moe else total_b
        confidence = "low"
        notes.append("Parameter count guessed from repo name; config incomplete or unsupported.")
    if not kv_supported:
        confidence = "low"

    return ModelSpec(
        repo_id=repo_id,
        architecture=archs[0] if archs else None,
        model_type=model_type,
        is_moe=is_moe,
        total_params_b=total_b,
        active_params_b=active_b,
        quantization=quant,
        num_layers=layers,
        hidden_size=h,
        num_attention_heads=n_heads,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        sliding_window=cfg.get("sliding_window") if n_sliding else None,
        num_sliding_layers=n_sliding,
        num_linear_layers=n_linear,
        full_attention_kv_heads=cfg.get("num_global_key_value_heads"),
        full_attention_head_dim=cfg.get("global_head_dim"),
        max_context=max_ctx,
        kv_cache_supported=kv_supported,
        confidence=confidence,
        notes=notes,
    )
