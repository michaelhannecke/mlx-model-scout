"""Deterministic memory-fit and speed estimation.

The LLM never does this arithmetic. Units:
- memory sizes in GiB (2**30 bytes), matching how Apple advertises unified memory
- bandwidth in GB/s (10**9 bytes/s), matching Apple's spec sheets

Formulas
--------
weights   = total_params * effective_bits / 8
kv_cache  = 2 (K,V) * layers * kv_heads * head_dim * context * kv_bytes_per_elem
required  = weights + kv_cache + runtime_overhead
available = min(wired_limit, unified_memory - reserved) per node, * nodes
decode    ~ bandwidth * efficiency / (active_weight_bytes + kv_cache_bytes)
            (efficiency calibrated separately for dense and MoE models)
            (decode is memory-bandwidth bound: every generated token reads all
            active weights plus the KV cache once)
"""

from mlx_model_scout.app.model_spec import effective_bits
from mlx_model_scout.config.settings import Settings, get_settings
from mlx_model_scout.schemas.fit import FitEstimate, FitVerdict
from mlx_model_scout.schemas.hardware import Chip, HardwareProfile
from mlx_model_scout.schemas.model import ModelSpec

GIB = 1024**3
KV_BYTES_PER_ELEM = 2  # fp16/bf16 KV cache (MLX default, no KV quantization)


def weights_bytes(spec: ModelSpec) -> float | None:
    """Memory for the model weights: actual file size if known, else params x effective bits.

    The file size is exact (MLX loads the safetensors as stored) and also covers vision towers
    and per-layer mixed precision, which the parameter-based formula misses.
    """
    if spec.weights_file_gb:
        return spec.weights_file_gb * GIB
    if spec.total_params_b is None:
        return None
    return spec.total_params_b * 1e9 * effective_bits(spec.quantization) / 8


def active_weights_bytes(spec: ModelSpec) -> float | None:
    """Bytes read per decoded token (active params only for MoE)."""
    params = spec.active_params_b or (None if spec.is_moe else spec.total_params_b)
    if params is None:
        return None
    return params * 1e9 * effective_bits(spec.quantization) / 8


def kv_cache_bytes(spec: ModelSpec, context_tokens: int) -> float | None:
    """KV cache size for a single sequence at the given context length.

    Per layer `2 (K,V) * kv_heads * head_dim * tokens * bytes`, where tokens = context for
    full-attention layers and min(context, window) for sliding-window layers (MLX uses a
    rotating cache there). Linear-attention layers keep a small constant state: not counted.
    """
    if not spec.kv_cache_supported:
        return None
    if not all((spec.num_layers, spec.num_kv_heads, spec.head_dim)):
        return None
    n_full = spec.num_layers - spec.num_sliding_layers - spec.num_linear_layers
    full_kv = spec.full_attention_kv_heads or spec.num_kv_heads
    full_dim = spec.full_attention_head_dim or spec.head_dim
    per_token_full = 2 * full_kv * full_dim * KV_BYTES_PER_ELEM
    per_token_sliding = 2 * spec.num_kv_heads * spec.head_dim * KV_BYTES_PER_ELEM
    sliding_tokens = min(context_tokens, spec.sliding_window or context_tokens)
    return (
        n_full * per_token_full * context_tokens
        + spec.num_sliding_layers * per_token_sliding * sliding_tokens
    )


def available_gb(profile: HardwareProfile, settings: Settings) -> tuple[float, list[str]]:
    """Memory usable by MLX across all nodes, plus the assumptions made."""
    assumptions: list[str] = []
    if profile.wired_limit_gb:
        wired = profile.wired_limit_gb
        assumptions.append(f"GPU wired limit {wired:.1f} GiB per node (explicit).")
    else:
        wired = profile.unified_memory_gb * settings.default_wired_fraction
        assumptions.append(
            f"GPU wired limit assumed {settings.default_wired_fraction:.0%} of unified memory "
            f"({wired:.1f} GiB); raise via `sudo sysctl iogpu.wired_limit_mb=...`."
        )
    per_node = min(wired, profile.unified_memory_gb - profile.reserved_gb)
    if profile.reserved_gb:
        assumptions.append(f"{profile.reserved_gb:.0f} GiB reserved per node for OS/apps.")
    if profile.nodes > 1:
        assumptions.append(
            f"Cluster of {profile.nodes} nodes: memory summed, assumes model sharding "
            "(e.g. pipeline/tensor parallel); no speed estimate."
        )
    return per_node * profile.nodes, assumptions


def estimate_fit(
    spec: ModelSpec,
    profile: HardwareProfile,
    chip: Chip | None,
    context_tokens: int | None = None,
    settings: Settings | None = None,
) -> FitEstimate:
    """Estimate whether `spec` fits on `profile` and how fast it decodes."""
    settings = settings or get_settings()
    ctx = context_tokens or settings.default_context_tokens
    assumptions: list[str] = []

    if spec.max_context and ctx > spec.max_context:
        assumptions.append(f"Requested context {ctx} exceeds model maximum {spec.max_context}.")

    w = weights_bytes(spec)
    if spec.weights_file_gb:
        assumptions.append("Weights: actual download size of the safetensors files.")
    elif w is not None:
        assumptions.append("Weights: computed from config.json (download size unavailable).")
    kv = kv_cache_bytes(spec, ctx)
    avail, avail_notes = available_gb(profile, settings)
    assumptions.extend(avail_notes)
    assumptions.append(f"Runtime overhead {settings.runtime_overhead_gb:.1f} GiB.")
    if kv is None:
        assumptions.append("KV cache not estimated (unsupported architecture or missing config).")

    weights_gb = w / GIB if w is not None else None
    kv_gb = kv / GIB if kv is not None else None

    required = None
    if weights_gb is not None:
        required = weights_gb + (kv_gb or 0.0) + settings.runtime_overhead_gb

    verdict: FitVerdict
    headroom = None
    if required is None:
        verdict = "unknown"
    else:
        headroom = avail - required
        if headroom < 0:
            verdict = "no_fit"
        elif headroom < settings.tight_headroom_gb:
            verdict = "tight"
        else:
            verdict = "fits"

    tps = None
    active = active_weights_bytes(spec)
    if profile.nodes == 1 and chip and chip.memory_bandwidth_gbs and active:
        bytes_per_token = active + (kv or 0.0)
        efficiency = settings.bandwidth_efficiency_moe if spec.is_moe else settings.bandwidth_efficiency_dense
        tps = chip.memory_bandwidth_gbs * 1e9 * efficiency / bytes_per_token
        assumptions.append(
            f"Decode estimate: {chip.memory_bandwidth_gbs:.0f} GB/s x {efficiency:.0%} efficiency "
            f"({'MoE' if spec.is_moe else 'dense'}), bandwidth-bound."
        )
        if not chip.verified:
            assumptions.append(f"Chip bandwidth for {chip.key} is unverified.")
    elif profile.nodes == 1:
        assumptions.append("No speed estimate (chip bandwidth or active params unknown).")

    confidence = spec.confidence
    if kv is None and confidence == "high":
        confidence = "medium"

    return FitEstimate(
        repo_id=spec.repo_id,
        profile_id=profile.id,
        context_tokens=ctx,
        weights_gb=round(weights_gb, 2) if weights_gb is not None else None,
        kv_cache_gb=round(kv_gb, 2) if kv_gb is not None else None,
        runtime_overhead_gb=settings.runtime_overhead_gb,
        required_gb=round(required, 2) if required is not None else None,
        available_gb=round(avail, 2),
        headroom_gb=round(headroom, 2) if headroom is not None else None,
        verdict=verdict,
        est_decode_tps=round(tps, 1) if tps is not None else None,
        confidence=confidence,
        assumptions=assumptions,
    )
