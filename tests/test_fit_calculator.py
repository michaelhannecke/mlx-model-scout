import pytest

from mlx_model_scout.app.fit_calculator import estimate_fit, kv_cache_bytes
from mlx_model_scout.app.model_spec import build_model_spec
from mlx_model_scout.app.profiles import load_chips, load_profiles
from mlx_model_scout.schemas.hardware import Chip, HardwareProfile

GIB = 1024**3


@pytest.fixture
def chips():
    return load_chips()


def test_kv_cache_llama3_8b(llama_config):
    spec = build_model_spec("x/llama", llama_config)
    # 2 * 32 layers * 8 kv heads * 128 dim * 8192 ctx * 2 bytes = 1 GiB
    assert kv_cache_bytes(spec, 8192) == pytest.approx(1 * GIB)


def test_llama_fits_on_studio(llama_config, chips):
    spec = build_model_spec("x/llama-4bit", llama_config)
    profile = load_profiles()["studio-m4max-64"]
    fit = estimate_fit(spec, profile, chips[profile.chip], 8192)
    assert fit.verdict == "fits"
    assert 4.0 < fit.weights_gb < 4.5  # 8.03B * 4.5 bit / 8
    assert fit.est_decode_tps and 70 < fit.est_decode_tps < 95  # dense: 80% of 546 GB/s / ~5.3 GB per token
    assert fit.available_gb == pytest.approx(48.0)  # min(64*0.75, 64-8)


def test_bf16_mixtral_does_not_fit_small_machine(mixtral_config, chips):
    cfg = {k: v for k, v in mixtral_config.items() if k != "quantization"}
    spec = build_model_spec("x/mixtral-bf16", cfg)
    small = HardwareProfile(id="t", name="t", chip="m4-pro-20c", unified_memory_gb=24)
    fit = estimate_fit(spec, small, chips["m4-pro-20c"], 4096)
    assert fit.verdict == "no_fit"
    assert fit.headroom_gb < 0


def test_explicit_wired_limit_is_used(llama_config, chips):
    spec = build_model_spec("x/llama", llama_config)
    p = HardwareProfile(id="t", name="t", chip="m4-max-40c", unified_memory_gb=64, wired_limit_gb=58)
    assert estimate_fit(spec, p, chips["m4-max-40c"]).available_gb == pytest.approx(58)


def test_cluster_has_no_speed_estimate(mixtral_config, chips):
    spec = build_model_spec("x/mixtral", mixtral_config)
    profile = load_profiles()["cluster-2x-studio-m4max-64"]
    fit = estimate_fit(spec, profile, chips[profile.chip])
    assert fit.est_decode_tps is None
    assert fit.available_gb == pytest.approx(96.0)


def test_chip_without_bandwidth_has_no_speed(llama_config):
    spec = build_model_spec("x/llama", llama_config)
    profile = HardwareProfile(id="t", name="t", chip="future", unified_memory_gb=128)
    chip = Chip(key="future", display_name="Future chip", memory_bandwidth_gbs=None, verified=False)
    fit = estimate_fit(spec, profile, chip)
    assert fit.verdict == "fits"
    assert fit.est_decode_tps is None


# Real config of mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit (relevant keys).
QWEN3_30B_A3B = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 151936,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "tie_word_embeddings": False,
    "quantization": {"bits": 4, "group_size": 64},
}


@pytest.mark.parametrize(("context", "measured_tps"), [(19, 106.0), (3020, 90.9)])
def test_decode_estimate_matches_measurement_on_m5_pro(chips, context, measured_tps):
    """Calibration anchor: decode speed measured on an M5 Pro 20c / 64 GB (omlx, 2026-10-28)."""
    spec = build_model_spec("mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", QWEN3_30B_A3B)
    profile = HardwareProfile(id="t", name="t", chip="m5-pro-20c", unified_memory_gb=64)
    fit = estimate_fit(spec, profile, chips["m5-pro-20c"], context)
    assert fit.est_decode_tps == pytest.approx(measured_tps, rel=0.1)


def _layered(model_type: str, layers: int, kv: int, head_dim: int, **extra) -> dict:
    return {
        "model_type": model_type,
        "hidden_size": 2048,
        "num_hidden_layers": layers,
        "num_attention_heads": 16,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "vocab_size": 32000,
        "intermediate_size": 8192,
        **extra,
    }


def test_kv_gemma4_sliding_and_global_layers():
    # Real gemma-4-12b layout: 5 sliding (window 1024) + 1 global per block; global layers 1 KV head x 512.
    types = (["sliding_attention"] * 5 + ["full_attention"]) * 8
    cfg = _layered(
        "gemma4_unified_text",
        48,
        8,
        256,
        layer_types=types,
        sliding_window=1024,
        num_global_key_value_heads=1,
        global_head_dim=512,
    )
    spec = build_model_spec("x/gemma-4-12b-4bit", cfg)
    assert spec.num_sliding_layers == 40 and spec.confidence == "high"
    full = 8 * 2 * 1 * 512 * 32768 * 2
    sliding = 40 * 2 * 8 * 256 * 1024 * 2
    assert kv_cache_bytes(spec, 32768) == full + sliding  # 0.81 GiB instead of 12 GiB if all were full


def test_kv_gemma3_default_pattern_from_type_defaults():
    # gemma-3-27b config has no sliding_window_pattern; transformers default is 6.
    spec = build_model_spec("x/gemma-3-27b-4bit", _layered("gemma3_text", 62, 16, 128, sliding_window=1024))
    assert spec.num_sliding_layers == 52
    expected = 10 * 2 * 16 * 128 * 8192 * 2 + 52 * 2 * 16 * 128 * 1024 * 2
    assert kv_cache_bytes(spec, 8192) == expected


def test_kv_sliding_window_larger_than_context_counts_context():
    types = ["sliding_attention", "full_attention"] * 12
    spec = build_model_spec(
        "x/gpt-oss", _layered("gpt_oss", 24, 8, 64, layer_types=types, sliding_window=128)
    )
    assert kv_cache_bytes(spec, 100) == 24 * 2 * 8 * 64 * 100 * 2


def test_kv_qwen3_next_counts_full_attention_layers_only():
    cfg = _layered("qwen3_next", 48, 2, 256, full_attention_interval=4)
    spec = build_model_spec("x/Qwen3-Next-80B-A3B-4bit", cfg)
    assert spec.kv_cache_supported and spec.num_linear_layers == 36
    assert spec.confidence == "medium"
    assert kv_cache_bytes(spec, 8192) == 12 * 2 * 2 * 256 * 8192 * 2


def test_bare_sliding_window_without_pattern_is_ignored():
    # Qwen2 sets sliding_window but disables it (use_sliding_window=false): stay conservative.
    spec = build_model_spec("x/qwen2", _layered("qwen2", 28, 4, 128, sliding_window=32768))
    assert spec.num_sliding_layers == 0
    assert kv_cache_bytes(spec, 8192) == 28 * 2 * 4 * 128 * 8192 * 2


# Real text_config of mlx-community/Qwen3.8-27B-4bit (relevant keys): hybrid Gated DeltaNet + gated attention.
QWEN38_27B = {
    "model_type": "qwen3_5_text",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "intermediate_size": 17408,
    "tie_word_embeddings": False,
    "attn_output_gate": True,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "quantization": {"bits": 4, "group_size": 64},
}


def test_qwen35_hybrid_param_count():
    spec = build_model_spec("mlx-community/Qwen3.8-27B-4bit", QWEN38_27B)
    # hand count from the safetensors index: 26.9B text params (the name says 27B)
    assert spec.total_params_b == pytest.approx(26.9, abs=0.05)
    assert spec.num_linear_layers == 48


def test_actual_file_size_is_preferred_for_memory(llama_config):
    spec = build_model_spec("x/llama", llama_config).model_copy(update={"weights_file_gb": 5.0})
    p = HardwareProfile(id="t", name="t", chip="m4-max-40c", unified_memory_gb=64)
    fit = estimate_fit(spec, p, load_chips()["m4-max-40c"], 8192)
    assert fit.weights_gb == 5.0
    assert any("actual download size" in a for a in fit.assumptions)


@pytest.mark.parametrize(("context", "measured_tps"), [(789, 17.6), (24089, 16.0)])
def test_dense_decode_estimate_matches_measurement_on_m5_pro(chips, context, measured_tps):
    """Calibration anchor (dense, hybrid linear attention): Qwen3.8-27B-4bit on an M5 Pro 20c, omlx,
    2026-10-28, 256 generated tokens, mean of 2 runs. Also validates the per-layer KV estimate:
    counting all 64 layers as full attention would predict ~12 tok/s at 24k instead of 16.0."""
    spec = build_model_spec("mlx-community/Qwen3.8-27B-4bit", QWEN38_27B)
    profile = HardwareProfile(id="t", name="t", chip="m5-pro-20c", unified_memory_gb=64)
    fit = estimate_fit(spec, profile, chips["m5-pro-20c"], context)
    assert fit.est_decode_tps == pytest.approx(measured_tps, rel=0.1)
