from mlx_model_scout.app.model_spec import build_model_spec, effective_bits, parse_quantization
from mlx_model_scout.schemas.model import QuantizationInfo


def test_llama3_8b_param_count(llama_config):
    spec = build_model_spec("mlx-community/Meta-Llama-3-8B-Instruct-4bit", llama_config)
    assert 7.9 < spec.total_params_b < 8.1  # official: 8.03B
    # active = params read per token: excludes the input-embedding table (row lookup only)
    assert 7.4 < spec.active_params_b < 7.6
    assert spec.head_dim == 128
    assert spec.confidence == "high"
    assert not spec.is_moe


def test_mixtral_moe_total_and_active(mixtral_config):
    spec = build_model_spec("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", mixtral_config)
    assert spec.is_moe
    assert 46.0 < spec.total_params_b < 47.5  # official: ~46.7B
    assert 12.0 < spec.active_params_b < 13.5  # official: ~12.9B


def test_quantization_parsing_and_overhead():
    q = parse_quantization({"quantization": {"bits": 4, "group_size": 64}})
    assert q.bits == 4 and q.group_size == 64
    assert effective_bits(q) == 4.5  # fp16 scale + bias per 64 weights
    assert parse_quantization({"torch_dtype": "bfloat16"}).bits == 16
    assert effective_bits(QuantizationInfo(bits=4, group_size=32, mode="mxfp4")) == 4.25


def test_mla_marks_kv_unsupported(llama_config):
    cfg = {**llama_config, "model_type": "deepseek_v3"}
    spec = build_model_spec("x/DeepSeek-V3-4bit", cfg)
    assert spec.kv_cache_supported is False
    assert spec.confidence == "low"


def test_incomplete_config_falls_back_to_name():
    spec = build_model_spec("mlx-community/SomeModel-14B-4bit", {"model_type": "exotic"})
    assert spec.total_params_b == 14.0
    assert spec.confidence == "low"


def test_gemma3_omitted_defaults_are_filled():
    # Real mlx-community/gemma-3-27b-it-4bit text_config: no vocab_size, no tie_word_embeddings.
    raw = {
        "model_type": "gemma3",
        "quantization": {"bits": 4, "group_size": 64},
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "num_attention_heads": 32,
            "num_hidden_layers": 62,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "sliding_window": 1024,
        },
    }
    spec = build_model_spec("mlx-community/gemma-3-27b-it-4bit", raw)
    assert 26.5 < spec.total_params_b < 27.5  # computed, not guessed from the name
    assert spec.max_context == 131_072
    assert spec.confidence == "high"


def test_mixed_quantization_lowers_confidence(llama_config):
    quant = {"bits": 4, "group_size": 32, "mode": "mxfp4", "model.layers.0.self_attn.q_proj": {"bits": 8}}
    spec = build_model_spec("x/model-MXFP4-Q8", {**llama_config, "quantization": quant})
    assert spec.quantization.bits == 4
    assert spec.confidence == "medium"
    assert any("Per-layer quantization" in n for n in spec.notes)


def test_router_only_overrides_keep_high_confidence(llama_config):
    # Qwen3-MoE 4-bit repos keep only the tiny router (`mlp.gate`) at 8 bit.
    quant = {"bits": 4, "group_size": 64, "model.layers.0.mlp.gate": {"bits": 8, "group_size": 64}}
    spec = build_model_spec("x/Qwen3-30B-A3B-4bit", {**llama_config, "quantization": quant})
    assert spec.confidence == "high"


def test_hybrid_architecture_has_no_kv_estimate(llama_config):
    spec = build_model_spec("x/Qwen3-Next-80B-A3B-4bit", {**llama_config, "model_type": "qwen3_next"})
    assert spec.kv_cache_supported is False
    assert any("KV cache not estimated" in n for n in spec.notes)


def test_multimodal_text_config_is_merged(llama_config):
    raw = {"model_type": "llava", "text_config": llama_config, "quantization": llama_config["quantization"]}
    spec = build_model_spec("x/vlm-4bit", raw)
    assert 7.9 < spec.total_params_b < 8.1
