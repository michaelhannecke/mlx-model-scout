from mlx_model_scout.app.model_card import (
    card_metadata,
    clean_card_text,
    is_boilerplate,
    split_front_matter,
    verify_benchmarks,
)

# Shape of real mlx-community cards (gemma-3-27b-it-4bit: front matter names the -pt model).
GEMMA_MLX_CARD = """---
license: gemma
pipeline_tag: image-text-to-text
base_model: google/gemma-3-27b-pt
tags:
- mlx
---

# mlx-community/gemma-3-27b-it-4bit
This model was converted to MLX format from [`google/gemma-3-27b-it`]() using mlx-vlm version **0.1.18**.
## Use with mlx

```bash
pip install -U mlx-vlm
```
"""

QWEN_MLX_CARD = """---
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-30B-A3B-Instruct-2507
---

# mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit

This model [mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit](https://huggingface.co/mlx-community/x) was
converted to MLX format from [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
using mlx-lm version **0.26.3**.
"""


def test_front_matter_split():
    meta, body = split_front_matter(QWEN_MLX_CARD)
    assert meta["license"] == "apache-2.0"
    assert body.lstrip().startswith("# mlx-community")
    assert split_front_matter("no front matter") == ({}, "no front matter")
    assert split_front_matter("---\n: : bad yaml [\n---\nbody")[0] == {}


def test_converted_from_line_beats_front_matter():
    assert card_metadata(GEMMA_MLX_CARD) == {
        "license": "gemma",
        "base_model": "google/gemma-3-27b-it",
        "pipeline_tag": "image-text-to-text",
    }
    assert card_metadata(QWEN_MLX_CARD)["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_front_matter_base_model_list_is_supported():
    card = "---\nbase_model:\n- org/base-a\n- org/base-b\n---\nSome prose."
    assert card_metadata(card)["base_model"] == "org/base-a"


def test_clean_card_text_strips_noise():
    text = clean_card_text(GEMMA_MLX_CARD + '\n<img src="x.png">\n![logo](l.png)\n\n\n\nEnd', 10_000)
    assert "pip install" not in text and "license:" not in text
    assert "<img" not in text and "![logo]" not in text and "\n\n\n" not in text
    # pseudo-tags keep their meaning but are neutralized: "<think>" is a special token for Qwen3
    cleaned = clean_card_text("Does not generate <think></think> blocks.<br>", 100)
    assert cleaned == "Does not generate [think][/think] blocks."
    assert clean_card_text("x" * 50, 10) == "x" * 10


def test_boilerplate_detection():
    assert is_boilerplate(QWEN_MLX_CARD)
    assert not is_boilerplate("A real card. " * 200)


def test_verify_benchmarks_drops_numbers_not_in_card():
    card = (
        "| Benchmark | Ours | Other |\n| MMLU-Pro | 78.4 | 169.6 |\n"
        "| AIME25 | 61.3 | 70 |\n| GPQA | 70.0 | 1 |"
    )
    kept, dropped = verify_benchmarks({"MMLU-Pro": 78.4, "GPQA": 70, "Arena": 69.6, "AIME25": 61.3}, card)
    assert kept == {"MMLU-Pro": 78.4, "GPQA": 70, "AIME25": 61.3}
    assert dropped == ["Arena"]  # 69.6 only appears inside 169.6
