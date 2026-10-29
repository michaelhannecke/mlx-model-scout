"""Analyst agent with a fake LLM: structured output, validation retry, graceful degradation."""

import json

from sqlmodel import select

from mlx_model_scout.agents.analyst import extract_card_insights
from mlx_model_scout.db.models import LLMCall
from mlx_model_scout.db.session import session_scope

MLX_CARD = """---
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-30B-A3B-Instruct-2507
---
This model was converted to MLX format from [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/x)
using mlx-lm version **0.26.3**.
"""

BASE_CARD = """---
license: apache-2.0
---
# Qwen3-30B-A3B-Instruct-2507
A non-thinking MoE instruct model with strong tool use and 256K context.

| Benchmark | Qwen3-30B-A3B-Instruct-2507 | Other model |
| MMLU-Pro | 78.4 | 81.1 |
| LiveCodeBench | 43.2 | 40.0 |
"""

GOOD = {
    "summary": "MoE instruct model for chat and tool use.",
    "primary_use_cases": ["chat", "agents"],
    "languages": ["English", "Chinese"],
    "strengths": ["tool use", "long context"],
    "caveats": [],
    "reported_benchmarks": {"MMLU-Pro": 78.4, "LiveCodeBench": 43.2, "Invented": 99.9},
}


async def test_uses_base_card_and_drops_hallucinated_benchmark(fake_llm):
    fake_llm.replies = [json.dumps(GOOD)]
    insights = await extract_card_insights("mlx-community/q-4bit", MLX_CARD, BASE_CARD)

    assert insights.llm_used and insights.card_source == "base_model"
    assert insights.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert insights.license == "apache-2.0"  # from front matter, not the LLM
    assert insights.reported_benchmarks == {"MMLU-Pro": 78.4, "LiveCodeBench": 43.2}
    assert any("Invented" in n for n in insights.notes)
    prompt = fake_llm.requests[0]["messages"]
    assert "JSON schema" in prompt[0]["content"]
    assert "Model: Qwen/Qwen3-30B-A3B-Instruct-2507" in prompt[1]["content"]
    assert "converted to MLX" not in prompt[1]["content"]


async def test_benchmarks_are_capped(fake_llm):
    many = {f"B{i}": float(i) for i in range(1, 15)}
    card = "# Model\n" + " ".join(f"{v:g}" for v in many.values())
    fake_llm.replies = [json.dumps({"reported_benchmarks": many})]
    insights = await extract_card_insights("x/m", "", card)

    assert list(insights.reported_benchmarks) == [f"B{i}" for i in range(1, 11)]
    assert any("first 10 of 14" in n for n in insights.notes)


async def test_invalid_json_is_retried_with_error_feedback(fake_llm):
    fake_llm.replies = [
        "<think>hmm</think>Sure! {not json",
        "```json\n" + json.dumps({**GOOD, "reported_benchmarks": {}}) + "\n```",
    ]
    insights = await extract_card_insights("mlx-community/q-4bit", MLX_CARD, BASE_CARD)

    assert insights.llm_used and insights.summary == GOOD["summary"]
    assert "Invalid output" in fake_llm.requests[1]["messages"][-1]["content"]
    with session_scope() as s:
        statuses = [r.status for r in s.exec(select(LLMCall).where(LLMCall.agent == "analyst"))]
    assert statuses.count("validation_error") == 1 and statuses.count("ok") == 2


async def test_llm_down_returns_metadata_only(fake_llm, llm_down):
    fake_llm.replies = [llm_down]
    insights = await extract_card_insights("mlx-community/q-4bit", MLX_CARD, BASE_CARD)

    assert not insights.llm_used
    assert insights.license == "apache-2.0" and insights.summary is None
    assert any("LLM unavailable" in n for n in insights.notes)


async def test_boilerplate_card_without_base_card_skips_llm(fake_llm):
    insights = await extract_card_insights("mlx-community/q-4bit", MLX_CARD, base_card="")

    assert fake_llm.requests == []
    assert insights.card_source == "none" and not insights.llm_used
    assert insights.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
