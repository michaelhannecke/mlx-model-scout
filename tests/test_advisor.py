"""Advisor agent: numbers attached by code, LLM output validated, fallbacks."""

import json

from mlx_model_scout.agents.advisor import (
    draft_report,
    memory_label,
    numbers_match_facts,
    rank_without_llm,
    speed_labels,
)
from mlx_model_scout.schemas.fit import FitEstimate
from mlx_model_scout.schemas.model import ModelCardInsights


def _fit(repo: str, verdict: str = "fits", tps: float | None = 50.0, headroom: float = 20.0) -> FitEstimate:
    return FitEstimate(
        repo_id=repo,
        profile_id="p",
        context_tokens=8192,
        weights_gb=16.0,
        kv_cache_gb=1.0,
        runtime_overhead_gb=1.5,
        required_gb=18.5,
        available_gb=48.0,
        headroom_gb=headroom,
        verdict=verdict,
        est_decode_tps=tps,
        confidence="high",
    )


FITS = [
    _fit("a/coder-4bit", tps=20.0),
    _fit("a/chat-moe-4bit", tps=73.2),
    _fit("a/tight-8bit", verdict="tight", tps=10.0, headroom=2.0),
    _fit("a/huge-bf16", verdict="no_fit", tps=5.0, headroom=-40.0),
]
CARDS = {
    "a/coder-4bit": ModelCardInsights(repo_id="a/coder-4bit", summary="A code model.", llm_used=True),
}


def _draft(*items: dict, summary: str = "Coder first, then the fast chat model.") -> str:
    return json.dumps({"ranking": list(items), "summary": summary})


async def test_llm_ranking_with_numbers_attached_by_code(fake_llm):
    fake_llm.replies = [
        _draft(
            {"repo_id": "a/coder-4bit", "rationale": "Built for code.", "trade_offs": ["slower"]},
            {"repo_id": "a/chat-moe-4bit", "rationale": "Fast generalist."},
            {"repo_id": "a/invented", "rationale": "?"},
            {"repo_id": "a/huge-bf16", "rationale": "Best model."},  # no_fit: may not be recommended
        )
    ]
    report = await draft_report("coding", "p", 8192, FITS, CARDS)

    assert report.generated_by == "llm"
    assert [r.repo_id for r in report.recommendations] == ["a/coder-4bit", "a/chat-moe-4bit"]
    assert [r.rank for r in report.recommendations] == [1, 2]
    assert report.recommendations[1].fit == FITS[1]  # numbers are the calculator's, unchanged
    assert report.excluded["a/huge-bf16"].startswith("no_fit")
    assert report.excluded["a/tight-8bit"] == "not ranked by the advisor for this use case"
    assert any("a/invented" in n and "a/huge-bf16" in n for n in report.notes)
    prompt = fake_llm.requests[0]["messages"][1]["content"]
    assert "A code model." in prompt and "a/huge-bf16" not in prompt


async def test_context_beyond_model_maximum_is_passed_as_warning(fake_llm):
    fit = _fit("a/codellama").model_copy(
        update={
            "assumptions": [
                "Requested context 32768 exceeds model maximum 16384.",
                "Runtime overhead 1.5 GiB.",
            ]
        }
    )
    fake_llm.replies = [_draft({"repo_id": "a/codellama", "rationale": "Python focus."})]
    await draft_report("python", "p", 32768, [fit])
    prompt = fake_llm.requests[0]["messages"][1]["content"]
    assert "warning: Requested context 32768 exceeds model maximum 16384." in prompt
    assert "Runtime overhead" not in prompt


async def test_altered_numbers_in_rationale_are_replaced(fake_llm):
    fake_llm.replies = [
        _draft(
            {"repo_id": "a/chat-moe-4bit", "rationale": "Runs at about 73.2 tok/s."},  # correct
            {"repo_id": "a/coder-4bit", "rationale": "Runs at 95 tok/s with 30 GiB free."},  # wrong
            summary="Expect up to 120 tok/s.",
        )
    ]
    report = await draft_report("chat", "p", 8192, FITS)

    assert report.recommendations[0].rationale == "Runs at about 73.2 tok/s."
    assert "95" not in report.recommendations[1].rationale
    assert "120" not in report.summary
    assert len(report.notes) == 2


def test_numbers_match_facts():
    assert numbers_match_facts("about 73 tok/s, 20 GiB headroom", FITS)
    assert numbers_match_facts("no numbers, 3 trade-offs", FITS)  # numbers without unit are ignored
    assert not numbers_match_facts("about 74 tokens/s", FITS)


async def test_llm_down_uses_deterministic_ranking(fake_llm, llm_down):
    fake_llm.replies = [llm_down]
    report = await draft_report("chat", "p", 8192, FITS)

    assert report.generated_by == "deterministic"
    assert [r.repo_id for r in report.recommendations] == ["a/chat-moe-4bit", "a/coder-4bit", "a/tight-8bit"]


async def test_nothing_fits_skips_llm(fake_llm):
    report = await draft_report("chat", "p", 8192, [FITS[3]])
    assert fake_llm.requests == [] and report.recommendations == []
    assert "a/huge-bf16" in report.excluded


def test_deterministic_ranking_caps_at_five():
    fits = [_fit(f"a/m{i}", tps=float(i)) for i in range(7)]
    report = rank_without_llm("chat", "p", 8192, fits)
    assert [r.repo_id for r in report.recommendations] == ["a/m6", "a/m5", "a/m4", "a/m3", "a/m2"]
    assert report.excluded["a/m0"] == "lower in deterministic ranking"


def test_memory_and_speed_labels_are_computed():
    # observed: the advisor LLM called 28 GiB of 48 GiB headroom "tight"; labels now come from code
    assert memory_label(_fit("a", headroom=28.0)) == "ample"
    assert memory_label(_fit("a", headroom=12.0)) == "moderate"
    assert memory_label(_fit("a", headroom=2.0)) == "tight"
    labels = speed_labels([*FITS[:3], _fit("a/unknown", tps=None)])
    assert labels == {
        "a/chat-moe-4bit": "rank 1 of 3 (fastest)",
        "a/coder-4bit": "rank 2 of 3",
        "a/tight-8bit": "rank 3 of 3 (slowest)",
    }


async def test_labels_reach_the_prompt(fake_llm):
    fake_llm.replies = [_draft({"repo_id": "a/coder-4bit", "rationale": "Code."})]
    await draft_report("coding", "p", 8192, FITS[:2])
    prompt = fake_llm.requests[0]["messages"][1]["content"]
    assert "memory headroom: moderate; speed: rank 2 of 2 (slowest)" in prompt
