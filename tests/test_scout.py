"""Scout agent: deterministic pre-filter, LLM shortlist validation, fallbacks."""

import json

import pytest

from mlx_model_scout.agents.scout import MAX_POOL, prefilter, select_candidates
from mlx_model_scout.app.model_spec import size_gb_from_name
from mlx_model_scout.schemas.model import HubModel


def _m(repo: str, pipeline: str | None = "text-generation", downloads: int = 0) -> HubModel:
    return HubModel(repo_id=f"mlx-community/{repo}", pipeline_tag=pipeline, downloads=downloads)


SCAN = [
    _m("Qwen2.5-Coder-32B-Instruct-4bit", downloads=500),
    _m("Qwen3-30B-A3B-Instruct-2507-4bit", downloads=900),
    _m("whisper-large-v3-turbo", pipeline="automatic-speech-recognition", downloads=5000),
    _m("DeepSeek-V3-0324-4bit", downloads=300),
    _m("Llama-3.2-3B-Instruct-4bit", downloads=100),
]


@pytest.mark.parametrize(
    ("repo", "expected"),
    [
        ("mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", 30 * 4.5 / 8 * 1e9 / 2**30),
        ("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit", 8 * 8.5 / 8 * 1e9 / 2**30),
        ("mlx-community/gpt-oss-20b-MXFP4-Q8", 20 * 4.5 / 8 * 1e9 / 2**30),
        ("mlx-community/Mixtral-8x7B-Instruct-v0.1", None),  # "8x7B": no plain size in the name
        ("mlx-community/Mistral-7B-Instruct-v0.3-bf16", 7 * 2 * 1e9 / 2**30),
        ("mlx-community/whisper-large-v3-turbo", None),
    ],
)
def test_size_from_name(repo, expected):
    size = size_gb_from_name(repo)
    assert size == (pytest.approx(expected) if expected else None)


def test_prefilter_drops_non_llm_and_too_large():
    pool, notes = prefilter(SCAN, available_gb=48)
    ids = [m.repo_id.split("/")[1] for m in pool]
    assert "whisper-large-v3-turbo" not in ids  # pipeline
    assert "DeepSeek-V3-0324-4bit" in ids  # no size in name -> kept; the calculator decides later
    assert len(notes) == 1
    pool, _ = prefilter([_m("Qwen2.5-72B-Instruct-4bit")], available_gb=24)
    assert pool == []


def test_prefilter_caps_pool_by_downloads():
    many = [_m(f"Model{i}-1B-4bit", downloads=i) for i in range(MAX_POOL + 5)]
    pool, notes = prefilter(many, available_gb=None)
    assert len(pool) == MAX_POOL and pool[0].downloads == MAX_POOL + 4
    assert any("capped" in n for n in notes)


async def test_llm_shortlist_is_validated(fake_llm):
    fake_llm.replies = [
        json.dumps(
            {
                "candidates": [
                    {"repo_id": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", "reason": "coder"},
                    {"repo_id": "mlx-community/Invented-Coder-7B", "reason": "made up"},
                    {"repo_id": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", "reason": "duplicate"},
                    {"repo_id": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "reason": "fast MoE"},
                ]
            }
        )
    ]
    result = await select_candidates("coding assistant", SCAN, available_gb=48, max_candidates=5)

    assert result.llm_used
    assert [c.repo_id.split("/")[1] for c in result.candidates] == [
        "Qwen2.5-Coder-32B-Instruct-4bit",
        "Qwen3-30B-A3B-Instruct-2507-4bit",
    ]
    assert any("Invented-Coder-7B" in n for n in result.notes)
    prompt = fake_llm.requests[0]["messages"][1]["content"]
    assert "whisper" not in prompt and "coding assistant" in prompt


async def test_llm_down_falls_back_to_downloads(fake_llm, llm_down):
    fake_llm.replies = [llm_down]
    result = await select_candidates("chat", SCAN, available_gb=48, max_candidates=2)

    assert not result.llm_used
    assert [c.repo_id.split("/")[1] for c in result.candidates] == [
        "Qwen3-30B-A3B-Instruct-2507-4bit",
        "Qwen2.5-Coder-32B-Instruct-4bit",
    ]


async def test_only_invented_ids_fall_back(fake_llm):
    fake_llm.replies = [json.dumps({"candidates": [{"repo_id": "x/made-up", "reason": "?"}]})]
    result = await select_candidates("chat", SCAN, available_gb=48, max_candidates=1)
    assert not result.llm_used and len(result.candidates) == 1
