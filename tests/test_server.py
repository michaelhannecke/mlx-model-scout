"""In-memory MCP round trips (no subprocess, no network)."""

import json

import httpx
from fastmcp import Client
from sqlmodel import select

from mlx_model_scout.app.hf_client import HFClient
from mlx_model_scout.db.audit import log_llm_call
from mlx_model_scout.db.models import LLMCall, ToolCall
from mlx_model_scout.db.session import session_scope
from mlx_model_scout.server import create_app
from mlx_model_scout.tools import hub_tools


async def test_capabilities_registered():
    async with Client(create_app()) as c:
        tools = {t.name for t in await c.list_tools()}
        assert {"scan_mlx_models", "estimate_fit", "save_report", "list_hardware_profiles"} <= tools
        assert "recommend_models" in {p.name for p in await c.list_prompts()}
        assert "hardware://profiles" in {str(r.uri) for r in await c.list_resources()}


async def test_prompt_contains_hitl_checkpoints():
    async with Client(create_app()) as c:
        p = await c.get_prompt("recommend_models", {"use_case": "coding", "profile_id": "studio-m4max-64"})
        text = p.messages[0].content.text
        assert "CHECKPOINT 1" in text and "CHECKPOINT 2" in text


async def test_download_requires_confirmation():
    async with Client(create_app()) as c:
        r = await c.call_tool("download_model", {"repo_id": "x/y"})
        assert r.data["status"] == "confirmation_required"


async def test_save_report_is_audited():
    async with Client(create_app()) as c:
        r = await c.call_tool("save_report", {"report": {"summary": "x"}, "approved": True})
        assert r.data["approved"] is True
    with session_scope() as s:
        assert s.exec(select(ToolCall).where(ToolCall.tool == "save_report")).first() is not None


async def test_analyze_model_card_tool_follows_base_model_and_caches(fake_llm, monkeypatch):
    cards = {
        "/mlx-community/q-4bit/raw/main/README.md": (
            "---\nlicense: apache-2.0\n---\nThis model was converted to MLX format from [org/q](https://x)."
        ),
        "/org/q/raw/main/README.md": "# q\nA coding model. MMLU 70.1.",
    }
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, text=cards[r.url.path]) if r.url.path in cards else httpx.Response(404)
    )
    monkeypatch.setattr(hub_tools, "HFClient", lambda: HFClient(transport=transport))
    fake_llm.replies = [json.dumps({"summary": "Coding model.", "reported_benchmarks": {"MMLU": 70.1}})]

    async with Client(create_app()) as c:
        first = (
            await c.call_tool("analyze_model_card", {"repo_id": "mlx-community/q-4bit"})
        ).structured_content
        second = (
            await c.call_tool("analyze_model_card", {"repo_id": "mlx-community/q-4bit"})
        ).structured_content
        catalog = await c.read_resource("catalog://models")  # card-only cache row must not break it

    assert first["base_model"] == "org/q" and first["card_source"] == "base_model"
    assert first["reported_benchmarks"] == {"MMLU": 70.1}
    assert second == first and len(fake_llm.requests) == 1  # second call served from cache
    assert catalog


async def test_all_resources_are_readable():
    async with Client(create_app()) as c:
        contents = {
            uri: json.loads((await c.read_resource(uri))[0].text)
            for uri in ("hardware://profiles", "hardware://chips", "catalog://models", "runs://recent")
        }
    assert "studio-m4max-64" in {p["id"] for p in contents["hardware://profiles"]}
    assert contents["hardware://chips"]["m5-pro-20c"]["memory_bandwidth_gbs"] == 307
    assert contents["catalog://models"] == [] and contents["runs://recent"] == []  # fresh test DB


def test_llm_log_truncates_by_default():
    log_llm_call(
        agent="t", model="m", messages=[{"role": "user", "content": "x" * 5000}], response="y" * 5000
    )
    with session_scope() as s:
        row = s.exec(select(LLMCall)).first()
        assert len(row.messages) < 1000 and len(row.response) == 500
        assert len(row.messages_sha256) == 64
