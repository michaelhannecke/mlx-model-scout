"""Test fixtures: isolated SQLite DB per test session, no network, no LLM."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from mlx_model_scout.agents import llm
from mlx_model_scout.config.settings import get_settings
from mlx_model_scout.db.session import get_engine


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    """A FakeLLM reply that requests one tool call (OpenAI shape)."""
    return {"tool_calls": [{"id": call_id, "name": name, "arguments": json.dumps(args)}]}


class FakeLLM:
    """Stand-in for AsyncOpenAI: returns queued replies; an Exception in the queue is raised.

    A reply is either text (assistant content) or a dict from `tool_call(...)`.
    """

    def __init__(self) -> None:
        self.replies: list[str | dict | Exception] = []
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            calls = [
                SimpleNamespace(
                    id=c["id"],
                    type="function",
                    function=SimpleNamespace(name=c["name"], arguments=c["arguments"]),
                )
                for c in reply["tool_calls"]
            ]
            message = SimpleNamespace(content=reply.get("content"), tool_calls=calls)
        else:
            message = SimpleNamespace(content=reply, tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage, model="fake")


@pytest.fixture
def fake_llm(monkeypatch):
    """Route every agents.llm call to a FakeLLM (also for code that doesn't pass `client`)."""
    fake = FakeLLM()
    monkeypatch.setattr(llm, "get_client", lambda settings=None: fake)
    return fake


@pytest.fixture
def llm_down():
    """An exception that mimics an unreachable LLM server."""
    return openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8000/v1"))


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SCOUT_DB_PATH", str(tmp_path / "test.db"))
    get_settings.cache_clear()
    get_engine.cache_clear()
    yield
    get_settings.cache_clear()
    get_engine.cache_clear()


LLAMA3_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "tie_word_embeddings": False,
    "quantization": {"bits": 4, "group_size": 64},
}

MIXTRAL_CONFIG = {
    "architectures": ["MixtralForCausalLM"],
    "model_type": "mixtral",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "max_position_embeddings": 32768,
    "quantization": {"bits": 4, "group_size": 64},
}


@pytest.fixture
def llama_config():
    return dict(LLAMA3_8B_CONFIG)


@pytest.fixture
def mixtral_config():
    return dict(MIXTRAL_CONFIG)
