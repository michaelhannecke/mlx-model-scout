"""Local LLM access (mlx-lm server via OpenAI-compatible API) with audit logging.

Structured outputs: mlx-lm's server does not guarantee JSON-schema-constrained
decoding. We therefore (1) put the JSON schema into the system prompt,
(2) request JSON, (3) validate with Pydantic, (4) retry with the validation
error fed back. Every attempt is logged to `llm_calls`.

All agents MUST go through `structured_call` / `chat` so nothing bypasses the
audit log. Do not use FastMCP `ctx.sample()`: that would delegate generation to
the MCP client's (possibly cloud) model and break the "fully local" guarantee.
"""

import json
import re
import time
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from mlx_model_scout.config.settings import Settings, get_settings
from mlx_model_scout.db.audit import log_llm_call

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    """Raised when the local LLM is unreachable or never returns valid output."""


def get_client(settings: Settings | None = None) -> AsyncOpenAI:
    settings = settings or get_settings()
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_s,
    )


def _extract_json(text: str) -> str:
    """Strip markdown fences and a LEADING thinking block some models emit.

    Only a block at the start is removed: model cards literally mention "<think></think>"
    and that text must survive inside JSON string values.
    """
    text = re.sub(r"\A\s*<think>.*?</think>", "", text, flags=re.DOTALL)
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text


async def chat(
    agent: str,
    messages: list[dict],
    run_id: str | None = None,
    client: AsyncOpenAI | None = None,
    settings: Settings | None = None,
    attempt: int = 1,
) -> str:
    """Plain chat completion with audit logging. Returns the message content."""
    message = await _complete(agent, messages, run_id, client, settings, attempt)
    return message.content or ""


async def chat_with_tools(
    agent: str,
    messages: list[dict],
    tools: list[dict],
    run_id: str | None = None,
    client: AsyncOpenAI | None = None,
    settings: Settings | None = None,
) -> Any:
    """Chat completion with OpenAI-format `tools`; returns the assistant message.

    The message has `.content` and `.tool_calls` (each with `.id` and `.function.name` /
    `.function.arguments` as JSON text). Used by the local MCP client's agent loop.
    """
    return await _complete(agent, messages, run_id, client, settings, tools=tools)


async def _complete(
    agent: str,
    messages: list[dict],
    run_id: str | None,
    client: AsyncOpenAI | None,
    settings: Settings | None,
    attempt: int = 1,
    **extra: Any,
) -> Any:
    """Single audited chat completion; `extra` is passed through (e.g. `tools`)."""
    settings = settings or get_settings()
    client = client or get_client(settings)
    start = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            **extra,
        )
    except Exception as exc:
        log_llm_call(
            run_id=run_id,
            agent=agent,
            model=settings.llm_model,
            messages=messages,
            latency_ms=int((time.perf_counter() - start) * 1000),
            attempt=attempt,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise LLMError(f"Local LLM call failed ({settings.llm_base_url}): {exc}") from exc

    message = resp.choices[0].message
    response = message.content or ""
    if tool_calls := getattr(message, "tool_calls", None):
        calls = [f"{c.function.name}({c.function.arguments})" for c in tool_calls]
        response = f"{response}\n[tool_calls] {'; '.join(calls)}".strip()
    usage = resp.usage
    log_llm_call(
        run_id=run_id,
        agent=agent,
        model=resp.model or settings.llm_model,
        messages=messages,
        response=response,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        latency_ms=int((time.perf_counter() - start) * 1000),
        attempt=attempt,
    )
    return message


async def structured_call[T: BaseModel](
    agent: str,
    output_model: type[T],
    system: str,
    user: str,
    run_id: str | None = None,
    client: AsyncOpenAI | None = None,
    settings: Settings | None = None,
) -> T:
    """Ask the local LLM for JSON matching `output_model`; validate and retry."""
    settings = settings or get_settings()
    schema = json.dumps(output_model.model_json_schema(), indent=2)
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                f"{system}\n\nRespond with ONE JSON object only, no prose, no markdown. "
                f"It must validate against this JSON schema:\n{schema}"
            ),
        },
        {"role": "user", "content": user},
    ]
    last_error: Exception | None = None
    for attempt in range(1, settings.llm_max_retries + 2):
        raw = await chat(agent, messages, run_id, client, settings, attempt)
        try:
            return output_model.model_validate_json(_extract_json(raw))
        except ValidationError as exc:
            last_error = exc
            log_llm_call(
                run_id=run_id,
                agent=agent,
                model=settings.llm_model,
                messages=[],
                response=raw,
                attempt=attempt,
                status="validation_error",
                error=str(exc)[:2000],
            )
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid output. Fix these errors and resend JSON only:\n{exc}"},
            ]
    raise LLMError(f"{agent}: no valid {output_model.__name__} after retries: {last_error}")
