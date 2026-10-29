"""Thin async client for the public Hugging Face Hub REST API.

Uses plain HTTP (httpx) instead of `huggingface_hub` to keep the dependency
surface small and the requests transparent/loggable. No token is required;
HF_TOKEN only raises rate limits.
"""

import re
from datetime import datetime
from typing import Any

import httpx

from mlx_model_scout.config.settings import Settings, get_settings
from mlx_model_scout.schemas.model import HubModel


class HubError(RuntimeError):
    """Raised when the Hub returns an error or unexpected payload."""


# Hub repo ids are "<org>/<name>" with letters, digits, "-", "_", "."; max 96 chars per part.
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$")


def validate_repo_id(repo_id: str) -> str:
    """Return `repo_id` if it is a plain Hub id, else raise HubError.

    repo ids come from tool arguments (client model, LLM, user input) and from model cards, and
    are interpolated into URL paths that carry the HF token. Without this check, values like
    `../api/whoami-v2` or `a/b?x=1` would reach other Hub endpoints (the host cannot change).
    """
    if not isinstance(repo_id, str) or not _REPO_ID.match(repo_id) or ".." in repo_id:
        raise HubError(f"Invalid Hugging Face repo id {repo_id!r}: expected 'org/name'.")
    return repo_id


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class HFClient:
    """Async Hub client. Use as `async with HFClient() as hf: ...`."""

    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings or get_settings()
        headers = {"User-Agent": f"{self.settings.server_name}/{self.settings.server_version}"}
        if self.settings.hf_token:
            headers["Authorization"] = f"Bearer {self.settings.hf_token}"
        self._client = httpx.AsyncClient(
            base_url=self.settings.hf_api_base,
            headers=headers,
            timeout=self.settings.hf_timeout_s,
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> HFClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def list_models(
        self,
        author: str,
        search: str | None = None,
        limit: int | None = None,
        sort: str = "lastModified",
    ) -> list[HubModel]:
        """List repos of an author, newest first."""
        params: dict[str, Any] = {
            "author": author,
            "sort": sort,
            "direction": -1,
            "limit": limit or self.settings.hf_scan_limit,
        }
        if search:
            params["search"] = search
        resp = await self._client.get("/api/models", params=params)
        if resp.status_code != 200:
            raise HubError(f"Hub list_models failed: HTTP {resp.status_code}")
        data = resp.json()
        if not isinstance(data, list):
            raise HubError("Unexpected list_models payload")
        return [
            HubModel(
                repo_id=item.get("id") or item.get("modelId"),
                author=item.get("author", author),
                last_modified=_parse_dt(item.get("lastModified")),
                created_at=_parse_dt(item.get("createdAt")),
                downloads=item.get("downloads"),
                likes=item.get("likes"),
                pipeline_tag=item.get("pipeline_tag"),
                tags=item.get("tags", []),
            )
            for item in data
            if item.get("id") or item.get("modelId")
        ]

    async def get_config(self, repo_id: str, revision: str = "main") -> dict[str, Any]:
        """Fetch and parse `config.json` of a repo."""
        resp = await self._client.get(f"/{validate_repo_id(repo_id)}/resolve/{revision}/config.json")
        if resp.status_code != 200:
            raise HubError(f"config.json for {repo_id}: HTTP {resp.status_code}")
        return resp.json()

    async def get_weights_size(self, repo_id: str) -> int | None:
        """Total bytes of the repo's `.safetensors` files, or None if unavailable.

        This is what MLX loads into memory (incl. vision towers, mixed-precision layers),
        so it is the most accurate weight-memory figure. Never raises.
        """
        try:
            resp = await self._client.get(
                f"/api/models/{validate_repo_id(repo_id)}", params={"blobs": "true"}
            )
            if resp.status_code != 200:
                return None
            files = resp.json().get("siblings") or []
        except httpx.HTTPError, HubError, ValueError, AttributeError:
            return None
        total = sum(f.get("size") or 0 for f in files if str(f.get("rfilename", "")).endswith(".safetensors"))
        return total or None

    async def get_model_card(self, repo_id: str, revision: str = "main") -> str:
        """Fetch the raw README.md (model card). Returns '' if absent."""
        resp = await self._client.get(f"/{validate_repo_id(repo_id)}/raw/{revision}/README.md")
        if resp.status_code == 404:
            return ""
        if resp.status_code != 200:
            raise HubError(f"README.md for {repo_id}: HTTP {resp.status_code}")
        return resp.text
