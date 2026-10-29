import httpx
import pytest

from mlx_model_scout.app.hf_client import HFClient, HubError


def _transport():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models":
            assert request.url.params["author"] == "mlx-community"
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "mlx-community/A-4bit",
                        "lastModified": "2026-10-01T10:00:00.000Z",
                        "downloads": 5,
                    },
                    {"modelId": "mlx-community/B-8bit"},
                ],
            )
        if request.url.path.endswith("/config.json"):
            return httpx.Response(200, json={"model_type": "llama"})
        if request.url.path.endswith("/README.md"):
            return httpx.Response(404)
        return httpx.Response(500)

    return httpx.MockTransport(handler)


async def test_list_models_parses_payload():
    async with HFClient(transport=_transport()) as hf:
        models = await hf.list_models("mlx-community")
    assert [m.repo_id for m in models] == ["mlx-community/A-4bit", "mlx-community/B-8bit"]
    assert models[0].last_modified.year == 2026


async def test_config_and_missing_card():
    async with HFClient(transport=_transport()) as hf:
        assert (await hf.get_config("mlx-community/A-4bit"))["model_type"] == "llama"
        assert await hf.get_model_card("mlx-community/A-4bit") == ""


async def test_http_error_raises():
    transport = httpx.MockTransport(lambda r: httpx.Response(503))
    async with HFClient(transport=transport) as hf:
        with pytest.raises(HubError):
            await hf.list_models("mlx-community")


async def test_weights_size_sums_safetensors_and_never_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models/a/b":
            siblings = [
                {"rfilename": "model-00001.safetensors", "size": 3},
                {"rfilename": "model-00002.safetensors", "size": 4},
                {"rfilename": "README.md", "size": 100},
            ]
            return httpx.Response(200, json={"siblings": siblings})
        return httpx.Response(404)

    async with HFClient(transport=httpx.MockTransport(handler)) as hf:
        assert await hf.get_weights_size("a/b") == 7
        assert await hf.get_weights_size("a/missing") is None
