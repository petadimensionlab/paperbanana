"""Tests for the generic OpenAI-compatible local VLM provider."""

from __future__ import annotations

import pytest
from PIL import Image

from paperbanana.providers.vlm.openai_compatible import OpenAICompatibleVLM


class _FakeResponse:
    status_code = 200

    def __init__(self, text: str = "ok"):
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [{"message": {"content": self._text}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }


class _FakeClient:
    def __init__(self, text: str = "ok"):
        self.captured = None
        self._text = text
        self.headers = {}

    async def post(self, url, json=None, **kwargs):
        self.captured = {"url": url, "json": json}
        return _FakeResponse(self._text)


@pytest.fixture
def provider():
    return OpenAICompatibleVLM(
        provider_name="tensorfold",
        base_url="http://localhost:8080/v1",
        model="qwen3.5-35b-a3b",
        api_key="secret",
        json_mode=True,
        reasoning_effort="off",
        extra_body={"top_k": 20, "model": "should-not-overwrite"},
    )


@pytest.mark.asyncio
async def test_payload_includes_local_knobs(provider, monkeypatch):
    monkeypatch.setattr(
        "paperbanana.providers.vlm.openai_compatible.image_to_base64",
        lambda *_a, **_k: "b64",
    )
    client = _FakeClient("hello")
    provider._client = client
    result = await provider.generate(
        "Describe", images=[Image.new("RGB", (4, 4))], response_format="json"
    )
    assert result == "hello"
    payload = client.captured["json"]
    assert payload["model"] == "qwen3.5-35b-a3b"
    assert payload["reasoning_effort"] == "off"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["top_k"] == 20
    assert client.captured["url"] == "/chat/completions"


@pytest.mark.asyncio
async def test_reasoning_effort_omitted_when_unset():
    provider = OpenAICompatibleVLM(
        provider_name="omlx",
        base_url="http://localhost:8000/v1",
        model="m",
        reasoning_effort=None,
    )
    provider._client = _FakeClient()
    await provider.generate("hi")
    assert "reasoning_effort" not in provider._client.captured["json"]


def test_extract_message_text_raises_on_reasoning_only():
    with pytest.raises(RuntimeError, match="reasoning but no content"):
        OpenAICompatibleVLM._extract_message_text({"content": "", "reasoning": "thinking"})
