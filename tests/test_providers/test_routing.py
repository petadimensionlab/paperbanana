"""Tests for per-role VLM routing and explicit local fallback."""

from __future__ import annotations

import pytest

from paperbanana.core.config import Settings
from paperbanana.providers.registry import ProviderRegistry
from paperbanana.providers.vlm.routing import FallbackVLM


def _settings(**routing):
    return Settings(
        vlm_provider="routing",
        vlm_routing=routing,
    )


def test_routing_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        """
vlm:
  provider: routing
  routing:
    default:
      provider: ollama
      base_url: http://127.0.0.1:11434/v1
      model: qwen3.5:9b
    structurer:
      provider: tensorfold
      base_url: http://127.0.0.1:8080/v1
      model: qwen3.6-35b-a3b
""",
        encoding="utf-8",
    )
    settings = Settings.from_yaml(path)
    assert settings.vlm_provider == "routing"
    assert settings.vlm_routing["structurer"]["model"] == "qwen3.6-35b-a3b"


def test_role_selection():
    settings = _settings(
        default={"provider": "ollama", "model": "qwen3.5:9b"},
        structurer={
            "provider": "tensorfold",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "qwen3.6-35b-a3b",
        },
    )
    default = ProviderRegistry.create_vlm(settings, role="planner")
    assert default.name == "ollama"
    assert default.model_name == "qwen3.5:9b"

    structurer = ProviderRegistry.create_vlm(settings, role="structurer")
    assert structurer.name == "tensorfold"
    assert structurer.model_name == "qwen3.6-35b-a3b"


def test_self_hosted_fallback_wrapped():
    settings = _settings(
        default={
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "pb-quality",
        },
        fallback={
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen3.5:9b",
        },
    )
    provider = ProviderRegistry.create_vlm(settings, role="critic")
    assert isinstance(provider, FallbackVLM)
    assert provider.name == "ollama"
    assert provider.fallback_name == "ollama"


def test_paid_fallback_rejected():
    settings = _settings(
        default={"provider": "ollama", "model": "qwen3.5:9b"},
        fallback={"provider": "openai", "model": "gpt-5.2"},
    )
    with pytest.raises(ValueError, match="fallback must be a self-hosted provider"):
        ProviderRegistry.create_vlm(settings, role="planner")


class _Failing:
    name = "primary"
    model_name = "primary"
    supports_json_mode = True
    cost_tracker = None

    async def generate(self, **kwargs):
        raise RuntimeError("primary down")


class _Working:
    name = "fallback"
    model_name = "fallback"
    supports_json_mode = True
    cost_tracker = None

    async def generate(self, **kwargs):
        return "fallback answer"


@pytest.mark.asyncio
async def test_fallback_generate():
    provider = FallbackVLM(_Failing(), _Working())
    result = await provider.generate("hi")
    assert result == "fallback answer"


def test_lmstudio_provider_alias():
    settings = Settings(
        vlm_provider="lmstudio",
        lmstudio_model="qwen/qwen3.6-35b-a3b",
    )
    provider = ProviderRegistry.create_vlm(settings)
    assert provider.name == "lmstudio"
    assert provider.model_name == "qwen/qwen3.6-35b-a3b"
    assert provider.supports_json_mode is False
