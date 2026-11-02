"""VLM routing helpers: explicit local fallback between providers."""

from __future__ import annotations

from typing import Optional

import structlog
from PIL import Image

from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class FallbackVLM(VLMProvider):
    """Try a primary local/self-hosted VLM, then an explicit fallback.

    The fallback is only constructed by the registry for self-hosted providers.
    Cloud providers are never used as an implicit fallback.
    """

    def __init__(self, primary: VLMProvider, fallback: VLMProvider):
        self._primary = primary
        self._fallback = fallback

    @property
    def name(self) -> str:
        return self._primary.name

    @property
    def model_name(self) -> str:
        return self._primary.model_name

    @property
    def supports_json_mode(self) -> bool:
        return self._primary.supports_json_mode

    @property
    def fallback_name(self) -> str:
        return self._fallback.name

    def is_available(self) -> bool:
        return self._primary.is_available() or self._fallback.is_available()

    async def close(self) -> None:
        for provider in (self._primary, self._fallback):
            close = getattr(provider, "close", None)
            if close is not None:
                await close()

    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        for provider in (self._primary, self._fallback):
            if self.cost_tracker is not None and provider.cost_tracker is None:
                provider.cost_tracker = self.cost_tracker
        try:
            return await self._primary.generate(
                prompt=prompt,
                images=images,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as exc:
            logger.warning(
                "Primary VLM failed; using local fallback",
                primary=self._primary.name,
                primary_model=self._primary.model_name,
                fallback=self._fallback.name,
                fallback_model=self._fallback.model_name,
                error=str(exc),
            )
            return await self._fallback.generate(
                prompt=prompt,
                images=images,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
