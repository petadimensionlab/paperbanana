"""Ollama VLM provider — local open-weight models via OpenAI-compatible API."""

from __future__ import annotations

from typing import Optional

import httpx
import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class OllamaVLM(VLMProvider):
    """VLM provider for locally-hosted models via Ollama."""

    def __init__(
        self,
        model: str = "qwen2.5-vl",
        base_url: str = "http://localhost:11434/v1",
        json_mode: bool = False,
        reasoning_effort: Optional[str] = "none",
        request_timeout: float = 900.0,
        image_max_edge: int = 1280,
        image_format: str = "JPEG",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._json_mode = json_mode
        self._reasoning_effort = reasoning_effort
        self._request_timeout = request_timeout
        self._image_max_edge = image_max_edge
        self._image_format = image_format.upper()
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_json_mode(self) -> bool:
        return self._json_mode

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._request_timeout,
            )
        return self._client

    def _prepare_image(self, image: Image.Image) -> str:
        """Downscale and encode an image for a local vision request.

        Local VLMs tile images, so sending a 2K reference figure at full size
        wastes both upload bandwidth and vision tokens.  The default 1280px
        long edge keeps labels legible while roughly halving the token count.
        """
        prepared = image
        if self._image_format == "JPEG" and prepared.mode != "RGB":
            prepared = prepared.convert("RGB")
        if self._image_max_edge and max(prepared.size) > self._image_max_edge:
            resized = prepared.copy()
            resized.thumbnail(
                (self._image_max_edge, self._image_max_edge),
                Image.Resampling.LANCZOS,
            )
            prepared = resized
        return image_to_base64(prepared, format=self._image_format)

    @staticmethod
    def _extract_message_text(message: dict) -> str:
        """Return assistant text, failing loudly when only reasoning was emitted."""
        text = message.get("content")
        if text is None or not str(text).strip():
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                raise RuntimeError(
                    "Ollama returned reasoning but no content. "
                    "The model spent its token budget thinking. Set "
                    "OLLAMA_REASONING_EFFORT=none (or use a non-thinking model)."
                )
            return ""
        return str(text)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_available(self) -> bool:
        try:
            root = self._base_url[:-3] if self._base_url.endswith("/v1") else self._base_url
            return httpx.get(root, timeout=3.0).status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=15))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/{self._image_format.lower()};base64,"
                        f"{self._prepare_image(img)}"
                    )
                },
            }
            for img in (images or [])
        ]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if response_format == "json" and self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = await self._get_client().post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage")
        logger.debug("Ollama response", model=self._model, usage=usage)

        if self.cost_tracker is not None and usage is not None:
            self.cost_tracker.record_vlm_call(
                provider=self.name,
                model=self._model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

        return self._extract_message_text(data["choices"][0]["message"])
