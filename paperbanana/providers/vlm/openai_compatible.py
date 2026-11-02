"""Generic OpenAI-compatible VLM provider for local/self-hosted servers.

Works with Ollama, oMLX, TensorFold, vLLM, and any other server that implements
``POST /v1/chat/completions`` with OpenAI-style image parts.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()

_PROTECTED_PAYLOAD_KEYS = {"model", "messages", "max_tokens", "temperature"}


class OpenAICompatibleVLM(VLMProvider):
    """VLM provider for any OpenAI-compatible chat-completions endpoint.

    The provider deliberately uses httpx directly instead of the OpenAI SDK so
    that local runtimes with non-standard fields (``reasoning_effort``,
    ``chat_template_kwargs``, ``top_k``) can be configured without installing an
    extra dependency.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        json_mode: bool = False,
        reasoning_effort: Optional[str] = None,
        request_timeout: float = 900.0,
        image_max_edge: int = 1280,
        image_format: str = "JPEG",
        extra_body: Optional[dict[str, Any]] = None,
    ):
        self._provider_name = provider_name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._json_mode = json_mode
        self._reasoning_effort = reasoning_effort
        self._request_timeout = request_timeout
        self._image_max_edge = image_max_edge
        self._image_format = image_format.upper()
        self._extra_body = dict(extra_body or {})
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_json_mode(self) -> bool:
        return self._json_mode

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._request_timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_available(self) -> bool:
        """Best-effort health check against the server's model list."""
        try:
            # Provider selection is synchronous, so use a short-lived sync
            # request rather than the async client.
            import httpx as _httpx

            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            resp = _httpx.get(
                f"{self._base_url}/models",
                headers=headers,
                timeout=min(5.0, self._request_timeout),
            )
            return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    def _prepare_image(self, image: Image.Image) -> str:
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
        text = message.get("content")
        if text is None or not str(text).strip():
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                raise RuntimeError(
                    "The local model returned reasoning but no content. "
                    "Disable thinking (reasoning_effort=off/none or "
                    "chat_template_kwargs.enable_thinking=false)."
                )
            return ""
        return str(text)

    def _payload(
        self,
        prompt: str,
        images: Optional[list[Image.Image]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        response_format: Optional[str],
    ) -> dict:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/{self._image_format.lower()};base64,{self._prepare_image(img)}"
                    )
                },
            }
            for img in (images or [])
        ]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if response_format == "json" and self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        # Non-standard knobs (chat_template_kwargs, top_k, ...). Never let them
        # replace the core request fields.
        for key, value in self._extra_body.items():
            if key not in _PROTECTED_PAYLOAD_KEYS:
                payload[key] = value
        return payload

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        payload = self._payload(
            prompt,
            images,
            system_prompt,
            temperature,
            max_tokens,
            response_format,
        )
        resp = await self._get_client().post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage")
        logger.debug(
            "OpenAI-compatible response",
            provider=self.name,
            model=self._model,
            usage=usage,
        )
        if self.cost_tracker is not None and usage is not None:
            self.cost_tracker.record_vlm_call(
                provider=self.name,
                model=self._model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        return self._extract_message_text(data["choices"][0]["message"])
