from __future__ import annotations

from typing import Any

import httpx

from core.config import Settings
from core.exceptions import LLMClientError


class OllamaClient:
    """Small transport-only client for Ollama; prompts and agent logic stay elsewhere."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.temperature = settings.ollama_temperature
        self.seed = settings.ollama_seed
        self.num_ctx = settings.ollama_num_ctx
        self.timeout = settings.ollama_timeout
        self._client = client

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system

        try:
            if self._client is not None:
                response = await self._client.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate", json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMClientError(f"Unable to generate text with Ollama: {exc}") from exc

        result = data.get("response")
        if not isinstance(result, str):
            raise LLMClientError("Ollama returned an invalid response payload.")
        return result

    async def generate_structured(
        self, prompt: str, *, system: str | None = None
    ) -> dict[str, Any]:
        """Request JSON output without imposing travel-specific schema or business rules."""
        payload_prompt = f"{prompt}\nReturn only valid JSON."
        raw = await self._generate_with_format(payload_prompt, system=system)
        try:
            import json

            result = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMClientError("Ollama returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise LLMClientError("Ollama structured output must be a JSON object.")
        return result

    async def _generate_with_format(
        self, prompt: str, *, system: str | None = None
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system
        try:
            if self._client is not None:
                response = await self._client.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate", json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMClientError(f"Unable to generate structured output: {exc}") from exc
        result = data.get("response")
        if not isinstance(result, str):
            raise LLMClientError("Ollama returned an invalid response payload.")
        return result
