import json

import httpx
import pytest

from core.config import Settings
from core.exceptions import LLMClientError
from services.llm_client import OllamaClient


def make_client(handler) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    return OllamaClient(Settings(), client=httpx.AsyncClient(transport=transport))


async def test_chat_structured_sends_schema_and_returns_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '{"message_type": "yes"}'}})

    client = make_client(handler)
    schema = {"type": "object"}
    result = await client.chat_structured([{"role": "user", "content": "yes"}], schema)

    assert result == '{"message_type": "yes"}'
    assert seen["url"].endswith("/api/chat")
    body = seen["body"]
    assert body["format"] == schema
    assert body["think"] is False
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert body["options"]["seed"] == 42
    assert body["options"]["num_ctx"] == 8192


async def test_chat_structured_raises_when_ollama_is_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMClientError):
        await make_client(handler).chat_structured([], {})


async def test_chat_structured_raises_on_bad_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(LLMClientError):
        await make_client(handler).chat_structured([], {})


async def test_chat_structured_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(LLMClientError):
        await make_client(handler).chat_structured([], {})
