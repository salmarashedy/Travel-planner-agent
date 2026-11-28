from datetime import date

import pytest
from fastapi.testclient import TestClient

from agents.graph import build_graph
from main import create_app
from services.chat_service import ChatService
from tests.fakes import CHAT, FakeLLM

TODAY = date(2026, 11, 21)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(dict(CHAT))


@pytest.fixture
def chat_service(fake_llm) -> ChatService:
    return ChatService(build_graph(fake_llm, today=lambda: TODAY))


@pytest.fixture
def client(chat_service) -> TestClient:
    with TestClient(create_app(chat_service=chat_service)) as test_client:
        yield test_client
