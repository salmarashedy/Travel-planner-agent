import asyncio

import pytest

from agents.reply import CONFIRMED, FALLBACK, QUESTIONS
from core.exceptions import LLMClientError
from schemas.chat import ChatRequest
from services.chat_service import ERROR_REPLY


def send(client, message, session="session_12345", **extra):
    response = client.post("/chat", json={"session_id": session, "message": message, **extra})
    assert response.status_code == 200, response.text
    return response.json()


# ---------- input checks (unchanged contract) ----------

@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "short", "message": "Hello"},
        {"session_id": "session_123", "message": ""},
        {"session_id": "session_123", "message": "x" * 1001},
        {"session_id": "session_123", "message": "hi", "extra": "field"},
    ],
)
def test_invalid_chat_requests_are_rejected(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_optional_user_id_is_accepted(client):
    body = send(client, "hi", user_id="user_123")
    assert body["status"] == "collecting"


# ---------- real replies ----------

def test_first_message(client):
    body = send(client, "7 days in Italy, 1500 euros, 4 stars")
    assert body["reply"] == f"{QUESTIONS['start_date']} {QUESTIONS['party_size']}"
    assert body["status"] == "collecting"
    assert body["trip_spec"]["country"] == "Italy"
    assert body["trip_spec"]["budget_amount"] == 1500
    assert body["trip_spec"]["start_date"] is None


def test_the_four_turn_example_through_the_api(client):
    send(client, "7 days in Italy, 1500 euros, 4 stars")
    body = send(client, "June 10, 2 people")
    assert body["reply"].startswith("Here is what I have:")
    assert body["status"] == "collecting"

    send(client, "actually make it 2000")
    body = send(client, "yes")
    assert body["reply"] == CONFIRMED
    assert body["status"] == "confirmed"
    assert body["trip_spec"] == {
        "country": "Italy",
        "region": None,
        "start_date": "2027-06-10",
        "num_days": 7,
        "budget_amount": 2000.0,
        "budget_currency": "EUR",
        "budget_scope": "total",
        "party_size": 2,
        "hotel_stars": 4,
        "min_review_score": None,
        "interests": [],
        "from_memory": [],
    }


def test_sessions_do_not_mix(client):
    send(client, "7 days in Italy, 1500 euros, 4 stars", session="session_one")
    body = send(client, "hi", session="session_two")
    assert body["trip_spec"]["country"] is None


# ---------- when things go wrong ----------

def test_model_down_gives_a_friendly_reply(client, fake_llm):
    fake_llm.answers["7 days in Italy, 1500 euros, 4 stars"] = LLMClientError("down")
    body = send(client, "7 days in Italy, 1500 euros, 4 stars")
    assert body["reply"] == FALLBACK
    assert body["status"] == "collecting"
    assert body["trip_spec"] is not None


def test_unexpected_crash_is_not_a_500(client, fake_llm):
    fake_llm.answers["boom"] = RuntimeError("something broke")
    body = send(client, "boom")
    assert body == {"reply": ERROR_REPLY, "trip_spec": None, "status": "error"}


def test_chat_works_again_after_a_crash(client, fake_llm):
    fake_llm.answers["boom"] = RuntimeError("something broke")
    send(client, "boom")
    assert send(client, "hi")["status"] == "collecting"


# ---------- two messages at once ----------

async def test_messages_sent_at_once_to_one_session_are_both_kept(chat_service, fake_llm):
    fake_llm.delay = 0.05  # long enough for the two turns to overlap without the lock
    first = ChatRequest(session_id="session_race", message="7 days in Italy, 1500 euros, 4 stars")
    second = ChatRequest(session_id="session_race", message="June 10, 2 people")

    await asyncio.gather(chat_service.handle(first), chat_service.handle(second))

    state = await chat_service.graph.aget_state({"configurable": {"thread_id": "session_race"}})
    assert len(state.values["messages"]) == 4
    spec = state.values["spec"]
    assert spec["country"] == "Italy" and spec["party_size"] == 2
