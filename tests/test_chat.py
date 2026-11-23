import pytest


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "short", "message": "Hello"},
        {"session_id": "session_123", "message": ""},
        {"session_id": "session_123", "message": "x" * 1001},
    ],
)
def test_invalid_chat_requests_are_rejected(client, payload):
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_valid_chat_request_accepts_optional_user_id(client):
    response = client.post(
        "/chat",
        json={
            "session_id": "session_12345",
            "message": "I want to visit Italy",
            "user_id": "user_123",
        },
    )
    assert response.status_code == 501
