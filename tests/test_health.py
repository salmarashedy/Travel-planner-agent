def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_are_accessible(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_health_does_not_require_ollama(client):
    assert client.get("/health").status_code == 200


def test_chat_returns_not_implemented(client):
    response = client.post(
        "/chat",
        json={"session_id": "session_12345", "message": "Plan a trip to Italy"},
    )
    assert response.status_code == 501
    assert response.json()["status"] == "error"
    assert "not been implemented" in response.json()["detail"]
