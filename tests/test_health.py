def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_are_accessible(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_health_does_not_require_ollama(client, fake_llm):
    assert client.get("/health").status_code == 200
    assert fake_llm.calls == []


def test_real_app_starts_without_contacting_ollama(monkeypatch):
    # Point at an address where nothing runs: building the app must not call it.
    from fastapi.testclient import TestClient

    from core.config import get_settings
    from main import create_app

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            assert client.get("/health").json() == {"status": "ok"}
    finally:
        get_settings.cache_clear()
