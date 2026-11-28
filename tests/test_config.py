from core.config import Settings


def test_settings_load_defaults():
    settings = Settings()
    assert settings.app_name == "Travel Agent"
    assert settings.app_version == "0.1.0"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "qwen3:4b-thinking-2507-q4_K_M"
    assert settings.ollama_timeout == 60
    assert settings.ollama_temperature == 0
    assert settings.ollama_seed == 42
    assert settings.ollama_num_ctx == 8192
    assert settings.database_url is None


def test_settings_accept_environment_overrides(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Configured Travel Agent")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "15")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/travel")
    settings = Settings()
    assert settings.app_name == "Configured Travel Agent"
    assert settings.ollama_timeout == 15
    assert settings.database_url == "postgresql://localhost/travel"
