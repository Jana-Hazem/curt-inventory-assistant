"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _no_fallback_model_from_env(monkeypatch):
    """app.llm calls load_dotenv() on import, so a developer's real .env can leak into the
    tests. Tests must not depend on it: no fallback model unless a test sets one itself."""
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)