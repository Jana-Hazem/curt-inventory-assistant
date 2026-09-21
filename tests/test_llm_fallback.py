"""Tests for retry-related behaviour in app.llm: fallback model and friendly errors."""
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from app import llm


class ScriptedClient:
    """Fake Gemini client: each call takes the next scripted item (an exception or a response)."""

    def __init__(self, script):
        self.script = list(script)
        self.models_used = []
        self.models = self          # chat() calls client.models.generate_content(...)

    def generate_content(self, model, contents, config):
        self.models_used.append(model)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def text_response(text):
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], function_calls=None)


def server_error(code=503):
    return errors.ServerError(code, {"error": {"message": "temporary failure"}})


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "primary-model")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    llm.conversations.clear()
    yield
    llm.conversations.clear()


def test_no_fallback_configured_returns_friendly_error():
    client = ScriptedClient([server_error(503)])
    out = llm.chat("fb-1", "hi", client=client)
    assert "busy or timed out" in out["response"]
    assert client.models_used == ["primary-model"]


def test_fallback_model_is_used_after_transient_error(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup-model")
    client = ScriptedClient([server_error(503), text_response("Hello from backup")])
    out = llm.chat("fb-2", "hi", client=client)
    assert out["response"] == "Hello from backup"
    assert client.models_used == ["primary-model", "backup-model"]


def test_fallback_is_not_used_for_non_transient_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup-model")
    client = ScriptedClient([errors.ClientError(404, {"error": {"message": "model not found"}})])
    out = llm.chat("fb-3", "hi", client=client)
    assert "rejected" in out["response"]
    assert client.models_used == ["primary-model"]


def test_both_models_failing_still_returns_friendly_error(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup-model")
    client = ScriptedClient([server_error(503), server_error(504)])
    out = llm.chat("fb-4", "hi", client=client)
    assert "busy or timed out" in out["response"]
    assert client.models_used == ["primary-model", "backup-model"]


def test_fallback_equal_to_primary_is_ignored(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "primary-model")
    assert llm.get_fallback_model() is None


def test_cancelled_499_gets_the_busy_message():
    err = errors.ClientError(499, {"error": {"message": "cancelled"}})
    out = llm.chat("fb-5", "hi", client=ScriptedClient([err]))
    assert "busy or timed out" in out["response"]


def test_failed_turn_is_not_saved_so_a_resend_starts_clean():
    llm.chat("fb-6", "first try", client=ScriptedClient([server_error(503)]))
    assert "fb-6" not in llm.conversations