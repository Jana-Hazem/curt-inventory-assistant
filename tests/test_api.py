import pytest
from fastapi.testclient import TestClient

from app import llm
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with TestClient(app) as test_client:      # runs startup: init + seed the temp DB
        yield test_client


def fake_chat(session_id, message, client=None):
    fake_chat.seen = (session_id, message)
    return {
        "response": "We have 1 ECU.",
        "tool_calls": [{"name": "check_stock", "args": {"item_name": "ECU"},
                        "result": {"found": True, "quantity": 1}}],
    }


# ---------- happy paths ----------
def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_inventory_returns_seeded_parts(client):
    r = client.get("/inventory")
    assert r.status_code == 200
    parts = r.json()
    assert len(parts) == 16
    assert set(parts[0]) == {"id", "name", "quantity", "category", "location"}


def test_chat_success(client, monkeypatch):
    monkeypatch.setattr(llm, "chat", fake_chat)
    r = client.post("/chat", json={"session_id": "abc-123", "message": "  how many ECUs?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "We have 1 ECU."
    assert body["tool_calls"][0]["name"] == "check_stock"
    assert fake_chat.seen == ("abc-123", "how many ECUs?")     # message was stripped


# ---------- validation ----------
@pytest.mark.parametrize("payload", [
    {"session_id": "s1", "message": ""},                        # empty
    {"session_id": "s1", "message": "     "},                   # whitespace only
    {"session_id": "s1", "message": "x" * 501},                 # too long
    {"session_id": "s1"},                                       # missing message
    {"message": "hi"},                                          # missing session
    {"session_id": "bad id!", "message": "hi"},                 # illegal characters
    {"session_id": "a" * 65, "message": "hi"},                  # session id too long
])
def test_chat_rejects_bad_input(client, monkeypatch, payload):
    monkeypatch.setattr(llm, "chat", fake_chat)
    assert client.post("/chat", json=payload).status_code == 422


# ---------- errors never leak ----------
def test_chat_unexpected_error_returns_500_without_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret-internal-detail")
    monkeypatch.setattr(llm, "chat", boom)
    r = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert r.status_code == 500
    assert "secret" not in r.text


def test_startup_fails_clearly_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        with TestClient(app):
            pass


def test_docs_are_available(client):
    assert client.get("/docs").status_code == 200