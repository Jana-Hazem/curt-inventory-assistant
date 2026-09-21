import pytest
from google.genai import errors, types

from app import db, llm, tools


# ---------- a scripted fake Gemini client (no network, no API key) ----------
class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []                       # the `contents` sent on each call

    def generate_content(self, model, contents, config):
        self.calls.append(list(contents))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def text_response(text):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]))])


def call_response(*calls):
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=parts))])


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    db.seed_db()
    tools.reset_flags()
    llm.conversations.clear()


# ---------- the loop ----------
def test_tool_round_trip():
    client = FakeClient([
        call_response(("check_stock", {"item_name": "ECU"})),
        text_response("We have 1 ECU. Stock is low; want me to flag it?"),
    ])
    out = llm.chat("s1", "how many ECUs?", client=client)
    assert out["response"].startswith("We have 1 ECU")
    assert out["tool_calls"][0]["name"] == "check_stock"
    assert out["tool_calls"][0]["result"]["quantity"] == 1
    # the second model call received the tool result as a function_response
    last = client.models.calls[1][-1]
    assert last.role == "user" and last.parts[0].function_response.response["quantity"] == 1


def test_follow_up_uses_session_history():
    client = FakeClient([text_response("Brake Pads: 12."), text_response("Mechanical Workshop.")])
    llm.chat("s1", "how many brake pads?", client=client)
    llm.chat("s1", "where are they stored?", client=client)
    second_call = client.models.calls[1]
    assert second_call[0].parts[0].text == "how many brake pads?"
    assert second_call[-1].parts[0].text == "where are they stored?"


def test_sessions_are_isolated():
    client = FakeClient([text_response("a"), text_response("b")])
    llm.chat("s1", "hello", client=client)
    llm.chat("s2", "hi", client=client)
    assert len(client.models.calls[1]) == 1


def test_unknown_tool_is_rejected_and_loop_recovers():
    client = FakeClient([call_response(("drop_table", {"x": "1"})), text_response("Sorry.")])
    out = llm.chat("s1", "do something", client=client)
    assert "error" in out["tool_calls"][0]["result"]
    assert out["response"] == "Sorry."


def test_parallel_tool_calls_are_all_answered():
    client = FakeClient([
        call_response(("check_stock", {"item_name": "ECU"}), ("check_stock", {"item_name": "Tire"})),
        text_response("done"),
    ])
    out = llm.chat("s1", "ECU and tires?", client=client)
    assert len(out["tool_calls"]) == 2
    assert len(client.models.calls[1][-1].parts) == 2


def test_max_iterations_stops_infinite_loop():
    client = FakeClient([call_response(("check_stock", {"item_name": "ECU"}))
                         for _ in range(llm.MAX_TOOL_ITERATIONS)])
    out = llm.chat("s1", "loop forever", client=client)
    assert "couldn't finish" in out["response"]
    assert len(client.models.calls) == llm.MAX_TOOL_ITERATIONS
    assert "s1" not in llm.conversations          # failed turn is not saved


# ---------- errors never crash the API ----------
def test_quota_error_is_friendly():
    err = errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    out = llm.chat("s1", "hi", client=FakeClient([err]))
    assert "too many requests" in out["response"]
    assert "s1" not in llm.conversations


def test_unexpected_error_does_not_leak_details():
    out = llm.chat("s1", "hi", client=FakeClient([RuntimeError("secret-key-123")]))
    assert "secret" not in out["response"]
    assert out["response"].startswith("Sorry")


def test_empty_model_response_is_handled():
    empty = types.GenerateContentResponse(candidates=[])
    out = llm.chat("s1", "hi", client=FakeClient([empty]))
    assert "couldn't generate" in out["response"]


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_client", None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.get_client()


# ---------- memory limits ----------
def user_text(t):
    return types.Content(role="user", parts=[types.Part(text=t)])


def test_history_is_capped_and_starts_on_a_user_message():
    client = FakeClient([text_response(f"r{i}") for i in range(15)])
    for i in range(15):
        llm.chat("s1", f"q{i}", client=client)
    history = llm.conversations["s1"]
    assert len(history) <= llm.MAX_HISTORY_MESSAGES
    assert history[0].role == "user" and history[0].parts[0].text


def test_trim_never_starts_on_a_function_response():
    model_call = types.Content(role="model", parts=[types.Part(
        function_call=types.FunctionCall(name="check_stock", args={"item_name": "ECU"}))])
    fn_response = types.Content(role="user", parts=[types.Part(
        function_response=types.FunctionResponse(name="check_stock", response={"quantity": 1}))])
    model_text = types.Content(role="model", parts=[types.Part(text="ok")])
    history = [user_text("q"), model_call, fn_response, model_text]
    history += [user_text("q"), model_text] * 9        # 22 messages: cut lands on fn_response
    trimmed = llm._trim(history)
    assert len(trimmed) <= llm.MAX_HISTORY_MESSAGES
    assert llm._is_user_text(trimmed[0])


def test_session_cap_drops_oldest(monkeypatch):
    monkeypatch.setattr(llm, "MAX_SESSIONS", 2)
    client = FakeClient([text_response("x") for _ in range(3)])
    for sid in ("a", "b", "c"):
        llm.chat(sid, "hi", client=client)
    assert list(llm.conversations) == ["b", "c"]


# ---------- config ----------
def test_config_disables_automatic_function_calling():
    cfg = llm._build_config()
    assert cfg.automatic_function_calling.disable is True
    assert len(cfg.tools[0].function_declarations) == 3
    assert "no direct access" in cfg.system_instruction