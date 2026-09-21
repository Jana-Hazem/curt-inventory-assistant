import json
import logging

import pytest

from app import db, tools


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    db.seed_db()
    tools.reset_flags()


# ---------- check_stock ----------
def test_check_stock_found():
    r = tools.check_stock("Brake Pads")
    assert r["found"] is True
    assert (r["name"], r["quantity"], r["location"]) == ("Brake Pads", 12, "Mechanical Workshop")
    assert r["low_stock"] is False


def test_check_stock_low_stock_flag():
    assert tools.check_stock("ECU")["low_stock"] is True
    assert tools.check_stock("Tire")["low_stock"] is False


def test_check_stock_misspelling_resolves():
    r = tools.check_stock("break pad")
    assert r["found"] and r["name"] == "Brake Pads" and r["match_type"] == "fuzzy"


def test_check_stock_unknown_returns_not_found():
    assert tools.check_stock("turbocharger") == {"found": False, "suggestions": []}


def test_check_stock_ambiguous_returns_suggestions():
    r = tools.check_stock("brake")
    assert r["found"] is False and "Brake Pads" in r["suggestions"]


# ---------- list_by_category ----------
def test_list_by_category_found_case_insensitive():
    r = tools.list_by_category("electronics")
    assert r["found"] and r["category"] == "Electronics" and len(r["items"]) == 4


def test_list_by_category_unknown():
    r = tools.list_by_category("engine")
    assert r["found"] is False and "Braking" in r["available_categories"]


# ---------- flag_shortage ----------
def test_flag_shortage_logs_and_is_idempotent(caplog):
    with caplog.at_level(logging.WARNING, logger="curt.tools"):
        first = tools.flag_shortage("ECU")
        second = tools.flag_shortage("ecu")
    assert first == {"status": "flagged", "item": "ECU", "already_flagged": False}
    assert second["already_flagged"] is True
    assert sum("Low stock flag for ECU" in m for m in caplog.messages) == 1


def test_flag_shortage_unknown_item():
    assert tools.flag_shortage("turbocharger")["status"] == "not_found"


# ---------- execute_tool (validation) ----------
def test_execute_tool_runs_registered_tool():
    assert tools.execute_tool("check_stock", {"item_name": "ECU"})["quantity"] == 1


@pytest.mark.parametrize("name, args", [
    ("drop_table", {"item_name": "x"}),            # unknown tool
    ("check_stock", {}),                            # missing argument
    ("check_stock", {"item_name": 5}),              # wrong type
    ("check_stock", {"item_name": "   "}),          # empty
    ("check_stock", {"item_name": "x" * 500}),      # too long
    ("check_stock", {"item_name": "ECU", "extra": 1}),  # unexpected argument
    ("list_by_category", {"item_name": "ECU"}),     # wrong argument name
])
def test_execute_tool_rejects_bad_calls(name, args):
    assert "error" in tools.execute_tool(name, args)


def test_execute_tool_does_not_leak_exceptions(monkeypatch):
    def boom(_):
        raise RuntimeError("secret internals")
    monkeypatch.setitem(tools.TOOL_REGISTRY, "check_stock", boom)
    result = tools.execute_tool("check_stock", {"item_name": "ECU"})
    assert result == {"error": "The tool failed. Please try again."}


# ---------- declarations ----------
def test_declarations_match_registry_and_are_json():
    names = {d["name"] for d in tools.TOOL_DECLARATIONS}
    assert names == set(tools.TOOL_REGISTRY)
    json.dumps(tools.TOOL_DECLARATIONS)


def test_declarations_load_in_gemini_sdk():
    from google.genai import types
    tool = types.Tool(function_declarations=tools.TOOL_DECLARATIONS)
    assert len(tool.function_declarations) == 3