"""CURT Inventory Assistant: Streamlit frontend.

Phase 1 (rule-based) runs in this process and reads the database through the service layer.
Phase 2 (LLM) talks to the FastAPI backend over HTTP (POST /chat, GET /inventory).

Run (from the project root, with the backend already running):
    streamlit run streamlit_app.py
"""
import os
import uuid

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from app import phase1
from app.db import init_db, seed_db
from app.service import InventoryService
from app.tools import LOW_STOCK_THRESHOLD

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CHAT_TIMEOUT_S = 60          # the backend may retry Gemini a few times before answering
INVENTORY_TIMEOUT_S = 5
MAX_QUESTION_CHARS = 500

PHASE_1 = "Phase 1 (rule-based)"
PHASE_2 = "Phase 2 (LLM)"

EXAMPLES = {
    PHASE_1: [
        "How many brake pads do we have?",
        "Where is the ECU stored?",
        "List all items in Electronics",
        "how many break pads",
        "Where is turbocharger?",
    ],
    PHASE_2: [
        "How many ECUs do we have?",
        "Where are they stored?",
        "What do we have in the Braking category?",
        "Which parts are running low?",
        "Flag a shortage for the ECU",
    ],
}

st.set_page_config(page_title="CURT Inventory Assistant", page_icon=None, layout="wide")


# ---------- setup ----------
@st.cache_resource
def _setup_database() -> bool:
    """Create and seed the database once per Streamlit process (safe to run twice)."""
    init_db()
    seed_db()
    return True


def _init_state() -> None:
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    # A separate message list per phase, so switching never mixes the two histories.
    st.session_state.setdefault("messages", {PHASE_1: [], PHASE_2: []})


# ---------- inventory table ----------
def load_inventory(phase: str) -> tuple[list[dict], str | None]:
    """Phase 1 reads through the service directly; Phase 2 uses the API."""
    if phase == PHASE_1:
        return InventoryService().get_all_parts(), None
    try:
        response = requests.get(f"{API_URL}/inventory", timeout=INVENTORY_TIMEOUT_S)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException:
        return [], (f"The backend is not reachable at {API_URL}. "
                    "Start it with: uvicorn app.main:app --reload")


def _highlight_low_stock(row: pd.Series) -> list[str]:
    low = row["quantity"] <= LOW_STOCK_THRESHOLD
    return ["background-color: rgba(255, 75, 75, 0.25)" if low else ""] * len(row)


def render_inventory(parts: list[dict]) -> None:
    if not parts:
        st.info("No inventory to show.")
        return
    columns = ["name", "quantity", "category", "location"]
    df = pd.DataFrame(parts)
    df = df[[c for c in columns if c in df.columns]]
    if "quantity" in df.columns:
        st.dataframe(df.style.apply(_highlight_low_stock, axis=1), hide_index=True)
        st.caption(f"Red rows: low stock (quantity {LOW_STOCK_THRESHOLD} or less).")
    else:
        st.dataframe(df, hide_index=True)


# ---------- the two assistants ----------
def ask_phase1(question: str) -> dict:
    result = phase1.answer(question)
    text = result.get("response", "") if isinstance(result, dict) else str(result)
    return {"content": text}


def ask_phase2(question: str, session_id: str) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"session_id": session_id, "message": question},
            timeout=CHAT_TIMEOUT_S,
        )
    except requests.ConnectionError:
        return {"content": f"I can't reach the backend at {API_URL}. "
                           "Please start it with: uvicorn app.main:app --reload"}
    except requests.Timeout:
        return {"content": "The assistant took too long to answer. Please try again."}
    except requests.RequestException:
        return {"content": "Something went wrong while contacting the backend. Please try again."}

    if response.status_code in (400, 422):
        return {"content": "That message was not accepted. Please type a shorter, non-empty question."}
    if not response.ok:
        return {"content": "The backend returned an error. Please try again."}
    try:
        data = response.json()
    except ValueError:
        return {"content": "The backend sent an unreadable answer. Please try again."}
    return {"content": data.get("response", ""), "tool_calls": data.get("tool_calls") or []}


# ---------- rendering ----------
def render_tool_calls(tool_calls: list[dict] | None) -> None:
    """Shows which tool the model called, with what arguments and what came back."""
    if not tool_calls:
        return
    with st.expander(f"Tool calls ({len(tool_calls)})"):
        for call in tool_calls:
            st.markdown(f"**{call.get('name', 'unknown')}**")
            st.json({"args": call.get("args"), "result": call.get("result")})


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_tool_calls(message.get("tool_calls"))


# ---------- page ----------
def main() -> None:
    _setup_database()
    _init_state()

    with st.sidebar:
        st.header("CURT Inventory")
        phase = st.radio("Assistant", [PHASE_1, PHASE_2])

        if phase == PHASE_2:
            st.caption(f"Session: {st.session_state.session_id[:8]}")

        col_clear, col_refresh = st.columns(2)
        if col_clear.button("Clear chat"):
            st.session_state.messages[phase] = []
            if phase == PHASE_2:
                # A new session id also resets the backend's memory, so both sides match.
                st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        col_refresh.button("Refresh")   # any click reruns the script, which reloads the table

        st.subheader("Live inventory")
        parts, inventory_error = load_inventory(phase)
        if inventory_error:
            st.warning(inventory_error)
        else:
            render_inventory(parts)

        with st.expander("Try asking"):
            for example in EXAMPLES[phase]:
                st.markdown(f"- {example}")

    st.title("CURT Inventory Assistant")
    st.caption("Rule-based assistant, no AI" if phase == PHASE_1
               else "Gemini with tool calling and per-session memory")

    history = st.session_state.messages[phase]
    for message in history:
        render_message(message)

    question = st.chat_input("Ask about a part, its location, or a category",
                             max_chars=MAX_QUESTION_CHARS)
    if question and question.strip():
        question = question.strip()
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if phase == PHASE_1:
                    reply = ask_phase1(question)
                else:
                    reply = ask_phase2(question, st.session_state.session_id)
            st.markdown(reply["content"])
            render_tool_calls(reply.get("tool_calls"))
        history.append({"role": "assistant", **reply})


main()