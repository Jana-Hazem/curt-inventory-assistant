# CURT Inventory Assistant

A two-phase inventory chatbot built for the CURT GenAI technical task. Phase 1 answers
stock, location and category questions with plain rule-based parsing and no AI. Phase 2
adds a Gemini-powered assistant with tool calling and per-session memory, served over a
FastAPI backend with a Streamlit frontend.

## Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Phase 1: rule-based assistant](#phase-1-rule-based-assistant)
- [Phase 2: Gemini assistant](#phase-2-gemini-assistant)
- [API reference](#api-reference)
- [Tool definitions](#tool-definitions)
- [Edge cases and design decisions](#edge-cases-and-design-decisions)
- [Reliability: retries and fallback model](#reliability-retries-and-fallback-model)
- [Security notes](#security-notes)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Reflection](#reflection)

## Architecture

```text
 ______________________    ______________________    ______________________
|   Streamlit UI       |  |   FastAPI backend     |  |   Gemini API          |
|   streamlit_app.py   |->|   app/main.py         |->|   (function call)     |
|                       |  |                       |  |                       |
|   Phase 1 toggle ------> |   POST /chat          |  |   tool calls          |
|   Phase 2 toggle ------> |   GET  /inventory     |<-|                       |
|_______________________|  |   GET  /health        |  |_______________________|
                            |________________________|
                                |            |
                                |            v
                                |    app/tools.py (execute_tool)
                                |            |
                                v            v
                    app/phase1.py    app/service.py (InventoryService)
                    (regex parser)           |
                            |                v
                            |->  app/db.py --> SQLite (curt_inventory.db)
```

Both entry points — Phase 1's regex parser and Phase 2's tools — call into
`InventoryService`, which is the only code in the project that runs SQL.

Two design choices shape everything else in this project:

1. **The model never touches SQL.** `InventoryService` in `app/service.py` is the only
   code that opens a database connection. Both Phase 1 (`app/phase1.py`) and Phase 2's
   tools (`app/tools.py`) call it and only it. Gemini never sees a connection string, a
   table name, or a query — only the three tool functions and their JSON results.
2. **Phase 1 and Phase 2 share the same matching logic.** `InventoryService.find_matches`
   does exact, normalized, partial and fuzzy matching for both phases, so "how many break
   pads" (Phase 1) and "how many ECUs" → "where are they stored" (Phase 2) are resolved
   the same way underneath.

## Setup

```bash
git clone <your-repo-url>
cd curt_inventory_assistant
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
```

Edit `.env` and add your key:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-flash-latest
GEMINI_FALLBACK_MODEL=
DB_PATH=curt_inventory.db
API_URL=http://localhost:8000
```

`GEMINI_FALLBACK_MODEL` is optional — see
[Reliability: retries and fallback model](#reliability-retries-and-fallback-model).

The database is created and seeded automatically (16 parts across five categories) the
first time the backend or Streamlit app starts; `app/db.py`'s `seed_db()` only inserts
rows when the table is empty, so restarts never duplicate data.

## Running the app

Two terminals, both with the virtual environment active:

```bash
# Terminal 1 — backend
uvicorn app.main:app --reload

# Terminal 2 — frontend
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Swagger docs for the API are at
`http://localhost:8000/docs`.

Command-line smoke tests, without the API or the UI:

```bash
python -m app.phase1   # Phase 1, no AI, no network
python -m app.llm      # Phase 2, talks to Gemini directly
```

## Phase 1: rule-based assistant

`app/phase1.py` parses intent with three ordered pattern groups — location, then
category, then stock — checked in that order because a phrase like "what do we have in
Electronics" would otherwise be misread as a stock question. It handles:

- **Stock**: "How many brake pads do we have?"
- **Location**: "Where is the ECU stored?"
- **Category**: "List all items in Electronics"

Every extracted item name goes through `InventoryService.find_matches`, which tries an
exact match, a normalized match (case, spacing, simple plurals), a partial match (every
typed word appears in the part name), and finally a fuzzy match, and reports which one
succeeded (`exact` / `normalized` / `partial` / `fuzzy`). Phase 1 has no AI and no network
dependency, so it works even if the Gemini API key is missing or the model is down.

## Phase 2: Gemini assistant

`app/llm.py` runs a manual tool-calling loop against `google-genai`, capped at 5 Gemini
calls per user message so a confused model can't loop forever. Automatic function calling
is explicitly disabled (`AutomaticFunctionCallingConfig(disable=True)`) so the app's own
loop — not the SDK — decides when a tool runs and logs every call.

Conversation history is kept per `session_id` in memory (`conversations: dict[str,
list[Content]]`), capped at the last 20 messages and trimmed only at a whole-turn boundary
so a function call is never separated from its response. A turn is only saved once the
whole exchange — including any tool calls — succeeds; a failed request leaves history
untouched, so resending the same message starts from the last good state rather than a
corrupted one.

## API reference

### `POST /chat`

```json
{
  "session_id": "demo1",
  "message": "How many ECUs do we have?"
}
```

```json
{
  "response": "We have 1 ECU (Electronics Cabinet). Stock is low — want me to flag a shortage?",
  "tool_calls": [
    {
      "name": "check_stock",
      "args": { "item_name": "ECU" },
      "result": {
        "found": true,
        "name": "ECU",
        "quantity": 1,
        "location": "Electronics Cabinet",
        "low_stock": true,
        "match_type": "exact"
      }
    }
  ]
}
```

`session_id` is validated against `^[A-Za-z0-9_-]{1,64}$` and `message` is capped at 500
characters and rejected if blank after stripping (both enforced by Pydantic in
`app/main.py`, before the message ever reaches Gemini).

### `GET /inventory`

Returns every part, ordered by category then name:

```json
[
  { "id": 5, "name": "ECU", "quantity": 1, "category": "Electronics", "location": "Electronics Cabinet" }
]
```

### `GET /health`

```json
{ "status": "ok" }
```

## Tool definitions

Gemini only ever sees these three functions, each taking one string argument. The model
never sees a database, only these results.

| Tool | Argument | Purpose |
|---|---|---|
| `check_stock` | `item_name` | Quantity, location and low-stock flag for one part; tolerant of misspellings and partial names |
| `list_by_category` | `category` | Every part in a category with quantity and location; returns available categories if the name doesn't match |
| `flag_shortage` | `item_name` | Records a low-stock flag (in-memory); only called when the user asks or agrees |

`execute_tool` in `app/tools.py` validates the tool name and its single string argument
before running anything, and catches any internal exception so a tool failure returns
`{"error": ...}` to the model instead of crashing the request.

## Edge cases and design decisions

| Edge case | Behavior | Where |
|---|---|---|
| Misspelled part name ("break pads") | Normalized/fuzzy match resolves it; response says "Showing results for Brake Pads." | `InventoryService.find_matches`, `match_type` |
| Ambiguous partial name (matches 2+ parts) | Asks "Did you mean X or Y?" instead of guessing | `find_matches` status `ambiguous` |
| Unknown part or category | States it wasn't found; category queries list the real categories | `phase1._lookup`, `tools.list_by_category` |
| Follow-up pronoun ("where are they stored?") | Resolved from conversation history, not re-asked | System prompt instruction + per-session `conversations` history |
| Vague quantity question ("how many do we have?") | Asks which item, rather than guessing one | `phase1.answer`, "Which item would you like to check?" |
| Off-topic question ("what's the weather?") | Politely declines and restates what it can help with | System prompt (Phase 2) / `HELP_MESSAGE` (Phase 1) |
| Zero-quantity part | Reported as "We're out of X" rather than "we have 0 X" | `phase1.answer` |
| Low stock | Flagged in the response; a shortage is only recorded after the user asks or agrees | `LOW_STOCK_THRESHOLD` in `tools.py`, system prompt rule |
| Empty or whitespace-only message | Rejected before reaching the model (Pydantic validator) | `ChatRequest.not_blank` in `main.py` |
| Oversized or malformed tool arguments from the model | Rejected with a clear error, tool never runs | `execute_tool` in `tools.py` |
| Gemini call fails (429/499/503/504) | Automatic retry with backoff; friendly message if all retries fail | See below |
| Server restarts | Conversation memory and shortage flags reset (both are in-memory by design, documented as a limitation) | `conversations`, `_flagged` |

## Reliability: retries and fallback model

Gemini occasionally returns transient errors under load: `503 UNAVAILABLE` ("high
demand"), `504 DEADLINE_EXCEEDED`, `499 CANCELLED`, or `429` rate limits. `app/llm.py`
handles this in two layers:

1. **SDK-level retry.** The client is configured with `HttpRetryOptions` (4 attempts,
   exponential backoff from 1s to 8s) for exactly those status codes, so a single slow or
   overloaded response is usually retried and resolved without the caller noticing.
2. **Optional fallback model.** If every retry on the primary model still fails with a
   transient error, and `GEMINI_FALLBACK_MODEL` is set in `.env`, the app tries that model
   once for the same request. It is not used for non-transient errors (e.g. an invalid API
   key or an unknown model name), since retrying those would only waste time.

If both layers are exhausted, the user gets a clear, non-technical message ("The AI
service is busy or timed out... please send your message again") and nothing is written
to that session's history, so resending is always safe.

## Security notes

- **No SQL reaches the model.** `InventoryService` is the sole point of database access;
  tools and Phase 1 both go through it.
- **Input validation on every boundary.** `ChatRequest` validates `session_id` format and
  message length/blankness before Gemini is called; `execute_tool` re-validates every
  argument Gemini sends before running a tool.
- **Secrets are never committed.** `.env` and `*.db` are in `.gitignore`; `.env.example`
  ships with placeholder values only.
- **Errors are never leaked verbatim.** Tool and Gemini exceptions are logged server-side
  and converted to generic, user-safe messages (`_friendly_error` in `llm.py`,
  `except Exception` in `execute_tool`).

## Testing

```bash
python -m pytest -q
```

Covers the service layer's matching logic, Phase 1's intent parsing, the Gemini tool loop
(including the retry/fallback behavior, using a scripted fake client), and the FastAPI
endpoints via `TestClient`.

## Known limitations

- Conversation memory and shortage flags are in-memory and reset when the server restarts.
- Phase 1's pattern matching is rule-based and won't generalize to phrasing outside its
  patterns the way Phase 2 does.
- The fallback model, if configured, is tried only after the primary model's retries are
  exhausted — it adds latency to the worst case rather than running in parallel.

## Reflection

The main edge-case decision was **where to draw the line between "ask for clarification"
and "guess."** Both phases refuse to guess a specific part when the name is ambiguous or
missing — Phase 1 asks "Which item?", Phase 2's system prompt instructs the same — because
a wrong inventory number is worse than one extra question. The one place a guess is
allowed is fuzzy matching a *single* clear candidate (e.g. "ECU" for "ecus"), and even then
the response says which name it matched against, so the user can catch a wrong guess
immediately.

The second significant decision was **how to handle Gemini's own reliability**, which only
showed up once the app was running against the live API: transient 503/504/499 errors
happened in normal use, not just in theory. Retrying with backoff, and falling back to a
second model only for those specific transient errors, kept the assistant answering
through temporary outages without masking real failures (like a bad API key) behind a
retry loop.
