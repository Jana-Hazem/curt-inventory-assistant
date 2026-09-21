"""FastAPI backend: POST /chat (Gemini + tools) and GET /inventory."""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

load_dotenv()  # load .env before the app modules read their settings

from app import llm  # noqa: E402
from app.db import init_db, seed_db  # noqa: E402
from app.service import InventoryService  # noqa: E402

logger = logging.getLogger("curt.api")

MAX_MESSAGE_LENGTH = 500


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    init_db()
    seed_db()   # safe to run on every start: only inserts when the table is empty
    yield


# ---------- request / response models ----------
class ChatRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    message: str = Field(max_length=MAX_MESSAGE_LENGTH)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value


class ToolCall(BaseModel):
    name: str
    args: dict
    result: dict


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[ToolCall] = []   # lets the UI show which tools ran


class PartOut(BaseModel):
    id: int
    name: str
    quantity: int
    category: str
    location: str


# ---------- app ----------
app = FastAPI(title="CURT Inventory Assistant API", version="1.0.0", lifespan=lifespan)
service = InventoryService()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # A plain `def` endpoint runs in FastAPI's thread pool, so the slow Gemini call
    # never blocks other requests.
    try:
        result = llm.chat(request.session_id, request.message)
    except Exception:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error.")
    return ChatResponse(**result)


@app.get("/inventory", response_model=list[PartOut])
def inventory() -> list[dict]:
    return service.get_all_parts()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}