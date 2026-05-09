"""
main.py - FastAPI application for the SHL Assessment Recommender.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → {"reply": str, "recommendations": [...], "end_of_conversation": bool}
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from agent import run_agent
from catalog import get_retriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas (non-negotiable per assignment spec)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list) -> list:
        if not v:
            raise ValueError("messages list must not be empty")
        if len(v) > 20:
            raise ValueError("messages list exceeds maximum length of 20")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# Lifespan: warm up the catalog index once on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up catalog retriever …")
    start = time.perf_counter()
    retriever = get_retriever()
    elapsed = time.perf_counter() - start
    logger.info(
        "Catalog ready: %d assessments indexed in %.2fs.",
        retriever.item_count(),
        elapsed,
    )
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL Individual Test Solutions "
        "through multi-turn dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Readiness check")
async def health() -> dict:
    """Returns 200 OK when the service is ready."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, summary="Chat with the SHL recommender")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless chat endpoint. Pass the full conversation history on every call.
    Returns the next agent reply and, when appropriate, a structured shortlist.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = run_agent(messages)
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal agent error. Please try again.")

    recs = [
        Recommendation(
            name=r["name"],
            url=r["url"],
            test_type=r["test_type"],
        )
        for r in result.get("recommendations", [])
    ]

    return ChatResponse(
        reply=result["reply"],
        recommendations=recs,
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ---------------------------------------------------------------------------
# Global exception handler — always return JSON (never HTML error pages)
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."},
    )


# ---------------------------------------------------------------------------
# Entry point for local dev: python main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
