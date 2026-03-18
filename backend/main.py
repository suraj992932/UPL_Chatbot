"""
FastAPI application for the UPL Policy Chatbot.

Exposes a /chat endpoint that answers questions using RAG
over UPL corporate policy PDFs.
"""

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.pdf_loader import load_and_split_pdfs
from backend.vector_store import get_or_build_vector_store
from backend.rag_chain import generate_answer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
vector_store = None


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load PDFs and build/load the FAISS index on startup."""
    global vector_store
    logger.info(" Starting UPL Policy Chatbot backend …")

    try:
        chunks = load_and_split_pdfs()
        vector_store = get_or_build_vector_store(chunks)
        logger.info(" Vector store ready — %d vectors loaded.", vector_store.index.ntotal)
    except Exception as exc:
        logger.error(" Failed to initialise vector store: %s", exc)
        raise

    yield  # app is running

    logger.info("Shutting down UPL Policy Chatbot backend.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="UPL Policy Chatbot API",
    description="RAG-powered Q&A over UPL corporate policy documents.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (or any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a user question using the RAG pipeline."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store is not ready yet.")

    try:
        logger.info(" Received query: %s", request.query[:100])
        result = generate_answer(request.query, vector_store)
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except Exception as exc:
        logger.error("Error processing query: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")
