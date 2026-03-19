"""
Configuration settings for the UPL Policy Chatbot backend.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PDF_DIR = PROJECT_ROOT / "knowledge_base"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"

# ── Text Splitting ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = 5

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_MODEL_NAME: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.5
