"""
FAISS vector store management.

Builds, persists, and loads a FAISS index from document chunks
using HuggingFace sentence-transformer embeddings.
"""

import logging
from pathlib import Path
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.config import EMBEDDING_MODEL_NAME, FAISS_INDEX_DIR, TOP_K_RESULTS

logger = logging.getLogger(__name__)

# Singleton for the embedding model (loaded once)
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_vector_store(chunks: List[Document]) -> FAISS:
    """Create a FAISS index from document chunks and persist to disk."""
    embeddings = get_embeddings()
    logger.info("Building FAISS index from %d chunks …", len(chunks))
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist index to disk for caching
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_INDEX_DIR))
    logger.info("FAISS index saved to %s", FAISS_INDEX_DIR)
    return vector_store


def load_vector_store() -> FAISS:
    """Load a previously-persisted FAISS index from disk."""
    embeddings = get_embeddings()
    logger.info("Loading FAISS index from %s", FAISS_INDEX_DIR)
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_or_build_vector_store(chunks: List[Document] | None = None) -> FAISS:
    """Load from disk if available, otherwise build from chunks."""
    index_file = FAISS_INDEX_DIR / "index.faiss"
    if index_file.exists():
        logger.info("Found cached FAISS index — loading from disk.")
        return load_vector_store()

    if chunks is None:
        raise ValueError(
            "No cached FAISS index found and no chunks provided to build one."
        )
    return build_vector_store(chunks)


def similarity_search(vector_store: FAISS, query: str, k: int = TOP_K_RESULTS) -> List[Document]:
    """Return the top-k most similar chunks for *query*."""
    return vector_store.similarity_search(query, k=k)
