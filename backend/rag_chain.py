"""
RAG (Retrieval-Augmented Generation) pipeline.

Retrieves relevant document chunks from FAISS and generates
an answer using Google Gemini, strictly grounded in the context.
"""

import logging
import time
from typing import Dict, List
import openai

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from backend.config import OPENAI_API_KEY, OPENAI_MODEL_NAME, LLM_TEMPERATURE, TOP_K_RESULTS
from backend.vector_store import similarity_search

logger = logging.getLogger(__name__)

# ── System prompt that enforces document-only answers ─────────────────────────
SYSTEM_PROMPT = """\
You are an assistant answering questions about UPL corporate policies.
RULES:
1. Answer ONLY from the provided context. If unknown, say: "Answer not found in provided policy documents."
2. Keep answers short (maximum 2-3 lines). Be concise.
3. Mention the source policy name.

CONTEXT:
{context}
"""

HUMAN_PROMPT = "{question}"

_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


def _get_llm() -> ChatOpenAI:
    """Return a configured OpenAI LLM instance."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please add it to backend/.env")
    return ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        temperature=LLM_TEMPERATURE,
    )


def _format_context(docs: List[Document]) -> str:
    """Concatenate retrieved document contents with source tags, truncated to 500 chars."""
    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    
    full_text = "\n\n---\n\n".join(parts)
    return full_text[:500]  # Truncate context to save input tokens


def _extract_sources(docs: List[Document]) -> List[str]:
    """Return a deduplicated list of source filenames."""
    seen = set()
    sources: List[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


# Simple in-memory cache to skip API calls for repeated queries
_answer_cache: Dict[str, Dict[str, object]] = {}

def generate_answer(
    query: str,
    vector_store: FAISS,
    k: int = TOP_K_RESULTS,
) -> Dict[str, object]:
    """
    Full RAG pipeline with caching, shortened context, and rate-limit handling.
    """
    query_lower = query.strip().lower()
    
    # 0. Check cache
    if query_lower in _answer_cache:
        logger.info("Serving answer from cache.")
        return _answer_cache[query_lower]

    # 1. Retrieve
    relevant_docs = similarity_search(vector_store, query, k=k)
    logger.info("Retrieved %d chunks for query: %s", len(relevant_docs), query[:80])

    if not relevant_docs:
        return {
            "answer": "Answer not found in provided policy documents.",
            "sources": [],
        }

    # 2. Build prompt
    context = _format_context(relevant_docs)
    prompt = _prompt_template.format_messages(context=context, question=query)

    # 3. Generate (with basic retry and fallback for 429 quota errors)
    llm = _get_llm()
    response = None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 2s delay between retries
            if attempt > 0:
                time.sleep(2)
            response = llm.invoke(prompt)
            break
        except Exception as exc:
            error_msg = str(exc).lower()
            if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
                if attempt == max_retries - 1:
                    logger.error("Rate limit exhausted after retries.")
                    return {
                        "answer": "Rate limit exceeded. Please try again after some time.",
                        "sources": [],
                    }
                logger.warning("Rate limit hit, retrying in 2s... (attempt %d/%d)", attempt + 1, max_retries)
            else:
                raise exc

    # 4. Package result
    sources = _extract_sources(relevant_docs)
    
    result = {
        "answer": response.content,
        "sources": sources,
    }
    
    # Save to cache
    _answer_cache[query_lower] = result
    return result
