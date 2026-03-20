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

SYSTEM_PROMPT = """\
You are the **UPL Policy Chatbot**, an AI assistant dedicated to answering questions about UPL corporate policies.

RULES:
1. If the user greets you (e.g., "Hi", "Hello", "Hii"), respond politely and introduce yourself.
2. If the user asks "Who are you?" or "Tell me about yourself", explain your purpose as the UPL Policy Chatbot.
3. For policy-related questions, answer ONLY from the provided context. If the answer is not in the context, say: "Answer not found in provided policy documents."
4. Keep answers concise by default, but provide detailed explanations if the user explicitly asks for them.
5. For policy-related answers, always mention the source policy name.
6. If an answer is partially available, try to infer from context.

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
    
    return "\n\n---\n\n".join(parts)


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


def generate_answer(
    query: str,
    vector_store: FAISS,
    k: int = TOP_K_RESULTS,
) -> Dict[str, object]:
    """
    Full RAG pipeline with shortened context and rate-limit handling.
    """
    # 1. Retrieve
    # relevant_docs = similarity_search(vector_store, query, k=k)
    # logger.info("Retrieved %d chunks for query: %s", len(relevant_docs), query[:80])

    # if not relevant_docs:
    #     return {
    #         "answer": "Answer not found in provided policy documents.",
    #         "sources": [],
    #     }
    # 1. Retrieve
    relevant_docs = similarity_search(vector_store, query, k=k)

    print("\n Retrieved Docs:\n")

    if not relevant_docs:
        print(" No documents retrieved!")
    else:
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\n--- Document {i} ---")
            print(doc.page_content[:300])

    logger.info("Retrieved %d chunks for query: %s", len(relevant_docs), query[:80])

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
    all_retrieved_sources = _extract_sources(relevant_docs)
    
    # Optional: Suppress sources if the message is a greeting or identity explanation,
    # or if the model says it couldn't find an answer.
    lowered_answer = response.content.lower()
    if "answer not found" in lowered_answer:
        sources = []
    else:
        # A simple but effective heuristic: Only return sources that the LLM 
        # actually mentioned in its answer (since rule #5 requires it).
        sources = [s for s in all_retrieved_sources if s.lower().replace(".pdf", "").replace("-", " ").replace("_", " ") in lowered_answer or s.lower() in lowered_answer]
        
        # Fallback if no specific source was mentioned by name but the model clearly didn't say "Answer not found". 
        # This handles cases where it might use a generic reference but we still want to show all retrieved docs as sources.
        if not sources and len(response.content.split()) > 40: # Probably a policy answer
             sources = all_retrieved_sources

    result = {
        "answer": response.content,
        "sources": sources,
    }
    
    return result
