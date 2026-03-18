"""
Streamlit frontend for the UPL Policy Chatbot.

A ChatGPT-style interface that sends queries to the FastAPI backend
and displays answers with source citations.
"""

import streamlit as st
# import requests
from backend.pdf_loader import load_and_split_pdfs
from backend.vector_store import get_or_build_vector_store
from backend.rag_chain import generate_answer
@st.cache_resource
def load_vector_store():
    chunks = load_and_split_pdfs()
    return get_or_build_vector_store(chunks)

vector_store = load_vector_store()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPL Policy Chatbot",
    page_icon="📋",
    layout="centered",
)

# # ── Backend URL ───────────────────────────────────────────────────────────────
# BACKEND_URL = "http://localhost:8000"

# ── Custom CSS for a polished chat UI ─────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #312e81, #4338ca);
        color: #c7d2fe;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        margin: 0.15rem 0.2rem 0.15rem 0;
    }

    .stChatMessage {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📋 UPL Policy Chatbot</h1>
        <p>Ask questions about UPL corporate policies — powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state for chat history ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a question about UPL policies …"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents …"):
            try:
                response = generate_answer(user_input, vector_store)
                answer = response.get("answer", "No answer received.")
                sources = response.get("sources", [])
             
                # response = requests.post(
                #     f"{BACKEND_URL}/chat",
                #     json={"query": user_input},
                #     timeout=120,
                # )
                # response.raise_for_status()
                # data = response.json()

                # answer = data.get("answer", "No answer received.")
                # sources = data.get("sources", [])

                # Format sources as badges
                if sources:
                    source_html = " ".join(
                        f'<span class="source-badge">📄 {s}</span>' for s in sources
                    )
                    full_response = f"{answer}\n\n**Sources:**\n{source_html}"
                else:
                    full_response = answer

                st.markdown(full_response, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as exc:
                error_msg = f"❌ Error: {str(exc)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ About")
    st.markdown(
        "This chatbot answers questions **only** from UPL corporate policy PDFs "
        "using a Retrieval-Augmented Generation (RAG) pipeline."
    )
    st.markdown("---")
    st.markdown("### 📄 Available Policies")
    st.markdown(
        """
        - Anti-Bribery & Corruption
        - Board Diversity
        - Child Labour
        - Code of Conduct
        - CSR Policy
        - Dividend Distribution
        - ERM Policy
        - Information Security
        - Materiality Policy
        - Nomination & Remuneration
        - Related Party Transactions
        - Tax Policy
        - Whistle Blower Policy
        - … and more
        """
    )
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
