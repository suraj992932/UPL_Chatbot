# UPL Policy Chatbot 📋

A RAG-powered chatbot that answers questions **exclusively** from UPL corporate policy PDFs, using FastAPI + LangChain + FAISS + Gemini + Streamlit.

## Project Structure

```
UPL_Policy_Chatbot/
├── backend/
│   ├── __init__.py
│   ├── config.py          # Settings & environment variables
│   ├── pdf_loader.py      # PDF loading & text chunking
│   ├── vector_store.py    # FAISS index build / load / search
│   ├── rag_chain.py       # RAG pipeline (retrieve → generate)
│   ├── main.py            # FastAPI app with /chat endpoint
│   └── .env               # API keys (not committed)
├── frontend/
│   └── app.py             # Streamlit chat UI
├── data_pdf/              # UPL policy PDFs (22 files)
├── faiss_index/           # Cached FAISS index (auto-generated)
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies

```bash
cd UPL_Policy_Chatbot
pip install -r requirements.txt
```

### 2. Set your Gemini API key

Edit `backend/.env`:
```
GEMINI_API_KEY=your_google_api_key_here
```

### 3. Run the backend

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

The first start will:
- Load all 22 policy PDFs
- Split text into chunks
- Build the FAISS vector index (cached for subsequent starts)

### 4. Run the frontend (in a separate terminal)

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## API Endpoints

| Method | Endpoint  | Description                  |
|--------|-----------|------------------------------|
| GET    | /health   | Health check                 |
| POST   | /chat     | Send a query, get an answer  |

### POST /chat

**Request:**
```json
{ "query": "What is UPL's whistle blower policy?" }
```

**Response:**
```json
{
  "answer": "UPL's Whistle Blower Policy allows employees to …",
  "sources": ["UPL-PROPOSED--WHISTLE-BLOWER-POLICY.pdf"]
}
```

## Tech Stack

- **FastAPI** — REST API backend
- **LangChain** — Document loading, splitting, RAG orchestration
- **FAISS** — Vector similarity search
- **HuggingFace** — Sentence embeddings (`all-MiniLM-L6-v2`)
- **Google Gemini** — LLM for answer generation
- **Streamlit** — Chat frontend
