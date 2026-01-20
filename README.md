# RAG Document Agent

A local Retrieval-Augmented Generation (RAG) system for answering questions strictly from provided documents.

## What this project does
- Ingests real-world documents (technical, legal, literary)
- Chunks and embeds them using Sentence Transformers
- Stores embeddings in FAISS
- Retrieves relevant passages with cosine similarity + threshold gating
- Generates answers using Gemini **only when evidence exists**

## Architecture
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector store:** FAISS (IndexFlatIP)
- **LLM:** Gemini 2.5 Flash
- **API:** FastAPI
- **UI:** Streamlit

## Safety & Guardrails
- No-answer refusal when no relevant context is found
- Minimum similarity score enforced before LLM call
- Context-only generation (no free hallucination)

## How to run locally
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

## Example use cases
- Technical specifications (RFCs, APIs)
- Legal / license documents
- Long-form public-domain texts

