from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Gemini
import google.generativeai as genai

# -----------------------------
# Paths / loading
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "indexes"

FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

load_dotenv()  # loads .env if present

# -----------------------------
# Models
# -----------------------------
def load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}

MANIFEST = load_manifest()
EMBED_MODEL_NAME = MANIFEST.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = MANIFEST.get("llm_model", "gemini-2.5-flash")

embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Load FAISS + chunks
index = faiss.read_index(str(FAISS_PATH))
chunks: List[Dict[str, Any]] = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

# Gemini config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Keep API alive without crashing; we’ll return a clear error on /ask
    genai_configured = False
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    genai_configured = True
    llm = genai.GenerativeModel(LLM_MODEL_NAME)

# -----------------------------
# Retrieval helpers
# -----------------------------
def embed_query(text: str) -> np.ndarray:
    v = embedder.encode([text], normalize_embeddings=True)
    return v.astype("float32")

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    q = embed_query(query)
    scores, ids = index.search(q, k)
    out = []
    for rank, (idx, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
        if idx < 0:
            continue
        c = chunks[idx]
        out.append(
            {
                "rank": rank,
                "score": float(score),
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "text": c.get("text", ""),
                "citation": c.get("citation", f'{c.get("doc_id")}#{c.get("chunk_id")}'),
            }
        )
    return out

def retrieve_with_threshold(query: str, k: int = 5, min_score: float = 0.35) -> List[Dict[str, Any]]:
    hits = retrieve(query, k=k)
    hits = [h for h in hits if h["score"] >= min_score]
    return hits

SYSTEM_RULES = """You are a careful RAG assistant.
Use ONLY the provided context passages.
If the answer is not present in the context, say: "I couldn't find that in the provided documents."
Keep answers short and factual.
Cite sources inline like [doc#chunk].
"""

def build_context(passages: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    # Simple safe truncation for prompt size
    parts = []
    total = 0
    for p in passages:
        block = f'[{p["citation"]}]\n{p["text"]}\n'
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()

# -----------------------------
# API
# -----------------------------
app = FastAPI(title="RAG Document Agent", version="0.1.0")

class AskRequest(BaseModel):
    question: str
    k: int = 5
    min_score: float = 0.35

@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_model": EMBED_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "faiss_type": type(index).__name__,
        "chunks": len(chunks),
        "gemini_configured": genai_configured,
    }

@app.post("/retrieve")
def retrieve_only(req: AskRequest):
    passages = retrieve_with_threshold(req.question, k=req.k, min_score=req.min_score)
    return {"passages": passages}

@app.post("/ask")
def ask(req: AskRequest):
    passages = retrieve_with_threshold(req.question, k=req.k, min_score=req.min_score)

    # HARD GUARDRAIL: no evidence => no LLM call
    if not passages:
        return {"answer": "I couldn't find that in the provided documents.", "citations": [], "passages": []}

    if not genai_configured:
        return {"answer": "GOOGLE_API_KEY is missing (.env). Add it and restart the API.", "citations": [], "passages": passages}

    context = build_context(passages)

    prompt = f"""{SYSTEM_RULES}

Context passages:
{context}

Question:
{req.question}

Answer (with inline citations):
"""

    resp = llm.generate_content(prompt)
    answer = resp.text.strip() if resp and getattr(resp, "text", None) else ""

    return {
        "answer": answer,
        "citations": [p["citation"] for p in passages],
        "passages": passages,
    }
