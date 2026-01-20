from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "indexes"

FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"


def load_artifacts() -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index: {FAISS_PATH}")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks.json: {CHUNKS_PATH}")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest.json: {MANIFEST_PATH}")

    index = faiss.read_index(str(FAISS_PATH))

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError("chunks.json is empty or not a list")

    return index, chunks, manifest


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode([text], normalize_embeddings=True)
    return vec.astype("float32")


def retrieve(
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    query: str,
    k: int = 5,
    min_score: float = 0.35,
) -> List[Dict[str, Any]]:
    q = embed_query(model, query)
    scores, ids = index.search(q, k)

    hits: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0:
            continue
        score_f = float(score)
        if score_f < min_score:
            continue

        c = chunks[int(idx)]
        hits.append(
            {
                "rank": rank,
                "score": score_f,
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "text": c.get("text", ""),
            }
        )
    return hits


def main() -> None:
    index, chunks, manifest = load_artifacts()
    embed_model_name = manifest.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    print("Loaded:")
    print(f"  index_type: {type(index)}")
    print(f"  chunks: {len(chunks)}")
    print(f"  embed_model: {embed_model_name}")
    print()

    model = SentenceTransformer(embed_model_name)

    while True:
        q = input("Ask> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break

        hits = retrieve(index, chunks, model, q, k=5, min_score=0.35)
        if not hits:
            print("No relevant context found (below threshold). Try rephrasing.\n")
            continue

        for h in hits:
            preview = h["text"].replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            print(f"{h['rank']}. score={h['score']:.3f} doc={h['doc_id']} chunk={h['chunk_id']} :: {preview}")
        print()


if __name__ == "__main__":
    main()
