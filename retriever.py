import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import minmax_scale

INDEX_DIR = "data/index"
TOP_K = 8  # Return more context for better answers

# ✅ Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    # Load FAISS index
    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    with open(os.path.join(INDEX_DIR, "texts.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(INDEX_DIR, "bm25.json"), "r", encoding="utf-8") as f:
        bm25_data = json.load(f)
    bm25 = BM25Okapi(bm25_data["tokenized"])
    return faiss_index, embeddings, chunks, bm25

def hybrid_search(query: str, top_k=TOP_K):
    faiss_index, embeddings, chunks, bm25 = load_index()

    # ✅ Encode query once
    q_vec = model.encode([query], normalize_embeddings=True)

    # ✅ FAISS search
    faiss_scores, ids = faiss_index.search(q_vec, top_k)
    faiss_results = [
        {"text": chunks[i]["text"], "meta": chunks[i]["meta"], "faiss_score": float(faiss_scores[0][idx])}
        for idx, i in enumerate(ids[0])
    ]

    # ✅ BM25 search
    tokenized_query = re.findall(r"\w+", query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [
        {"text": chunks[i]["text"], "meta": chunks[i]["meta"], "bm25_score": float(bm25_scores[i])}
        for i in bm25_indices
    ]

    # ✅ Merge with normalization
    combined = []
    for r in faiss_results:
        r["bm25_score"] = 0.0
        combined.append(r)
    for r in bm25_results:
        r["faiss_score"] = 0.0
        combined.append(r)

    # Normalize both scores to 0-1
    faiss_norm = minmax_scale([r["faiss_score"] for r in combined])
    bm25_norm = minmax_scale([r["bm25_score"] for r in combined])
    for idx, r in enumerate(combined):
        r["score"] = 0.6 * faiss_norm[idx] + 0.4 * bm25_norm[idx]  # Weighted hybrid

    # ✅ Deduplicate and sort
    seen = set()
    merged = []
    for r in sorted(combined, key=lambda x: x["score"], reverse=True):
        key = (r["meta"]["source"], r["meta"]["chunk_id"])
        if key not in seen:
            seen.add(key)
            merged.append(r)
        if len(merged) >= top_k:
            break

    return merged

if __name__ == "__main__":
    results = hybrid_search("What is blood cancer?")
    for r in results:
        print(f"[{r['meta']['source']} - chunk {r['meta']['chunk_id']}] {r['text'][:200]}...")
