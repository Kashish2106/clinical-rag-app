import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "data/index"
CHUNKS_FILE = "data/chunks/chunks.json"

# ✅ Load embedding model once
model = SentenceTransformer(EMBEDDING_MODEL)

def build_index(overwrite=True):
    """
    Build a new FAISS + BM25 index from scratch OR overwrite existing.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    print("📥 Generating embeddings...")
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(INDEX_DIR, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    # ✅ BM25 index
    tokenized_texts = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25_data = {"docs": chunks, "tokenized": tokenized_texts}
    with open(os.path.join(INDEX_DIR, "bm25.json"), "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False)

    print("✅ Full index built successfully!")

def update_index(new_chunks):
    """
    Incrementally update FAISS + BM25 with new chunks without rebuilding everything.
    """
    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    existing_embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    with open(os.path.join(INDEX_DIR, "texts.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(INDEX_DIR, "bm25.json"), "r", encoding="utf-8") as f:
        bm25_data = json.load(f)

    # ✅ Generate new embeddings
    new_texts = [c["text"] for c in new_chunks]
    new_embeddings = model.encode(new_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)

    # ✅ Update FAISS
    faiss_index.add(new_embeddings)
    faiss.write_index(faiss_index, os.path.join(INDEX_DIR, "faiss.index"))

    # ✅ Update numpy embeddings
    updated_embeddings = np.vstack([existing_embeddings, new_embeddings])
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), updated_embeddings)

    # ✅ Update metadata
    chunks.extend(new_chunks)
    with open(os.path.join(INDEX_DIR, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    # ✅ Update BM25
    tokenized_new = [re.findall(r"\w+", t.lower()) for t in new_texts]
    bm25_data["docs"].extend(new_chunks)
    bm25_data["tokenized"].extend(tokenized_new)
    with open(os.path.join(INDEX_DIR, "bm25.json"), "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False)

    print(f"✅ Index updated with {len(new_chunks)} new chunks!")

if __name__ == "__main__":
    build_index(overwrite=True)
