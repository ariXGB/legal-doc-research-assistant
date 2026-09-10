import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "vectorstore/index.faiss"
METADATA_PATH = "vectorstore/metadata.json"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks):
    model = get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def build_index(chunks):
    """Build a brand-new index from scratch, discarding anything previously indexed."""
    embeddings = embed_chunks(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)  
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    _save_index(index, chunks)
    return index, chunks


def add_documents(new_chunks, existing_index=None, existing_chunks=None):
    """
    Embed new_chunks and append them to an existing index/metadata set rather
    than overwriting it, so processing a new batch of PDFs doesn't erase
    documents that were indexed earlier in the session.

    If no existing index/chunks are provided, this behaves like build_index.
    """
    if existing_chunks is None:
        existing_chunks = []

    embeddings = embed_chunks(new_chunks)
    faiss.normalize_L2(embeddings)

    if existing_index is None:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
    else:
        index = existing_index

    index.add(embeddings)

    all_chunks = list(existing_chunks)
    next_id = (all_chunks[-1]["chunk_id"] + 1) if all_chunks else 0
    for chunk in new_chunks:
        chunk["chunk_id"] = next_id
        next_id += 1
    all_chunks.extend(new_chunks)

    _save_index(index, all_chunks)
    return index, all_chunks


def _save_index(index, chunks):
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(chunks, f)


def load_index():
    if not os.path.exists(INDEX_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH) as f:
        chunks = json.load(f)
    return index, chunks