import faiss
from app.embeddings import get_model

DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.35


def retrieve(question, index, chunks, top_k=DEFAULT_TOP_K, threshold=SIMILARITY_THRESHOLD):
    model = get_model()
    query_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < threshold:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "source": chunk["source"],
            "score": float(score)
        })

    return results