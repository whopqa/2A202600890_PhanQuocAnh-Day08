"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import json
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    import sys

    # Load data store and embeddings
    DATA_DIR = Path(__file__).parent.parent / "data"
    try:
        with open(DATA_DIR / "vector_store.json", "r", encoding="utf-8") as f:
            data_store = json.load(f)
        embeddings = np.load(DATA_DIR / "vector_store_embeddings.npy")
    except FileNotFoundError:
        return []

    # Model parameters must match Task 4
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]

    # Calculate cosine similarity
    # We can use dot product if embeddings are normalized, otherwise cosine sim formula:
    query_norm = np.linalg.norm(query_embedding)
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    
    similarities = np.dot(embeddings, query_embedding) / (embeddings_norm * query_norm)
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "content": data_store[idx]["content"],
            "score": float(similarities[idx]),
            "metadata": data_store[idx]["metadata"]
        })
        
    return results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
