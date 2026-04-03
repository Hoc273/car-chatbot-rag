from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# ── Cấu hình ──────────────────────────────────────────────────────────────────
# Model đa ngữ, hỗ trợ tiếng Việt tốt, nhẹ (~270MB)
# Alternatives:
#   "keepitreal/vietnamese-sbert"          → tiếng Việt chuyên biệt
#   "intfloat/multilingual-e5-large"       → chất lượng cao hơn, nặng hơn (~560MB)
#   "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  → cân bằng
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384  # thay thành 768 nếu dùng model large
BATCH_SIZE = 64  # local model → batch lớn được, tùy RAM/VRAM

_model: SentenceTransformer = None  # lazy load


def get_model() -> SentenceTransformer:
    """Lazy load model một lần duy nhất."""
    global _model
    if _model is None:
        print(f"🔄 Loading model '{EMBED_MODEL}'...")
        _model = SentenceTransformer(EMBED_MODEL)
        print(f"✅ Model loaded (dim={_model.get_sentence_embedding_dimension()})")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed danh sách texts, trả về list of vectors."""
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity = dot product → tối ưu cho Qdrant
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Embed câu hỏi của user."""
    model = get_model()
    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Thêm vector embedding vào mỗi chunk."""
    texts = [c["content"] for c in chunks]
    print(f"🔢 Đang embed {len(texts)} chunks...")

    vectors = embed_texts(texts)

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec

    print(f"✅ Đã embed xong {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    from data_processing.extract_pdf import extract_pdf_with_metadata
    from chunking import chunk_documents

    pdf_path = "documents\ATBM HTTT - Hoàng Xuân Dậu.pdf"
    docs = extract_pdf_with_metadata(pdf_path)
    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks)

    print(f"\nSample vector (first 5 dims): {chunks[0]['embedding'][:5]}")
    print(f"Vector dim: {len(chunks[0]['embedding'])}")
