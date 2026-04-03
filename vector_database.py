import os
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest,
)

# ── Cấu hình ──────────────────────────────────────────────────────────────────
QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)   # None nếu self-hosted local
COLLECTION    = "atbm_httt"
EMBED_DIM     = 384
BATCH_SIZE    = 100


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_collection(recreate: bool = False):
    """Tạo collection Qdrant (idempotent)."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION in existing:
        if recreate:
            client.delete_collection(COLLECTION)
            print(f"🗑️  Đã xoá collection cũ: {COLLECTION}")
        else:
            print(f"ℹ️  Collection '{COLLECTION}' đã tồn tại, bỏ qua tạo mới.")
            return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"✅ Đã tạo collection '{COLLECTION}' (dim={EMBED_DIM}, cosine)")


def upsert_chunks(chunks: List[Dict[str, Any]]):
    """Upsert toàn bộ chunks vào Qdrant theo batch."""
    client = get_client()
    total = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])),
                vector=c["embedding"],
                payload={
                    "chunk_id":    c["chunk_id"],
                    "content":     c["content"],
                    "source":      c["metadata"]["source"],
                    "page":        c["metadata"]["page"],
                    "total_pages": c["metadata"]["total_pages"],
                    "chunk_index": c["metadata"]["chunk_index"],
                },
            )
            for c in batch
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
        print(f"   ⬆️  Upserted {total}/{len(chunks)} points")

    print(f"✅ Hoàn tất upsert {total} điểm vào '{COLLECTION}'")


def search(
    query_vector: List[float],
    top_k: int = 5,
    score_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """Vector search, trả về list kết quả có payload."""
    client = get_client()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return [
        {
            "score":   r.score,
            "content": r.payload["content"],
            "page":    r.payload["page"],
            "source":  r.payload["source"],
        }
        for r in results
    ]


def get_collection_info() -> Dict:
    client = get_client()
    info = client.get_collection(COLLECTION)
    return {
        "vectors_count": info.vectors_count,
        "status":        info.status,
    }


if __name__ == "__main__":
    from data_processing.extract_pdf import extract_pdf_with_metadata
    from chunking import chunk_documents
    from embed import embed_chunks

    pdf_path = "documents\ATBM HTTT - Hoàng Xuân Dậu.pdf"
    docs   = extract_pdf_with_metadata(pdf_path)
    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks)

    create_collection(recreate=True)
    upsert_chunks(chunks)
    print(get_collection_info())