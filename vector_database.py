import argparse
import os
import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION = "atbm_httt"
EMBED_DIM = 384
BATCH_SIZE = 100


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def qdrant_point_id(chunk_id: str) -> str:
    """Return a deterministic UUID point id from the stable chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def create_collection(recreate: bool = False) -> None:
    """Create the Qdrant collection, optionally deleting the old one first."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION in existing:
        if recreate:
            client.delete_collection(COLLECTION)
            print(f"[INFO] Da xoa collection cu: {COLLECTION}")
        else:
            print(f"[INFO] Collection '{COLLECTION}' da ton tai, bo qua tao moi.")
            return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"[INFO] Da tao collection '{COLLECTION}' (dim={EMBED_DIM}, cosine)")


def _chunk_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata", {})
    return {
        "chunk_id": chunk["chunk_id"],
        "content": chunk["content"],
        "source": metadata.get("source"),
        "source_id": metadata.get("source_id"),
        "source_path": metadata.get("source_path"),
        "page": metadata.get("page"),
        "total_pages": metadata.get("total_pages"),
        "chunk_index": metadata.get("chunk_index"),
        "char_start": metadata.get("char_start"),
        "char_end": metadata.get("char_end"),
        "content_hash": metadata.get("content_hash"),
    }


def upsert_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Upsert chunks into Qdrant using deterministic point ids."""
    client = get_client()
    total = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        points = [
            PointStruct(
                id=qdrant_point_id(c["chunk_id"]),
                vector=c["embedding"],
                payload=_chunk_payload(c),
            )
            for c in batch
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
        print(f"   Upserted {total}/{len(chunks)} points")

    print(f"[INFO] Hoan tat upsert {total} diem vao '{COLLECTION}'")


def search(
    query_vector: List[float],
    top_k: int = 5,
    score_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    client = get_client()

    if hasattr(client, "query_points"):
        results = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points
    else:
        results = client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

    return [
        {
            "score": r.score,
            "content": r.payload["content"],
            "page": r.payload["page"],
            "source": r.payload["source"],
            "chunk_id": r.payload.get("chunk_id"),
        }
        for r in results
    ]


def get_collection_info() -> Dict[str, Any]:
    client = get_client()
    result = client.count(collection_name=COLLECTION)
    return {
        "vectors_count": result.count,
        "status": "ok",
    }


def upsert_summary_chunk(car_names: List[str]) -> None:
    """Upsert one summary chunk for listing all available cars."""
    from embed import embed_texts

    unique_names = sorted({name for name in car_names if name})
    content = "Danh sach tat ca cac xe Toyota hien co: " + ", ".join(unique_names)
    vector = embed_texts([content])[0]
    chunk_id = "summary_all_cars"
    client = get_client()

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=qdrant_point_id(chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk_id,
                    "content": content,
                    "source": "summary",
                    "source_id": "summary",
                    "source_path": None,
                    "page": 0,
                    "total_pages": 0,
                    "chunk_index": -1,
                    "char_start": 0,
                    "char_end": len(content),
                    "content_hash": None,
                },
            )
        ],
    )
    print(f"[INFO] Da upsert summary chunk: {content}")


def _car_names_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    names = []
    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source")
        if source:
            names.append(source.removesuffix(".pdf"))
    return sorted(set(names))


def ingest_documents(
    rebuild: bool = False,
    pdf_paths: List[str] | None = None,
    glob_pattern: str | None = None,
) -> Dict[str, Any]:
    """
    Ingest PDFs into Qdrant.

    rebuild=True is a clean rebuild: recreate collection, read every PDF from
    disk, chunk, embed, upsert, then create the summary chunk. The registry is
    only updated as a cache and is never used to decide rebuild input data.
    """
    from chunking import chunk_documents
    from data_processing.extract_pdf import extract_all_pdfs, extract_multiple_pdfs
    from embed import embed_chunks

    if rebuild:
        print("[INFO] Rebuild sach: recreate collection va doc toan bo PDF.")
        create_collection(recreate=True)
        docs = extract_all_pdfs(
            pdf_paths=pdf_paths,
            glob_pattern=glob_pattern,
            update_registry=True,
        )
    else:
        print("[INFO] Ingest incremental: chi doc PDF moi/da thay doi theo cache.")
        create_collection(recreate=False)
        docs = extract_multiple_pdfs(
            pdf_paths=pdf_paths,
            glob_pattern=glob_pattern,
            skip_processed=True,
        )

    if not docs:
        print("[INFO] Khong co tai lieu de ingest.")
        return get_collection_info()

    chunks = chunk_documents(docs)
    if not chunks:
        print("[INFO] Khong tao duoc chunk nao.")
        return get_collection_info()

    chunks = embed_chunks(chunks)
    upsert_chunks(chunks)
    upsert_summary_chunk(_car_names_from_chunks(chunks))

    info = get_collection_info()
    print(info)
    return info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDF documents into Qdrant.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recreate collection and ingest every PDF from disk.",
    )
    parser.add_argument(
        "--pdf-path",
        action="append",
        dest="pdf_paths",
        help="PDF file or folder to ingest. Can be passed multiple times.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help="Glob pattern when --pdf-path points to a folder, e.g. '*.pdf'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest_documents(
        rebuild=args.rebuild,
        pdf_paths=args.pdf_paths,
        glob_pattern=args.glob_pattern,
    )
