import argparse
import os
import sys
import uuid
from collections.abc import Callable
from typing import Any, Dict, List, TypeVar

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION = "atbm_httt"
EMBED_DIM = 384
BATCH_SIZE = 100
T = TypeVar("T")


def _qdrant_hint() -> str:
    return (
        f"Khong ket noi duoc Qdrant tai {QDRANT_URL}.\n"
        "Hay khoi dong Qdrant truoc khi ingest/search:\n"
        "  docker compose up -d\n"
        "Sau do kiem tra dashboard: http://localhost:6333/dashboard"
    )


def _qdrant_call(action: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except ResponseHandlingException as exc:
        raise RuntimeError(f"{_qdrant_hint()}\nTac vu loi: {action}") from exc


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def qdrant_point_id(chunk_id: str) -> str:
    """Return a deterministic UUID point id from the stable chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def create_collection(recreate: bool = False) -> bool:
    """Create the Qdrant collection, optionally deleting the old one first."""
    client = get_client()
    collections = _qdrant_call("get collections", client.get_collections)
    existing = [c.name for c in collections.collections]
    collection_created = False

    if COLLECTION in existing:
        if recreate:
            _qdrant_call(
                f"delete collection '{COLLECTION}'",
                lambda: client.delete_collection(COLLECTION),
            )
            print(f"[INFO] Da xoa collection cu: {COLLECTION}")
        else:
            print(f"[INFO] Collection '{COLLECTION}' da ton tai, bo qua tao moi.")
            return False

    _qdrant_call(
        f"create collection '{COLLECTION}'",
        lambda: client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        ),
    )
    print(f"[INFO] Da tao collection '{COLLECTION}' (dim={EMBED_DIM}, cosine)")
    collection_created = True
    return collection_created


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
        _qdrant_call(
            f"upsert {len(points)} points",
            lambda: client.upsert(collection_name=COLLECTION, points=points),
        )
        total += len(points)
        print(f"   Upserted {total}/{len(chunks)} points")

    print(f"[INFO] Hoan tat upsert {total} diem vao '{COLLECTION}'")


def search(
    query_vector: List[float],
    top_k: int = 5,
    score_threshold: float = 0.4,
    source_names: List[str] | None = None,
) -> List[Dict[str, Any]]:
    client = get_client()
    query_filter = _source_filter(source_names)

    if hasattr(client, "query_points"):
        results = _qdrant_call(
            "query points",
            lambda: client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            ).points,
        )
    else:
        results = _qdrant_call(
            "search points",
            lambda: client.search(
                collection_name=COLLECTION,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            ),
        )

    return [
        _point_payload_result(r.payload, score=r.score)
        for r in results
    ]


def search_neighbor_chunks(
    *,
    source_id: str | None,
    page: int | None,
    chunk_index: int | None,
    window: int = 1,
) -> List[Dict[str, Any]]:
    if not source_id or page is None or chunk_index is None or chunk_index < 0:
        return []

    neighbor_indices = [
        idx
        for idx in range(chunk_index - window, chunk_index + window + 1)
        if idx >= 0 and idx != chunk_index
    ]
    if not neighbor_indices:
        return []

    client = get_client()
    query_filter = Filter(
        must=[
            FieldCondition(key="source_id", match=MatchValue(value=source_id)),
            FieldCondition(key="page", match=MatchValue(value=page)),
            FieldCondition(
                key="chunk_index",
                match=MatchAny(any=neighbor_indices),
            ),
        ]
    )
    points, _ = _qdrant_call(
        "scroll neighbor chunks",
        lambda: client.scroll(
            collection_name=COLLECTION,
            scroll_filter=query_filter,
            limit=len(neighbor_indices) + 2,
            with_payload=True,
            with_vectors=False,
        ),
    )
    neighbors = [_point_payload_result(p.payload, score=0.0) for p in points]
    neighbors.sort(key=lambda item: item.get("chunk_index") or 0)
    return neighbors


def scroll_chunks(
    *,
    source_names: List[str] | None = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    client = get_client()
    points, _ = _qdrant_call(
        "scroll chunks",
        lambda: client.scroll(
            collection_name=COLLECTION,
            scroll_filter=_source_filter(source_names),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ),
    )
    return [_point_payload_result(p.payload, score=0.0) for p in points]


def _source_filter(source_names: List[str] | None) -> Filter | None:
    query_filter = None
    if source_names:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchAny(any=source_names),
                )
            ]
        )
    return query_filter


def _point_payload_result(payload: Dict[str, Any], score: float = 1.0) -> Dict[str, Any]:
    return {
        "score": score,
        "content": payload["content"],
        "page": payload["page"],
        "source": payload["source"],
        "source_id": payload.get("source_id"),
        "chunk_id": payload.get("chunk_id"),
        "chunk_index": payload.get("chunk_index"),
        "char_start": payload.get("char_start"),
        "char_end": payload.get("char_end"),
    }


def get_collection_info() -> Dict[str, Any]:
    client = get_client()
    result = _qdrant_call(
        f"count collection '{COLLECTION}'",
        lambda: client.count(collection_name=COLLECTION),
    )
    return {
        "vectors_count": result.count,
        "status": "ok",
    }


def collection_point_count() -> int:
    client = get_client()
    return _qdrant_call(
        f"count collection '{COLLECTION}'",
        lambda: client.count(collection_name=COLLECTION),
    ).count


def upsert_summary_chunk(car_names: List[str]) -> None:
    """Upsert one summary chunk for listing all available cars."""
    from embed import embed_texts

    unique_names = sorted({name for name in car_names if name})
    content = "Danh sach tat ca cac xe Toyota hien co: " + ", ".join(unique_names)
    vector = embed_texts([content])[0]
    chunk_id = "summary_all_cars"
    client = get_client()

    _qdrant_call(
        "upsert summary chunk",
        lambda: client.upsert(
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
        ),
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
    from data_processing.extract_pdf import (
        extract_all_pdfs,
        extract_multiple_pdfs,
        mark_documents_processed,
    )
    from embed import embed_chunks

    if rebuild:
        print("[INFO] Rebuild sach: recreate collection va doc toan bo PDF.")
        create_collection(recreate=True)
        docs = extract_all_pdfs(
            pdf_paths=pdf_paths,
            glob_pattern=glob_pattern,
        )
    else:
        print("[INFO] Ingest incremental: chi doc PDF moi/da thay doi theo cache.")
        collection_created = create_collection(recreate=False)
        if collection_created or collection_point_count() == 0:
            print("[INFO] Collection rong, bo qua registry va doc toan bo PDF.")
            docs = extract_all_pdfs(
                pdf_paths=pdf_paths,
                glob_pattern=glob_pattern,
            )
        else:
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
    mark_documents_processed(docs)

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


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    _configure_console_encoding()
    args = _parse_args()
    try:
        ingest_documents(
            rebuild=args.rebuild,
            pdf_paths=args.pdf_paths,
            glob_pattern=args.glob_pattern,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc
