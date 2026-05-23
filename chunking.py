# chunking.py

from typing import List, Dict, Any
import hashlib
import re


CONTENT_HASH_LENGTH = 12
MIN_CHUNK_LENGTH = 50
TECHNICAL_SHORT_PATTERN = re.compile(
    r"(\d|:|/100km|lít|lit|mm|cm|m\b|kw|hp|nm|cc|sao|túi khí|tui khi|hybrid|4x4|awd)",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Làm sạch text: bỏ ký tự thừa, chuẩn hóa khoảng trắng."""
    text = re.sub(r"\f", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:CONTENT_HASH_LENGTH]


def _stable_source_id(source: Any) -> str:
    source_id = str(source or "unknown").strip().lower()
    source_id = re.sub(r"\s+", "-", source_id)
    source_id = re.sub(r"[^a-z0-9._-]+", "-", source_id)
    source_id = re.sub(r"-{2,}", "-", source_id).strip("-")
    return source_id or "unknown"


def _stable_chunk_id(
    source: Any,
    page: Any,
    chunk_index: int,
    char_start: int,
    content_hash: str,
) -> str:
    try:
        page_part = f"{int(page):04d}"
    except (TypeError, ValueError):
        page_part = _stable_source_id(page)

    return (
        f"{_stable_source_id(source)}"
        f"__p{page_part}"
        f"__c{chunk_index:04d}"
        f"__s{char_start:06d}"
        f"__h{content_hash}"
    )


def _has_indexable_short_content(text: str) -> bool:
    return TECHNICAL_SHORT_PATTERN.search(text) is not None


def _make_chunk(
    chunk_text: str,
    metadata: Dict[str, Any],
    source_id: str,
    page: Any,
    chunk_index: int,
    start: int,
    end: int,
) -> Dict[str, Any]:
    content_hash = _content_hash(chunk_text)
    chunk_id = _stable_chunk_id(
        source_id,
        page,
        chunk_index,
        start,
        content_hash,
    )
    return {
        "chunk_id": chunk_id,
        "content": chunk_text,
        "metadata": {
            **metadata,
            "chunk_id": chunk_id,
            "source_id": source_id,
            "chunk_index": chunk_index,
            "char_start": start,
            "char_end": end,
            "content_hash": content_hash,
        },
    }


def _same_source_page(chunk: Dict[str, Any], source_id: str, page: Any) -> bool:
    metadata = chunk.get("metadata", {})
    return metadata.get("source_id") == source_id and metadata.get("page") == page


def _merge_short_chunk(
    chunk: Dict[str, Any],
    chunk_text: str,
    source_id: str,
    page: Any,
    end: int,
) -> None:
    metadata = chunk["metadata"]
    merged_text = f"{chunk['content']}\n{chunk_text}".strip()
    content_hash = _content_hash(merged_text)
    chunk_id = _stable_chunk_id(
        source_id,
        page,
        metadata["chunk_index"],
        metadata["char_start"],
        content_hash,
    )
    chunk["chunk_id"] = chunk_id
    chunk["content"] = merged_text
    metadata["chunk_id"] = chunk_id
    metadata["char_end"] = end
    metadata["content_hash"] = content_hash


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError(
            f"chunk_overlap phải trong khoảng [0, chunk_size): "
            f"{chunk_overlap} vs {chunk_size}"
        )

    chunks = []

    for doc in documents:
        text = clean_text(doc["content"])
        if not text or re.match(r"^\[Trang", text, re.IGNORECASE):
            continue

        metadata = doc.get("metadata", {})
        source_id = metadata.get("source_id") or _stable_source_id(
            metadata.get("source_path") or metadata.get("source", "unknown")
        )
        page = metadata.get("page", 0)
        chunk_index = 0

        start = 0
        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                para_break = text.rfind("\n\n", start, end)
                if para_break != -1 and para_break > start + chunk_overlap:
                    end = para_break
                else:
                    sentence_break = max(
                        text.rfind(". ", start, end),
                        text.rfind(".\n", start, end),
                    )
                    if sentence_break != -1 and sentence_break > start + chunk_overlap:
                        end = sentence_break + 1

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= MIN_CHUNK_LENGTH or _has_indexable_short_content(chunk_text):
                chunks.append(
                    _make_chunk(
                        chunk_text,
                        metadata,
                        source_id,
                        page,
                        chunk_index,
                        start,
                        end,
                    )
                )
                chunk_index += 1
            elif chunk_text:
                source = metadata.get("source", "unknown")
                if chunks and _same_source_page(chunks[-1], source_id, page):
                    _merge_short_chunk(chunks[-1], chunk_text, source_id, page, end)
                    print(
                        f"[WARN] Gộp chunk ngắn ({len(chunk_text)} chars) "
                        f"vào chunk trước: {source} trang {page}: {chunk_text[:60]!r}"
                    )
                else:
                    chunks.append(
                        _make_chunk(
                            chunk_text,
                            metadata,
                            source_id,
                            page,
                            chunk_index,
                            start,
                            end,
                        )
                    )
                    chunk_index += 1
                    print(
                        f"[WARN] Giữ chunk ngắn ({len(chunk_text)} chars): "
                        f"{source} trang {page}: {chunk_text[:60]!r}"
                    )

            next_start = end - chunk_overlap
            start = next_start if next_start > start else start + 1

    print(f"[INFO] Tạo được {len(chunks)} chunks từ {len(documents)} trang")
    return chunks
