# chunking.py

from typing import List, Dict, Any
import hashlib
import re


CONTENT_HASH_LENGTH = 12


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
        source = metadata.get("source", "unknown")
        source_id = _stable_source_id(source)
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

            if len(chunk_text) > 50:
                content_hash = _content_hash(chunk_text)
                chunk_id = _stable_chunk_id(
                    source,
                    page,
                    chunk_index,
                    start,
                    content_hash,
                )
                chunks.append({
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
                })
                chunk_index += 1
            elif chunk_text:
                print(f"[WARN] Bỏ chunk ngắn ({len(chunk_text)} chars): {chunk_text[:60]!r}")

            next_start = end - chunk_overlap
            start = next_start if next_start > start else start + 1

    print(f"[INFO] Tạo được {len(chunks)} chunks từ {len(documents)} trang")
    return chunks


# ── Chạy trực tiếp để test ───────────────────────────────────────────────────
if __name__ == "__main__":
    from data_processing.extract_pdf import extract_multiple_pdfs  # ← đổi ở đây

    # Tự động đọc toàn bộ documents/, chỉ xử lý file mới
    docs = extract_multiple_pdfs()

    if not docs:
        print("[INFO] Không có tài liệu mới để chunk.")
    else:
        chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=150)

        print("\n--- 3 chunks đầu tiên ---")
        for c in chunks[:3]:
            print(f"🧩 {c['chunk_id']} | {c['metadata']['source']} "
                  f"| Page {c['metadata']['page']} | {len(c['content'])} chars")
            print(f"   {c['content'][:120]}...")
            print("-" * 80)
