from typing import List, Dict, Any
import re


def clean_text(text: str) -> str:
    """Làm sạch text: bỏ ký tự thừa, chuẩn hóa khoảng trắng."""
    text = re.sub(r"\f", "\n", text)        # 1. form feed trước
    text = re.sub(r"\n{3,}", "\n\n", text)  # 2. nhiều dòng trống → 2 dòng
    text = re.sub(r"[ \t]+", " ", text)     # 3. normalize space sau cùng
    return text.strip()


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
    chunk_id = 0

    for doc in documents:
        text = clean_text(doc["content"])
        if not text or re.match(r"^\[Trang", text, re.IGNORECASE):
            continue

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
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id:05d}",
                        "content": chunk_text,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": chunk_id,
                            "char_start": start,
                            "char_end": end,
                        },
                    }
                )
                chunk_id += 1
            elif chunk_text:
                print(f"[WARN] Bỏ chunk ngắn ({len(chunk_text)} chars): {chunk_text[:60]!r}")

            next_start = end - chunk_overlap
            start = next_start if next_start > start else start + 1

    print(f"Tạo được {len(chunks)} chunks từ {len(documents)} trang")
    return chunks


if __name__ == "__main__":
    from data_processing.extract_pdf import extract_pdf_with_metadata

    pdf_path = r"documents\TOYOTA.pdf"  # raw string — tránh escape issue
    docs = extract_pdf_with_metadata(pdf_path)
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=150)

    for c in chunks[:3]:
        print(
            f"🧩 {c['chunk_id']} | Page {c['metadata']['page']} | {len(c['content'])} chars"
        )
        print(f"   {c['content'][:120]}...")
        print("-" * 80)