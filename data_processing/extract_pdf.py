# data_processing/extract_pdf.py

import hashlib
import json
from pathlib import Path
import re
from typing import Union
import unicodedata

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
PROCESSED_REGISTRY = PROJECT_ROOT / ".processed_pdfs.json"


def _get_file_hash(path: Path) -> str:
    """Return an MD5 hash used only to skip unchanged files in incremental runs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_registry() -> dict:
    """Load the runtime cache of processed PDFs."""
    if not PROCESSED_REGISTRY.exists():
        return {}

    try:
        return json.loads(PROCESSED_REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[WARN] Registry loi JSON, bo qua cache: {PROCESSED_REGISTRY}")
        return {}


def _save_registry(registry: dict) -> None:
    """Save the runtime cache of processed PDFs."""
    PROCESSED_REGISTRY.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _source_key(pdf_file: Path) -> str:
    """Return a location-stable key for a PDF, preferring project-relative path."""
    resolved = pdf_file.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _slugify_source_key(source_key: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", source_key)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = ascii_text.strip().lower()
    slug = re.sub(r"[\\/]+", "/", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9._/-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-/")
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:10]
    return f"{slug or 'pdf'}__{digest}"


def _source_id(pdf_file: Path) -> str:
    return _slugify_source_key(_source_key(pdf_file))


def _resolve_pdf_paths(
    pdf_paths: Union[list[str], str] = None,
    use_documents_folder: bool = True,
    glob_pattern: str = None,
) -> list[Path]:
    """Resolve PDF files from explicit paths or the default documents folder."""
    if pdf_paths is None and use_documents_folder:
        pdf_paths = str(DOCUMENTS_DIR)

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_files: list[Path] = []
    for path_str in pdf_paths or []:
        path = Path(path_str)
        if path.is_dir():
            all_files.extend(sorted(path.glob(glob_pattern or "**/*.pdf")))
        elif path.is_file() and path.suffix.lower() == ".pdf":
            all_files.append(path)
        else:
            print(f"[WARN] Bo qua: {path}")

    resolved_files = []
    seen = set()
    for pdf_file in all_files:
        resolved = pdf_file.resolve()
        if resolved not in seen:
            seen.add(resolved)
            resolved_files.append(pdf_file)

    if not resolved_files:
        print("[WARN] Khong tim thay file PDF nao.")

    return resolved_files


def _update_registry_entry(registry: dict, pdf_file: Path, file_hash: str) -> None:
    registry[str(pdf_file.resolve())] = {
        "filename": pdf_file.name,
        "hash": file_hash,
    }


def mark_documents_processed(documents: list[dict]) -> None:
    """Commit processed PDFs to the runtime cache after indexing succeeds."""
    registry = _load_registry()
    processed_files = {}

    for doc in documents:
        metadata = doc.get("metadata", {})
        source_path = metadata.get("source_path")
        source_hash = metadata.get("source_hash")
        if source_path:
            processed_files[source_path] = source_hash

    for source_path, source_hash in processed_files.items():
        pdf_file = Path(source_path)
        file_hash = source_hash or _get_file_hash(pdf_file)
        _update_registry_entry(registry, pdf_file, file_hash)

    _save_registry(registry)
    print(f"[INFO] Da cap nhat registry cho {len(processed_files)} PDF.")


def extract_pdf_with_metadata(pdf_path: str, file_hash: str = None) -> list[dict]:
    pdf_file = Path(pdf_path)
    source_key = _source_key(pdf_file)
    source_id = _source_id(pdf_file)
    file_hash = file_hash or _get_file_hash(pdf_file)
    doc = fitz.open(str(pdf_file))
    documents = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            content = page.get_text("text").strip()

            if not content:
                content = (
                    f"[Trang {page_num + 1} - chu yeu la hinh anh, "
                    "khong co text selectable]"
                )

            documents.append({
                "content": content,
                "metadata": {
                    "source": pdf_file.name,
                    "source_path": str(pdf_file.resolve()),
                    "source_key": source_key,
                    "source_id": source_id,
                    "source_hash": file_hash,
                    "page": page_num + 1,
                    "total_pages": len(doc),
                },
            })
    finally:
        doc.close()

    return documents


def extract_all_pdfs(
    pdf_paths: Union[list[str], str] = None,
    use_documents_folder: bool = True,
    glob_pattern: str = None,
    update_registry: bool = False,
) -> list[dict]:
    """
    Read every PDF from disk, ignoring .processed_pdfs.json.

    This is the rebuild path: the registry is a runtime cache only, not the
    source of truth for what belongs in the vector index.
    """
    pdf_files = _resolve_pdf_paths(pdf_paths, use_documents_folder, glob_pattern)
    if not pdf_files:
        return []

    all_documents = []

    for pdf_file in pdf_files:
        print(f"[INFO] Dang doc PDF: {pdf_file.name}")
        try:
            docs = extract_pdf_with_metadata(str(pdf_file))
            all_documents.extend(docs)

            print(f"       -> {len(docs)} trang")
        except Exception as e:
            print(f"[ERROR] Loi khi doc {pdf_file.name}: {e}")

    if update_registry:
        print("[WARN] update_registry bi bo qua khi doc PDF; registry chi cap nhat sau khi index thanh cong.")

    print(
        f"\n[INFO] Hoan thanh doc toan bo: "
        f"{len(pdf_files)} file | {len(all_documents)} trang."
    )
    return all_documents


def extract_multiple_pdfs(
    pdf_paths: Union[list[str], str] = None,
    use_documents_folder: bool = True,
    glob_pattern: str = None,
    force_reprocess: bool = False,
    skip_processed: bool = True,
) -> list[dict]:
    """
    Incremental extractor.

    .processed_pdfs.json is only a runtime cache for skipping unchanged PDFs.
    Use extract_all_pdfs() for clean rebuilds so the index is based on the PDF
    files currently present on disk.
    """
    if force_reprocess:
        return extract_all_pdfs(
            pdf_paths=pdf_paths,
            use_documents_folder=use_documents_folder,
            glob_pattern=glob_pattern,
            update_registry=True,
        )

    pdf_files = _resolve_pdf_paths(pdf_paths, use_documents_folder, glob_pattern)
    if not pdf_files:
        return []

    registry = _load_registry()
    new_files: list[tuple[Path, str, str]] = []
    skipped_files = []

    for pdf_file in pdf_files:
        file_hash = _get_file_hash(pdf_file)
        key = str(pdf_file.resolve())
        already_processed = key in registry and registry[key].get("hash") == file_hash

        if skip_processed and already_processed:
            skipped_files.append(pdf_file.name)
        else:
            new_files.append((pdf_file, file_hash, key))

    if skipped_files:
        print(
            f"[INFO] Bo qua {len(skipped_files)} file da xu ly: "
            f"{', '.join(skipped_files)}"
        )

    if not new_files:
        print("[INFO] Khong co file moi. Khong can chunk lai.")
        return []

    all_documents = []

    for pdf_file, file_hash, key in new_files:
        label = "(moi)" if key not in registry else "(da thay doi noi dung)"
        print(f"[INFO] Dang xu ly {label}: {pdf_file.name}")
        try:
            docs = extract_pdf_with_metadata(str(pdf_file), file_hash=file_hash)
            all_documents.extend(docs)
            print(f"       -> {len(docs)} trang")
        except Exception as e:
            print(f"[ERROR] Loi khi doc {pdf_file.name}: {e}")

    print(
        f"\n[INFO] Hoan thanh incremental: "
        f"{len(new_files)} file | {len(all_documents)} trang."
    )
    return all_documents
