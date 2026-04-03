import fitz  # PyMuPDF
from pathlib import Path

def extract_pdf_with_metadata(pdf_path: str):
    doc = fitz.open(pdf_path)
    documents = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        content = page.get_text("text").strip()   # "text" là mặc định, rõ ràng hơn
        
        # Xử lý trang chỉ toàn hình ảnh (như trang bìa)
        if not content:
            content = f"[Trang {page_num+1} - chủ yếu là hình ảnh, không có text selectable]"
        
        documents.append({
            "content": content,
            "metadata": {
                "source": str(Path(pdf_path).name),
                "page": page_num + 1,
                "total_pages": len(doc)
            }
        })
    
    doc.close()
    return documents
