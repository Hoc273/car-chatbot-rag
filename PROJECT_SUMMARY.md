# Tổng hợp dự án Car Chatbot RAG

## Mục tiêu dự án

Dự án xây dựng chatbot tư vấn xe Toyota theo mô hình RAG
(Retrieval-Augmented Generation). Hệ thống đọc tài liệu PDF, chia nhỏ nội dung
thành chunks, tạo embedding, lưu vào Qdrant và dùng truy vấn của người dùng để
tìm ngữ cảnh phù hợp trước khi sinh câu trả lời.

## Luồng hoạt động chính

1. Người dùng đặt câu hỏi trong `rag.py`.
2. Hệ thống phân loại intent bằng `intent_classifier.py`.
3. Business rules trong `business_rules.py` kiểm tra các trường hợp cần chặn
   hoặc cảnh báo.
4. `slot_extractor.py` trích xuất thông tin nhu cầu khách hàng.
5. `conversation_state_manager.py` lưu trạng thái hội thoại.
6. `logic_smart_car_consultant.py` quyết định có cần dùng RAG hay không.
7. Nếu cần RAG:
   - `embed.py` tạo embedding cho query.
   - `vector_database.py` search Qdrant.
   - `rag.py` build context từ các chunk tìm được.
8. LLM sinh câu trả lời dựa trên prompt hệ thống, lịch sử hội thoại và context.

## Luồng ingest tài liệu

Luồng ingest nằm chính trong `vector_database.py`:

```bash
python vector_database.py --rebuild
```

Chế độ rebuild sạch sẽ:

1. Recreate collection Qdrant.
2. Đọc toàn bộ PDF trong thư mục `documents/`.
3. Chunk nội dung bằng `chunking.py`.
4. Embed chunks bằng `embed.py`.
5. Upsert chunks vào Qdrant.
6. Tạo summary chunk danh sách xe.
7. Cập nhật `.processed_pdfs.json` sau khi upsert thành công.

Chế độ incremental:

```bash
python vector_database.py
```

Chế độ này chỉ đọc PDF mới hoặc PDF đã thay đổi theo cache
`.processed_pdfs.json`. Nếu collection Qdrant rỗng, hệ thống sẽ bỏ qua cache và
đọc lại toàn bộ PDF để tránh index rỗng.

## Những phần đã hoàn thiện

### 1. Xử lý PDF

File: `data_processing/extract_pdf.py`

- Đọc PDF bằng PyMuPDF.
- Lấy text theo từng trang.
- Gắn metadata cho từng page:
  - `source`
  - `source_path`
  - `source_key`
  - `source_id`
  - `source_hash`
  - `page`
  - `total_pages`
- Có hai chế độ đọc:
  - `extract_all_pdfs()` cho rebuild sạch.
  - `extract_multiple_pdfs()` cho incremental.

### 2. Registry runtime cache

File: `.processed_pdfs.json`

- Registry chỉ còn là cache runtime để skip PDF không đổi.
- Registry không còn là nguồn dữ liệu chính khi rebuild.
- Registry chỉ được cập nhật sau khi index/upsert thành công qua
  `mark_documents_processed()`.

Điều này tránh lỗi: đọc PDF thành công nhưng embed/upsert lỗi, lần sau hệ thống
lại tưởng file đã xử lý và skip nhầm.

### 3. Chunk id ổn định

File: `chunking.py`

Chunk id đã được đổi từ dạng global counter:

```text
chunk_00000
chunk_00001
```

sang dạng ổn định theo:

```text
source_id + page + chunk_index + char_start + content_hash
```

Ví dụ:

```text
documents-bantai.pdf__d2f1e29606__p0001__c0000__s000000__h56ed82d97847
```

Lợi ích:

- Thêm PDF mới không làm đổi id của chunk cũ.
- Rebuild lại vẫn tạo id ổn định nếu nội dung không đổi.
- Tránh overwrite sai trong Qdrant.
- `source_id` dùng đường dẫn tương đối kèm hash ngắn nên ổn định hơn filename.

### 4. Qdrant point id ổn định

File: `vector_database.py`

Qdrant point id được tạo deterministic từ `chunk_id`:

```python
uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)
```

Điều này giúp upsert đúng điểm cũ khi chunk không đổi, thay vì tạo point mới
hoặc overwrite nhầm.

### 5. Rebuild sạch collection

File: `vector_database.py`

Đã thêm chế độ:

```bash
python vector_database.py --rebuild
```

Chế độ này recreate collection, đọc toàn bộ PDF, chunk, embed, upsert và tạo
summary chunk.

### 6. Xử lý collection rỗng

File: `vector_database.py`

Khi chạy incremental, hệ thống kiểm tra collection có rỗng không. Nếu collection
mới tạo hoặc count bằng 0, hệ thống sẽ đọc lại toàn bộ PDF thay vì tin vào
registry cache.

Điều này tránh lỗi Qdrant rỗng nhưng `.processed_pdfs.json` vẫn khiến hệ thống
skip toàn bộ tài liệu.

### 7. Embedding

File: `embed.py`

- Dùng model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

- Hỗ trợ tiếng Việt.
- Vector dimension hiện tại: `384`.
- Embedding được normalize để phù hợp cosine similarity.

### 8. RAG pipeline

File: `rag.py`

Đã có đầy đủ các bước:

- Intent classifier.
- Business rules.
- Slot extractor.
- Conversation state.
- Smart consultant.
- RAG retrieve từ Qdrant.
- LLM generation với provider:
  - Groq
  - Ollama
  - Gemini
  - Auto fallback

## Các file chính

| File | Vai trò |
| --- | --- |
| `rag.py` | Pipeline hỏi đáp chính |
| `vector_database.py` | Tạo collection, ingest, upsert, search Qdrant |
| `chunking.py` | Làm sạch text và chia chunk |
| `embed.py` | Tạo embedding cho chunk và query |
| `data_processing/extract_pdf.py` | Đọc PDF và quản lý registry cache |
| `intent_classifier.py` | Phân loại intent |
| `business_rules.py` | Kiểm tra rule nghiệp vụ |
| `slot_extractor.py` | Trích xuất slot nhu cầu người dùng |
| `conversation_state_manager.py` | Quản lý trạng thái hội thoại |
| `logic_smart_car_consultant.py` | Quyết định prompt và skip/use RAG |

## Lệnh chạy thường dùng

Khởi động Qdrant:

```bash
docker-compose up -d
```

Rebuild toàn bộ index:

```bash
python vector_database.py --rebuild
```

Ingest incremental:

```bash
python vector_database.py
```

Chạy chatbot CLI:

```bash
python rag.py
```

Kiểm tra cú pháp các file chính:

```bash
python -m py_compile data_processing/extract_pdf.py chunking.py vector_database.py embed.py rag.py
```

## Kết quả kiểm tra gần nhất

Dry-run đọc tài liệu:

```text
5 PDF
165 trang
638 chunks
0 duplicate chunk_id
```

Ví dụ metadata/id:

```text
sample_source_id:
documents/bantai.pdf__d2f1e29606

sample_chunk_id:
documents-bantai.pdf__d2f1e29606__p0001__c0000__s000000__h56ed82d97847
```

## Những điểm có thể cải thiện tiếp

- Hiển thị `source` và `chunk_id` rõ hơn trong response sources của `rag.py`.
- Summary chunk có thể lấy danh sách xe từ danh sách PDF/docs thay vì chỉ từ
  chunks.
- Thêm test tự động cho:
  - stable chunk id
  - registry timing
  - collection rỗng
  - search payload
- Chuẩn hóa lại encoding/log tiếng Việt trong terminal Windows để tránh mojibake.
