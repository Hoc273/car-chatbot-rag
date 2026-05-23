# Kế Hoạch Đưa RAG Vào `ai-service`

## Summary
- Đóng gói RAG Python hiện có thành một microservice riêng tên `ai-service`, chạy bằng FastAPI.
- Hệ Spring Boot gọi `ai-service` qua API Gateway/BFF bằng REST.
- `ai-service` chịu trách nhiệm: chat RAG, quản lý session bằng Redis, upload/ingest tài liệu PDF, truy vấn Qdrant, gọi Groq để sinh câu trả lời.
- Không rewrite RAG sang Java ở giai đoạn này để giảm rủi ro và tận dụng pipeline Python hiện có.

## Kiến Trúc Đề Xuất
- Thành phần:
  - `api-gateway` hoặc BFF Spring Boot: nhận request từ frontend, xác thực user, gọi `ai-service`.
  - `ai-service` Python/FastAPI: expose REST API cho chat và quản trị tài liệu.
  - `redis`: lưu conversation state theo `session_id`.
  - `qdrant`: lưu vector index.
  - `groq`: LLM provider duy nhất ở giai đoạn đầu.
  - volume `documents/`: lưu PDF upload phục vụ ingest.
- Luồng chat:
  - Frontend gửi câu hỏi tới Gateway.
  - Gateway forward sang `POST /api/v1/chat`.
  - `ai-service` lấy session state từ Redis, chạy pipeline `answer(query, session_id)`, search Qdrant, gọi Groq, trả answer + sources + slots.
  - Gateway trả response về frontend.
- Luồng ingest:
  - Admin upload PDF qua Gateway.
  - Gateway gọi `POST /api/v1/documents`.
  - `ai-service` lưu file vào `documents/`, chạy ingest async/incremental vào Qdrant.
  - Admin kiểm tra trạng thái qua `GET /api/v1/documents/jobs/{job_id}`.

## API/Public Interface
- `POST /api/v1/chat`
```json
{
  "session_id": "string",
  "user_id": "string",
  "message": "string"
}
```
Response:
```json
{
  "answer": "string",
  "sources": [
    {
      "source": "string",
      "page": 1,
      "score": 0.87
    }
  ],
  "intent": "string",
  "stage": "string",
  "slots": {},
  "session_id": "string"
}
```

- `POST /api/v1/documents`
  - `multipart/form-data`
  - fields: `file`, optional `rebuild=false`
  - response:
```json
{
  "job_id": "string",
  "status": "queued"
}
```

- `GET /api/v1/documents/jobs/{job_id}`
```json
{
  "job_id": "string",
  "status": "queued|running|success|failed",
  "message": "string",
  "indexed_pages": 0,
  "indexed_chunks": 0
}
```

- `POST /api/v1/sessions/{session_id}/reset`
```json
{
  "session_id": "string",
  "status": "reset"
}
```

- `GET /health`
```json
{
  "status": "ok",
  "qdrant": "ok",
  "redis": "ok",
  "llm": "configured"
}
```

## Implementation Changes
- Tạo FastAPI entrypoint cho RAG:
  - Wrap `rag.answer(query, session_id)` thành endpoint `/api/v1/chat`.
  - Không để FastAPI gọi CLI loop trong `rag.py`.
  - Validate request/response bằng Pydantic.
- Thay conversation state in-memory bằng Redis:
  - Serialize `ConversationState` theo `session_id`.
  - TTL mặc định: 1 giờ, giữ logic hiện có về slots/history/stage.
  - Nếu Redis lỗi trong demo: trả lỗi rõ ràng, không fallback âm thầm sang in-memory.
- Tách ingest thành background job:
  - Upload PDF lưu vào `documents/`.
  - Gọi `ingest_documents(rebuild=false, pdf_paths=[uploaded_file])`.
  - Với `rebuild=true`, recreate collection rồi ingest toàn bộ documents.
  - Lưu job status tạm trong Redis.
- Cấu hình môi trường:
```env
AI_SERVICE_PORT=8000
GROQ_API_KEY=...
LLM_PROVIDER=groq
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
REDIS_URL=redis://redis:6379/0
DOCUMENTS_DIR=/app/documents
COLLECTION=atbm_httt
```
- Docker hóa:
  - `ai-service` image Python.
  - `qdrant`, `redis`, `ai-service` cùng network Docker Compose.
  - Spring Boot Gateway gọi `http://ai-service:8000/api/v1/chat`.

## Test Plan
- Unit tests:
  - `POST /chat` validate thiếu `message`, thiếu `session_id`.
  - Redis save/load conversation state đúng slots/history.
  - Reset session xóa state trong Redis.
- Integration tests:
  - Health check kiểm tra Redis + Qdrant.
  - Upload PDF tạo job, job chuyển `queued -> running -> success`.
  - Chat sau ingest trả answer có `sources`.
  - Khi Qdrant không có dữ liệu, trả thông báo không có thông tin thay vì crash.
- Manual demo scenario:
  - Start Docker Compose.
  - Upload PDF Toyota.
  - Gửi câu: “Tôi cần xe 7 chỗ tầm 1 tỷ”.
  - Gửi tiếp: “Nếu đi địa hình thì sao?”
  - Kiểm tra session vẫn nhớ nhu cầu cũ và sources trả về đúng tài liệu.

## Assumptions
- `ai-service` sẽ là Python/FastAPI service riêng, không rewrite sang Spring Boot.
- Spring Boot API Gateway/BFF là điểm duy nhất gọi `ai-service`.
- Giai đoạn đầu dùng Groq only, không dùng Ollama/Gemini fallback.
- Session state lưu Redis để dễ scale hơn in-memory.
- Upload tài liệu chỉ dành cho admin/internal demo, xác thực và phân quyền do Gateway xử lý.
