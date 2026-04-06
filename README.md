# Car Chatbot (RAG)

Repository này chứa mã mẫu cho một chatbot dạng RAG (retrieval-augmented generation) dùng để trả lời thông tin về ô tô TOYOTA.

## Yêu cầu
- Python 3.10 hoặc mới hơn
- `pip` và `virtualenv` (hoặc venv)
- Docker & Docker Compose (tùy chọn, để chạy Qdrant bằng container)

## Cài đặt nhanh (local)

1. Clone repo:

	 git clone https://github.com/Hoc273/car-chatbot-rag.git
	 cd car-chatbot-rag

2. Tạo và kích hoạt môi trường ảo:

	 - PowerShell (Windows):
		 .venv\\Scripts\\Activate.ps1
	 - CMD (Windows):
		 .venv\\Scripts\\activate
	 - macOS / Linux:
		 source .venv/bin/activate

	 Nếu chưa có venv:

	 python -m venv .venv

3. Cài đặt phụ thuộc:

	 pip install -r requirements.txt

## Thiết lập Qdrant (vector DB)

Repo có tập tin `docker-compose.yml` để khởi chạy Qdrant. Để chạy Qdrant bằng Docker Compose:

	 docker-compose up -d

Dữ liệu Qdrant sẽ được lưu trong thư mục `qdrant_storage` trong repo.

## Chạy ứng dụng

Lưu ý: file `main.py` trong repo hiện trống; tùy thuộc vào cách bạn triển khai, có một số cách khởi chạy:

- Nếu bạn phát triển API với FastAPI, khởi động bằng Uvicorn:

	uvicorn main:app --reload

- Nếu bạn có một script chính để chạy (ví dụ `rag.py`), chạy trực tiếp:

	python rag.py

Thay `main:app` hoặc `rag.py` bằng điểm vào thực tế của dự án khi bạn đã triển khai entrypoint.

## Cấu trúc chính

- `main.py` — (entrypoint API, có thể là FastAPI)
- `rag.py` — xử lý luồng RAG
- `embed.py` — tạo embedding
- `vector_database.py` — tương tác với Qdrant
- `data_processing/` — mã trích xuất và tiền xử lý tài liệu
- `qdrant_storage/` — dữ liệu Qdrant (đã có trong repo mẫu)

## Ghi chú

- Kiểm tra và đặt các biến môi trường (nếu cần) trước khi chạy.
- Nếu muốn chạy trong môi trường sản xuất, cân nhắc cấu hình và bảo mật cho Qdrant và API server.

---

Nếu bạn muốn, tôi có thể bổ sung ví dụ cấu hình môi trường, script khởi tạo bộ dữ liệu, hoặc một entrypoint `main.py` mẫu để chạy FastAPI.