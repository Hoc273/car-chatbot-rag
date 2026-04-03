import os
from typing import List, Dict, Any
from groq import Groq
import google.generativeai as genai

from embed import embed_query
from vector_db import search

# ── Cấu hình ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

GROQ_MODEL   = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"
TOP_K        = 5

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp chuyên về An toàn và Bảo mật Hệ thống Thông tin.
Chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu ngữ cảnh không đủ thông tin, hãy nói rõ điều đó.
Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc. Trích dẫn số trang nếu có thể."""


def build_context(retrieved: List[Dict[str, Any]]) -> str:
    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(
            f"[{i}] (Trang {r['page']}, độ liên quan: {r['score']:.2f})\n{r['content']}"
        )
    return "\n\n---\n\n".join(parts)


def ask_groq(messages: List[Dict]) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def ask_gemini(messages: List[Dict]) -> str:
    """Fallback khi Groq lỗi/hết quota."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    # Ghép messages thành prompt đơn giản
    prompt = "\n\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in messages
        if m["role"] != "system"
    )
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\n{prompt}",
        generation_config={"temperature": 0.2, "max_output_tokens": 1024},
    )
    return response.text


def answer(query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
    """
    RAG pipeline hoàn chỉnh:
    1. Embed query
    2. Retrieve từ Qdrant
    3. Build prompt với context
    4. Gọi Groq (fallback Gemini)
    """
    if chat_history is None:
        chat_history = []

    # 1. Retrieve
    query_vec  = embed_query(query)
    retrieved  = search(query_vec, top_k=TOP_K, score_threshold=0.35)

    if not retrieved:
        return {
            "answer":    "Không tìm thấy thông tin liên quan trong tài liệu.",
            "sources":   [],
            "model_used": None,
        }

    # 2. Build messages
    context = build_context(retrieved)
    user_msg = f"""Ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {query}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history[-6:])   # giữ 3 lượt gần nhất
    messages.append({"role": "user", "content": user_msg})

    # 3. Generate với fallback
    model_used = GROQ_MODEL
    try:
        answer_text = ask_groq(messages)
    except Exception as e:
        print(f"⚠️  Groq lỗi ({e}), fallback sang Gemini...")
        answer_text = ask_gemini(messages)
        model_used  = GEMINI_MODEL

    sources = [{"page": r["page"], "score": round(r["score"], 3)} for r in retrieved]

    return {
        "answer":     answer_text,
        "sources":    sources,
        "model_used": model_used,
    }


# ── CLI loop ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 RAG Chatbot — ATBM HTTT (gõ 'exit' để thoát)\n")
    history = []

    while True:
        query = input("Bạn: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "thoát"):
            break

        result = answer(query, history)

        print(f"\n🤖 ({result['model_used']}):")
        print(result["answer"])
        print(f"\n📚 Nguồn: {result['sources']}\n")
        print("-" * 60)

        # Cập nhật history (chỉ giữ nội dung câu hỏi gốc, không kèm context)
        history.append({"role": "user",      "content": query})
        history.append({"role": "assistant", "content": result["answer"]})