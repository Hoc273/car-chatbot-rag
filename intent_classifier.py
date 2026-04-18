import json
from typing import Dict
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Danh sách intent ──────────────────────────────────────────────────────────
INTENTS = {
    "greeting":         "Chào hỏi, cảm ơn, tạm biệt, hỏi bot là ai",
    "car_advice":       "Hỏi tư vấn mua xe, so sánh xe, hỏi giá, thông số kỹ thuật Toyota",
    "car_info":         "Hỏi thông tin chung về một dòng xe Toyota cụ thể",
    "budget_filter":    "Hỏi xe theo ngân sách / tầm giá",
    "seat_filter":      "Hỏi xe theo số chỗ ngồi",
    "usage_filter":     "Hỏi xe theo mục đích dùng (gia đình, kinh doanh, off-road...)",
    "out_of_scope":     "Câu hỏi không liên quan đến Toyota hoặc mua xe",
    "sensitive":        "Nội dung nhạy cảm, chính trị, bạo lực, 18+",
}

CLASSIFIER_PROMPT = """
Bạn là bộ phân loại ý định (intent classifier) cho chatbot tư vấn mua xe Toyota.

Danh sách intent và mô tả:
{intent_list}

Nhiệm vụ:
- Đọc câu hỏi của người dùng
- Chọn ĐÚNG MỘT intent phù hợp nhất từ danh sách trên
- Trả về JSON theo định dạng sau (KHÔNG giải thích thêm):

{{
  "intent": "<tên intent>",
  "confidence": <số thực 0.0 - 1.0>,
  "reason": "<giải thích ngắn gọn 1 câu>"
}}
""".strip()


def classify_intent(query: str) -> Dict:
    """
    Phân loại intent của query.
    Trả về dict: { intent, confidence, reason }
    Fallback về 'car_advice' nếu LLM lỗi hoặc parse thất bại.
    """
    intent_list = "\n".join(
        f"- {k}: {v}" for k, v in INTENTS.items()
    )

    messages = [
        {
            "role": "system",
            "content": CLASSIFIER_PROMPT.format(intent_list=intent_list),
        },
        {
            "role": "user",
            "content": f'Câu hỏi: "{query}"',
        },
    ]

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0,      # deterministic
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences nếu có
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

        # Validate intent hợp lệ
        if result.get("intent") not in INTENTS:
            result["intent"] = "car_advice"

        return result

    except Exception as e:
        print(f"⚠️  Intent classifier lỗi ({e}), fallback 'car_advice'")
        return {
            "intent": "car_advice",
            "confidence": 0.5,
            "reason": "fallback do lỗi classifier",
        }


