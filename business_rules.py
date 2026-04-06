import re
from typing import Dict, Tuple, Optional

# ── Cấu hình ──────────────────────────────────────────────────────────────────

# Ngân sách hợp lệ (triệu VND)
BUDGET_MIN_MILLION = 400
BUDGET_MAX_MILLION = 5_000

# Số chỗ hợp lệ
SEAT_VALID = {2, 4, 5, 7, 8, 9}

# Từ khoá nhạy cảm
SENSITIVE_KEYWORDS = [
    "chính trị", "đảng", "bầu cử", "biểu tình",
    "bạo lực", "giết", "khủng bố", "ma túy",
    "sex", "18+", "khiêu dâm", "cờ bạc",
]

# Intent bị chặn hoàn toàn
BLOCKED_INTENTS = {"out_of_scope", "sensitive"}

# Câu trả lời cố định cho từng trường hợp
RULE_RESPONSES = {
    "out_of_scope": (
        "Xin lỗi, tôi chỉ có thể tư vấn về xe Toyota. "
        "Bạn có muốn tìm hiểu dòng xe nào của Toyota không? 🚗"
    ),
    "sensitive": (
        "Xin lỗi, tôi không thể hỗ trợ nội dung này. "
        "Hãy đặt câu hỏi liên quan đến xe Toyota nhé!"
    ),
    "budget_too_low": (
        "Ngân sách bạn nhập có vẻ thấp hơn mức tối thiểu của xe Toyota "
        f"(khoảng {BUDGET_MIN_MILLION} triệu đồng). "
        "Bạn có muốn xem các dòng xe entry-level như Vios hay Wigo không?"
    ),
    "budget_too_high": (
        "Ngân sách bạn nhập vượt quá dải giá hiện tại trong dữ liệu của tôi. "
        "Tôi sẽ tư vấn các dòng xe cao cấp nhất Toyota hiện có."
    ),
    "seat_invalid": (
        "Số chỗ ngồi bạn yêu cầu không phổ biến trong dòng xe Toyota. "
        f"Các lựa chọn thông thường là: {sorted(SEAT_VALID)} chỗ. "
        "Bạn muốn chọn loại nào?"
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_budget(query: str) -> Optional[float]:
    """
    Trích xuất ngân sách từ query (đơn vị triệu VND).
    Hỗ trợ: '800 triệu', '1.5 tỷ', '1 tỷ 2', '800tr'
    """
    query_lower = query.lower()

    # Dạng: "1 tỷ 2" hoặc "1 tỷ rưỡi"
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*tỷ\s*(\d+)?', query_lower)
    if m:
        ty = float(m.group(1).replace(",", "."))
        extra = float(m.group(2)) * 100 if m.group(2) else 0
        return ty * 1000 + extra

    # Dạng: "1.5 tỷ"
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*tỷ', query_lower)
    if m:
        return float(m.group(1).replace(",", ".")) * 1000

    # Dạng: "800 triệu" hoặc "800tr"
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:triệu|tr\b)', query_lower)
    if m:
        return float(m.group(1).replace(",", "."))

    return None


def _extract_seats(query: str) -> Optional[int]:
    """
    Trích xuất số chỗ ngồi từ query.
    Hỗ trợ: '7 chỗ', '5 chỗ ngồi'
    """
    m = re.search(r'(\d+)\s*chỗ', query.lower())
    if m:
        return int(m.group(1))
    return None


# ── Engine chính ──────────────────────────────────────────────────────────────

class BusinessRules:
    """
    Kiểm tra tất cả business rules trước khi đưa query vào RAG pipeline.

    Cách dùng:
        rules = BusinessRules()
        blocked, reason, response = rules.check(query, intent_result)
        if blocked:
            return response
    """

    def check(
        self,
        query: str,
        intent_result: Dict,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Trả về (blocked: bool, rule_name: str, response: str | None)
        - blocked=True  → dừng pipeline, trả response luôn
        - blocked=False → tiếp tục RAG bình thường
        """
        intent = intent_result.get("intent", "")

        # Rule 1: Chặn intent ngoài scope
        if intent in BLOCKED_INTENTS:
            return True, intent, RULE_RESPONSES[intent]

        # Rule 2: Chặn nội dung nhạy cảm theo keyword (double-check)
        query_lower = query.lower()
        for kw in SENSITIVE_KEYWORDS:
            if kw in query_lower:
                return True, "sensitive", RULE_RESPONSES["sensitive"]

        # Rule 3: Validate ngân sách
        budget = _extract_budget(query)
        if budget is not None:
            if budget < BUDGET_MIN_MILLION:
                return True, "budget_too_low", RULE_RESPONSES["budget_too_low"]
            if budget > BUDGET_MAX_MILLION:
                # Không chặn cứng, chỉ cảnh báo → trả về warning nhưng vẫn tiếp tục
                return False, "budget_too_high", RULE_RESPONSES["budget_too_high"]

        # Rule 4: Validate số chỗ
        seats = _extract_seats(query)
        if seats is not None and seats not in SEAT_VALID:
            return True, "seat_invalid", RULE_RESPONSES["seat_invalid"]

        # Rule 5: Giới hạn xe chỉ trong DB — enforce qua RAG (nếu không retrieve được → trả lời mặc định)
        # → Đã xử lý trong rag.py khi retrieved = []

        return False, "pass", None


# Singleton
rules_engine = BusinessRules()


# ── Test nhanh ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from intent_classifier import classify_intent

    tests = [
        "Tôi muốn mua xe 7 chỗ khoảng 1 tỷ 2",
        "Cho tôi xe 3 chỗ",
        "Ngân sách 50 triệu mua được xe không?",
        "Hôm nay thời tiết thế nào?",
        "Tôi muốn xe 5 chỗ khoảng 800 triệu",
    ]

    for q in tests:
        intent = classify_intent(q)
        blocked, rule, response = rules_engine.check(q, intent)
        status = "🚫 BLOCKED" if blocked else "✅ PASS"
        print(f"Q: {q}")
        print(f"→ Intent: {intent['intent']} | Rule: {rule} | {status}")
        if response:
            print(f"→ Response: {response}")
        print()