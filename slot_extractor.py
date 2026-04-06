"""
slot_extractor.py
Trích xuất các slot thông tin từ câu hỏi người dùng bằng LLM + regex fallback.

Slots:
    budget          : ngân sách (triệu VND, float)
    seats           : số chỗ ngồi (int)
    purpose         : mục đích sử dụng (str)
    fuel            : nhiên liệu — xăng | dầu | hybrid | điện (str)
    region          : khu vực đi lại — thành phố | đường dài | địa hình | hỗn hợp (str)
    brand_preference: hãng xe ưa thích — toyota | không quan trọng (str)
"""

import json
import re
from typing import Dict, Any, Optional

from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
groq_client  = Groq(api_key=GROQ_API_KEY)

# ── Định nghĩa slots ──────────────────────────────────────────────────────────

SLOT_SCHEMA = {
    "budget": {
        "type": "float",
        "unit": "triệu VND",
        "description": "Ngân sách mua xe",
        "example": 800.0,
    },
    "seats": {
        "type": "int",
        "description": "Số chỗ ngồi mong muốn",
        "example": 7,
    },
    "purpose": {
        "type": "str",
        "enum": ["gia đình", "kinh doanh", "cá nhân", "off-road", "chạy dịch vụ", "hỗn hợp"],
        "description": "Mục đích sử dụng chính",
        "example": "gia đình",
    },
    "fuel": {
        "type": "str",
        "enum": ["xăng", "dầu", "hybrid", "điện"],
        "description": "Loại nhiên liệu ưa thích",
        "example": "xăng",
    },
    "region": {
        "type": "str",
        "enum": ["thành phố", "đường dài", "địa hình", "hỗn hợp"],
        "description": "Khu vực / địa hình sử dụng chính",
        "example": "thành phố",
    },
    "brand_preference": {
        "type": "str",
        "enum": ["toyota", "không quan trọng"],
        "description": "Hãng xe ưa thích",
        "example": "toyota",
    },
}

EXTRACTOR_SYSTEM = """
Bạn là bộ trích xuất thông tin (slot extractor) cho chatbot tư vấn xe Toyota.

Nhiệm vụ: đọc câu của người dùng và trích xuất các slot sau:
- budget          : ngân sách (số thực, đơn vị triệu VND). Ví dụ: "1 tỷ" → 1000, "800 triệu" → 800, "1.5 tỷ" → 1500
- seats           : số chỗ ngồi (số nguyên)
- purpose         : một trong [gia đình, kinh doanh, cá nhân, off-road, chạy dịch vụ, hỗn hợp]
- fuel            : một trong [xăng, dầu, hybrid, điện]
- region          : một trong [thành phố, đường dài, địa hình, hỗn hợp]
- brand_preference: một trong [toyota, không quan trọng]

Ngoài ra, hãy xác định 2 field đặc biệt:

"overrides": danh sách slot người dùng CHỦ ĐỘNG THAY ĐỔI sang giá trị mới
- Dấu hiệu: "thay vì", "đổi thành", "thôi cho tôi X", nêu lại slot với giá trị khác rõ ràng
- Ví dụ: "thôi cho tôi xe 4 chỗ"      → overrides: ["seats"],  seats: 4
- Ví dụ: "đổi ngân sách thành 1 tỷ"   → overrides: ["budget"], budget: 1000
- Ví dụ: "tôi muốn xe xăng thôi"      → overrides: ["fuel"],   fuel: "xăng"
- Nếu không có thay đổi rõ ràng       → overrides: []

"clears": danh sách slot người dùng muốn XOÁ / BỎ yêu cầu đó đi (set về null)
- Dấu hiệu: "không cần X nữa", "bỏ yêu cầu X", "không quan tâm X", "thôi không cần X"
- Ví dụ: "không cần xe 7 chỗ nữa"         → clears: ["seats"],  seats: null
- Ví dụ: "thôi không cần hybrid"           → clears: ["fuel"],   fuel: null
- Ví dụ: "bỏ yêu cầu ngân sách đi"        → clears: ["budget"], budget: null
- Ví dụ: "không quan tâm số chỗ và vùng"  → clears: ["seats", "region"]
- Slot trong clears phải để null trong JSON
- Nếu không có xoá rõ ràng               → clears: []

Lưu ý: một slot không thể vừa trong overrides vừa trong clears.

Quy tắc:
- Nếu không tìm thấy thông tin cho slot nào → để null
- Chỉ trả về JSON, không giải thích thêm
- Không được suy đoán nếu không có thông tin rõ ràng

Định dạng trả về:
{
  "budget": <float|null>,
  "seats": <int|null>,
  "purpose": <str|null>,
  "fuel": <str|null>,
  "region": <str|null>,
  "brand_preference": <str|null>,
  "overrides": <list[str]>,
  "clears": <list[str]>
}
""".strip()


# ── Regex fallback ────────────────────────────────────────────────────────────

def _regex_extract_budget(text: str) -> Optional[float]:
    t = text.lower()
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*tỷ\s*(\d+)?', t)
    if m:
        ty    = float(m.group(1).replace(",", "."))
        extra = float(m.group(2)) * 100 if m.group(2) else 0
        return ty * 1000 + extra
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*tỷ', t)
    if m:
        return float(m.group(1).replace(",", ".")) * 1000
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:triệu|tr\b)', t)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _regex_extract_seats(text: str) -> Optional[int]:
    m = re.search(r'(\d+)\s*chỗ', text.lower())
    return int(m.group(1)) if m else None


def _regex_extract_fuel(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["hybrid", "xăng lai"]):        return "hybrid"
    if any(k in t for k in ["điện", "electric"]):          return "điện"
    if any(k in t for k in ["dầu", "diesel"]):             return "dầu"
    if "xăng" in t:                                         return "xăng"
    return None


def _regex_extract_region(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["off-road", "địa hình", "núi", "rừng", "đường xấu"]): return "địa hình"
    if any(k in t for k in ["đường dài", "cao tốc", "liên tỉnh", "du lịch"]):      return "đường dài"
    if any(k in t for k in ["thành phố", "nội thành", "phố", "đô thị"]):           return "thành phố"
    return None


def _regex_fallback(text: str) -> Dict[str, Any]:
    return {
        "budget":           _regex_extract_budget(text),
        "seats":            _regex_extract_seats(text),
        "purpose":          None,
        "fuel":             _regex_extract_fuel(text),
        "region":           _regex_extract_region(text),
        "brand_preference": None,
        "overrides":        [],
        "clears":           [],   # regex không detect được intent xoá
    }


# ── LLM extractor ─────────────────────────────────────────────────────────────

def extract_slots(text: str) -> Dict[str, Any]:
    """
    Trích xuất slots từ text (có thể là 1 câu hoặc đoạn hội thoại).
    Trả về dict với 6 keys, giá trị None nếu không tìm thấy.
    """
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM},
        {"role": "user",   "content": f'Đoạn hội thoại:\n"""\n{text}\n"""'},
    ]

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown nếu có
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        slots = json.loads(raw.strip())

        # Đảm bảo đủ keys
        for key in SLOT_SCHEMA:
            slots.setdefault(key, None)
        slots.setdefault("overrides", [])
        slots.setdefault("clears", [])

        # Validate là list
        if not isinstance(slots["overrides"], list):
            slots["overrides"] = []
        if not isinstance(slots["clears"], list):
            slots["clears"] = []

        # Đảm bảo slot trong clears là null
        for key in slots["clears"]:
            if key in SLOT_SCHEMA:
                slots[key] = None

        # Type coercion nhẹ
        if slots["budget"] is not None:
            slots["budget"] = float(slots["budget"])
        if slots["seats"] is not None:
            slots["seats"] = int(slots["seats"])

        return slots

    except Exception as e:
        print(f"⚠️  Slot extractor LLM lỗi ({e}), dùng regex fallback")
        return _regex_fallback(text)


def merge_slots(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge slots mới vào slots đã có.

    Logic:
    - Slot trong `clears`    → set về None (xoá yêu cầu)
    - Slot trong `overrides` → ghi đè bằng giá trị mới
    - Slot còn lại           → chỉ fill nếu hiện tại là None
    """
    merged    = dict(existing)
    overrides = new.get("overrides", []) or []
    clears    = new.get("clears",    []) or []

    # Bước 1: xoá slot user không cần nữa
    for key in clears:
        if key in SLOT_SCHEMA:
            merged[key] = None

    # Bước 2: merge slot mới
    for key in SLOT_SCHEMA:
        val = new.get(key)
        if val is None:
            continue
        if key in overrides or merged.get(key) is None:
            merged[key] = val

    return merged


def empty_slots() -> Dict[str, Any]:
    """Trả về dict slots rỗng (tất cả None)."""
    return {key: None for key in SLOT_SCHEMA}


def filled_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """Chỉ trả về các slots đã có giá trị."""
    return {k: v for k, v in slots.items() if v is not None}


def missing_slots(slots: Dict[str, Any]) -> list:
    """Danh sách slot chưa có giá trị."""
    return [k for k, v in slots.items() if v is None]


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test extract
    tests = [
        "Tôi muốn mua xe 7 chỗ, ngân sách khoảng 1 tỷ 2, chủ yếu đi trong thành phố",
        "Thôi cho tôi xe 4 chỗ thôi",
        "Đổi ngân sách thành 900 triệu",
        "Tôi muốn xe xăng thay vì hybrid",
        "Không cần xe 7 chỗ nữa",
        "Thôi không cần hybrid, bỏ luôn yêu cầu địa hình",
        "Không quan tâm số chỗ ngồi nữa",
    ]
    for q in tests:
        slots = extract_slots(q)
        print(f"Q: {q}")
        print(f"→ Overrides : {slots.get('overrides', [])}")
        print(f"→ Clears    : {slots.get('clears', [])}")
        print(f"→ Slots     : { {k:v for k,v in slots.items() if k not in ('overrides','clears')} }\n")

    # Test merge: override + clear cùng lúc
    print("=" * 55)
    print("Test merge_slots — override + clear:")
    existing = {
        "budget": 500.0, "seats": 7, "purpose": "kinh doanh",
        "fuel": "hybrid", "region": "địa hình", "brand_preference": None,
    }
    new = {
        "budget": None, "seats": 4,   "purpose": None,
        "fuel":   None, "region": None, "brand_preference": None,
        "overrides": ["seats"],
        "clears":    ["fuel", "region"],
    }
    result = merge_slots(existing, new)
    print(f"Before : {existing}")
    print(f"Action : overrides={new['overrides']}, clears={new['clears']}")
    print(f"After  : {result}")
    assert result["seats"]  == 4,    "❌ override seats thất bại"
    assert result["fuel"]   is None, "❌ clear fuel thất bại"
    assert result["region"] is None, "❌ clear region thất bại"
    assert result["budget"] == 500.0,"❌ budget bị thay đổi nhầm"
    print("✅ Tất cả assertions passed")