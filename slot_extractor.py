"""
slot_extractor.py
Trích xuất các slot thông tin từ câu hỏi người dùng bằng LLM + regex fallback.

Slots:
    budget          : ngân sách (triệu VND, float)
    seats           : số chỗ ngồi (int)
    purpose         : mục đích sử dụng (str)
    fuel            : nhiên liệu — xăng | dầu | hybrid | điện (str)
    region          : khu vực đi lại — thành phố | đường dài | địa hình | hỗn hợp (str)
    type_car           : loại xe — sedan | SUV | Đa dụng | Bán Tải | hatchback (str)
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

SLOT_EXTRACTION_ENABLED = os.getenv("ENABLE_SLOT_EXTRACTION", "true").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SLOT_EXTRACTION_INTENTS = {
    intent.strip()
    for intent in os.getenv(
        "SLOT_EXTRACTION_INTENTS",
        "car_advice,budget_filter,seat_filter,usage_filter",
    ).split(",")
    if intent.strip()
}
IMPLICIT_SLOT_OVERRIDE_ENABLED = os.getenv(
    "ENABLE_IMPLICIT_SLOT_OVERRIDE",
    "true",
).lower() in {"1", "true", "yes", "on"}


def should_extract_slots(intent: str) -> bool:
    """Return True when slot extraction should run for this intent."""
    return SLOT_EXTRACTION_ENABLED and intent in SLOT_EXTRACTION_INTENTS

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
    "type_car": {
        "type": "str",
        "enum": ["sedan", "SUV", "đa dụng", "bán tải", "hatchback"],
        "description": "Loại xe mong muốn",
        "example": "SUV",
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
- type_car        : một trong [sedan, SUV, đa dụng, bán tải, hatchback]

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
- Chỉ điền slot khi người dùng nói rõ slot đó hoặc dùng từ đồng nghĩa trực tiếp.
- Không được tự suy ra "gia đình" chỉ vì người dùng nói "đi du lịch" hoặc "mang nhiều đồ".
- Không được tự suy ra "SUV" chỉ vì người dùng nói "đi du lịch", "gầm cao", "rộng" hoặc "mang nhiều đồ".
- Không được tự suy ra "đường dài" chỉ vì người dùng nói "đi du lịch"; chỉ điền khi có từ như "đường dài", "cao tốc", "liên tỉnh", "đường trường".
- Các nhu cầu mô tả chung như "mang nhiều đồ", "cốp rộng", "chở hành lý" chưa thuộc slot nào trong schema → để các slot không liên quan là null.

Định dạng trả về:
{
  "budget": <float|null>,
  "seats": <int|null>,
  "purpose": <str|null>,
  "fuel": <str|null>,
  "region": <str|null>,
  "type_car": <str|null>,
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
    if any(k in t for k in ["đường dài", "cao tốc", "liên tỉnh", "đường trường"]): return "đường dài"
    if any(k in t for k in ["thành phố", "nội thành", "phố", "đô thị"]):           return "thành phố"
    return None


def _regex_fallback(text: str) -> Dict[str, Any]:
    return {
        "budget":           _regex_extract_budget(text),
        "seats":            _regex_extract_seats(text),
        "purpose":          None,
        "fuel":             _regex_extract_fuel(text),
        "region":           _regex_extract_region(text),
        "type_car":         None,
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

        if slots["seats"] is not None and _mentions_people_count_without_seat(text):
            slots["seats"] = None
            slots["overrides"] = [
                key for key in slots["overrides"] if key != "seats"
            ]
        slots = _drop_inferred_slots_without_evidence(text, slots)

        return slots

    except Exception as e:
        print(f"⚠️  Slot extractor LLM lỗi ({e}), dùng regex fallback")
        return _regex_fallback(text)


def merge_slots(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge newly extracted slots into state.

    In a consulting conversation, a short follow-up such as "xe 5 cho thi sao"
    usually means "change the seat filter to 5", even if the extractor did not
    explicitly mark it as an override. We only apply that implicit override when
    exactly one slot is present in the new turn to avoid clobbering broader
    preferences by accident.
    """
    merged    = dict(existing)
    overrides = [
        key for key in (new.get("overrides", []) or [])
        if key in SLOT_SCHEMA
    ]
    clears = [
        key for key in (new.get("clears", []) or [])
        if key in SLOT_SCHEMA
    ]
    overrides = [key for key in overrides if key not in clears]

    # Bước 1: xoá slot user không cần nữa
    for key in clears:
        if key in SLOT_SCHEMA:
            merged[key] = None

    # Bước 2: nếu chỉ có đúng 1 slot được extract và không có slot nào khác
    # → coi như user đang chỉnh slot đó (implicit override)
    filled_new = [k for k in SLOT_SCHEMA if new.get(k) is not None]
    if (
        IMPLICIT_SLOT_OVERRIDE_ENABLED
        and len(filled_new) == 1
        and filled_new[0] not in overrides
    ):
        overrides = overrides + filled_new  # ← tự động override slot duy nhất

    # Bước 3: merge slot mới
    for key in SLOT_SCHEMA:
        val = new.get(key)
        if val is None:
            continue
        if key in overrides or merged.get(key) is None:
            merged[key] = val

    return merged


def _mentions_people_count_without_seat(text: str) -> bool:
    t = text.lower()
    has_people_count = (
        re.search(r"\d+\s*(?:ng\u01b0\u1eddi|th\u00e0nh vi\u00ean)", t)
        is not None
    )
    seat_words = [
        "ch\u1ed7",
        "gh\u1ebf",
        "ng\u1ed3i",
        "seat",
    ]
    has_seat_word = any(k in t for k in seat_words)
    return has_people_count and not has_seat_word


def _drop_inferred_slots_without_evidence(
    text: str,
    slots: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = dict(slots)
    for key in ("purpose", "region", "type_car"):
        value = cleaned.get(key)
        if value is not None and not _has_slot_evidence(text, key, str(value)):
            cleaned[key] = None
            cleaned["overrides"] = [
                item for item in cleaned.get("overrides", []) if item != key
            ]
    return cleaned


def _has_slot_evidence(text: str, key: str, value: str) -> bool:
    t = text.lower()
    v = value.lower()
    evidence = {
        "purpose": {
            "gia đình": ["gia đình", "nhà tôi", "vợ", "chồng", "con", "bố mẹ", "ba mẹ"],
            "kinh doanh": ["kinh doanh", "doanh nghiệp", "công ty", "buôn bán"],
            "cá nhân": ["cá nhân", "một mình", "đi làm", "riêng tôi"],
            "off-road": ["off-road", "leo núi", "địa hình", "đường xấu", "đồi núi", "rừng"],
            "chạy dịch vụ": ["chạy dịch vụ", "dịch vụ", "taxi", "grab", "be", "chở khách"],
            "hỗn hợp": ["hỗn hợp", "nhiều mục đích", "đa mục đích", "vừa"],
        },
        "region": {
            "thành phố": ["thành phố", "nội thành", "đô thị", "trong phố"],
            "đường dài": ["đường dài", "cao tốc", "liên tỉnh", "đường trường"],
            "địa hình": ["địa hình", "off-road", "leo núi", "đường xấu", "đồi núi", "rừng"],
            "hỗn hợp": ["hỗn hợp", "nhiều địa hình", "nhiều cung đường", "đa địa hình"],
        },
        "type_car": {
            "sedan": ["sedan"],
            "suv": ["suv", "xe thể thao đa dụng"],
            "đa dụng": ["đa dụng", "mpv"],
            "bán tải": ["bán tải", "pickup", "pick-up"],
            "hatchback": ["hatchback"],
        },
    }
    return any(marker in t for marker in evidence.get(key, {}).get(v, []))


def empty_slots() -> Dict[str, Any]:
    """Trả về dict slots rỗng (tất cả None)."""
    return {key: None for key in SLOT_SCHEMA}
