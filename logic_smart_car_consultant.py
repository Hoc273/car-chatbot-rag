"""
logic_smart_car_consultant.py
Logic tư vấn xe thông minh — quyết định cách trả lời dựa trên state + slots.

Chiến lược: Tư vấn ngay với thông tin đã có, hỏi thêm sau.
"""

from typing import Dict, Any, List, Optional, Tuple
from conversation_state_manager import ConversationState, STAGE_COLLECTING, STAGE_ADVISING

# ── Cấu hình ──────────────────────────────────────────────────────────────────

# Tên hiển thị của từng slot (dùng khi hỏi lại người dùng)
SLOT_DISPLAY = {
    "budget":           "ngân sách",
    "seats":            "số chỗ ngồi",
    "purpose":          "mục đích sử dụng",
    "fuel":             "loại nhiên liệu",
    "region":           "khu vực đi lại",
    "brand_preference": "hãng xe ưa thích",
}

# Câu hỏi gợi ý cho từng slot còn thiếu
SLOT_QUESTIONS = {
    "budget":   "Anh/chị có thể cho biết ngân sách dự kiến là bao nhiêu không? (ví dụ: 800 triệu, 1 tỷ 2...)",
    "seats":    "Anh/chị cần xe mấy chỗ ngồi?",
    "purpose":  "Xe chủ yếu dùng cho mục đích gì ạ? (gia đình, kinh doanh, cá nhân, off-road...)",
    "fuel":     "Anh/chị có ưu tiên loại nhiên liệu nào không? (xăng, dầu, hybrid, điện)",
    "region":   "Xe chủ yếu chạy ở đâu ạ? (thành phố, đường dài, địa hình khó...)",
    "brand_preference": "Anh/chị có muốn xe Toyota cụ thể hay chỉ cần phù hợp nhu cầu?",
}

# Slot ưu tiên hỏi trước (theo thứ tự quan trọng)
SLOT_PRIORITY = ["budget", "seats", "purpose", "fuel", "region", "brand_preference"]

# Số slot tối thiểu để tư vấn ngay
MIN_SLOTS_TO_ADVISE = 1


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_slot_context(slots: Dict[str, Any]) -> str:
    """Tóm tắt thông tin đã biết về khách hàng thành đoạn văn."""
    filled = {k: v for k, v in slots.items() if v is not None}
    if not filled:
        return "Chưa có thông tin cụ thể về nhu cầu của khách hàng."

    lines = ["📋 Thông tin đã biết về khách hàng:"]
    mapping = {
        "budget":           lambda v: f"  • Ngân sách: {v:,.0f} triệu VND",
        "seats":            lambda v: f"  • Số chỗ ngồi: {v} chỗ",
        "purpose":          lambda v: f"  • Mục đích sử dụng: {v}",
        "fuel":             lambda v: f"  • Nhiên liệu ưa thích: {v}",
        "region":           lambda v: f"  • Khu vực đi lại: {v}",
        "brand_preference": lambda v: f"  • Hãng xe: {v}",
    }
    for key in SLOT_PRIORITY:
        if key in filled:
            lines.append(mapping[key](filled[key]))

    return "\n".join(lines)


def build_followup_question(missing: List[str], turn_count: int) -> Optional[str]:
    """
    Tạo câu hỏi follow-up cho slot còn thiếu quan trọng nhất.
    Không hỏi nếu đã hỏi nhiều lượt (tránh làm phiền người dùng).
    """
    if not missing or turn_count >= 6:
        return None

    # Hỏi theo thứ tự priority
    for slot in SLOT_PRIORITY:
        if slot in missing:
            return SLOT_QUESTIONS[slot]

    return None


def build_advise_prompt(
    query: str,
    state: ConversationState,
    context: str,
) -> str:
    """
    Xây dựng user message đưa vào LLM để tư vấn.
    Gộp: slot summary + RAG context + lịch sử + câu hỏi.
    """
    slot_ctx = build_slot_context(state.slots)
    history  = state.get_history_text(n_turns=3)

    parts = []

    if slot_ctx:
        parts.append(slot_ctx)

    if history:
        parts.append(f"📝 Lịch sử hội thoại gần đây:\n{history}")

    if context:
        parts.append(f"📚 Ngữ cảnh từ tài liệu:\n{context}")

    parts.append(f"❓ Câu hỏi hiện tại: {query}")

    return "\n\n---\n\n".join(parts)


def build_greeting_prompt(query: str, state: ConversationState) -> str:
    """Prompt đơn giản cho lượt chào hỏi đầu tiên."""
    return query


# ── Decision engine ───────────────────────────────────────────────────────────

class SmartCarConsultant:
    """
    Quyết định cách xử lý mỗi lượt hội thoại.

    Chiến lược: tư vấn ngay với thông tin đã có, append câu hỏi follow-up sau.

    Workflow:
        decision = consultant.decide(query, state, retrieved_docs)
        # decision.prompt   → đưa vào LLM
        # decision.followup → append vào cuối response
        # decision.skip_rag → True nếu không cần gọi RAG
    """

    # ── Quyết định chính ──────────────────────────────────────────────────────

    def decide(
        self,
        query: str,
        state: ConversationState,
        rag_context: str = "",
    ) -> "ConsultDecision":
        """
        Trả về ConsultDecision gồm:
            prompt    : message gửi vào LLM
            followup  : câu hỏi bổ sung (append sau response LLM)
            skip_rag  : True nếu không cần RAG retrieval
            stage     : stage hiện tại
        """
        stage   = state.stage
        missing = state.get_missing_slots()

        # ── Greeting ─────────────────────────────────────────────────────────
        if stage == "greeting" or state.last_intent == "greeting":
            return ConsultDecision(
                prompt    = build_greeting_prompt(query, state),
                followup  = self._opening_question(),
                skip_rag  = True,
                stage     = stage,
            )

        # ── Collecting + Advising: tư vấn ngay với info đã có ────────────────
        prompt   = build_advise_prompt(query, state, rag_context)
        followup = build_followup_question(missing, state.turn_count)

        return ConsultDecision(
            prompt   = prompt,
            followup = followup,
            skip_rag = False,
            stage    = stage,
        )

    def _opening_question(self) -> str:
        return (
            "\n\nĐể tôi tư vấn tốt hơn, anh/chị có thể cho biết:\n"
            "- Ngân sách dự kiến?\n"
            "- Cần xe mấy chỗ?\n"
            "- Mục đích sử dụng chính?"
        )

    # ── Post-process response ─────────────────────────────────────────────────

    def compose_final_response(
        self,
        llm_response: str,
        decision: "ConsultDecision",
    ) -> str:
        """
        Ghép LLM response + follow-up question.
        """
        if decision.followup:
            return f"{llm_response}\n\n---\n💬 {decision.followup}"
        return llm_response

    # ── Slot adequacy check ───────────────────────────────────────────────────

    def needs_more_info(self, state: ConversationState) -> bool:
        """True nếu chưa có đủ thông tin tư vấn."""
        return not state.has_enough_info()

    def get_slot_summary_for_user(self, state: ConversationState) -> str:
        """Tóm tắt ngắn gọn những gì đã biết — dùng khi cần confirm lại."""
        filled = state.get_filled_slots()
        if not filled:
            return "Tôi chưa có thông tin nào về nhu cầu của anh/chị."
        parts = [f"{SLOT_DISPLAY.get(k, k)}: {v}" for k, v in filled.items()]
        return "Thông tin tôi đã ghi nhận: " + ", ".join(parts) + "."


# ── Data class cho decision ───────────────────────────────────────────────────

class ConsultDecision:
    def __init__(
        self,
        prompt:   str,
        followup: Optional[str],
        skip_rag: bool,
        stage:    str,
    ):
        self.prompt   = prompt
        self.followup = followup
        self.skip_rag = skip_rag
        self.stage    = stage

    def __repr__(self):
        return (
            f"ConsultDecision(stage={self.stage!r}, "
            f"skip_rag={self.skip_rag}, "
            f"has_followup={self.followup is not None})"
        )


# Singleton
smart_consultant = SmartCarConsultant()


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from conversation_state_manager import ConversationStateManager
    from slot_extractor import extract_slots

    mgr   = ConversationStateManager()
    state = mgr.create("test-001")

    turns = [
        ("Xin chào!", "greeting"),
        ("Tôi muốn mua xe gia đình, ngân sách 1 tỷ", "car_advice"),
        ("Cần 7 chỗ, chạy trong thành phố", "usage_filter"),
    ]

    for query, intent in turns:
        slots = extract_slots(query)
        state.update_slots(slots)
        state.update_stage(intent)

        decision = smart_consultant.decide(query, state, rag_context="[mock context]")
        print(f"Turn {state.turn_count + 1} | Stage: {decision.stage}")
        print(f"Decision: {decision}")
        print(f"Prompt preview: {decision.prompt[:80]}...")
        print(f"Follow-up: {decision.followup}")
        print()

        state.add_turn(query, "[mock response]")