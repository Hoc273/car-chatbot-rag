"""
conversation_state_manager.py
Quản lý trạng thái hội thoại theo từng session (in-memory).

State mỗi session gồm:
    session_id      : định danh session
    history         : lịch sử hội thoại [{"role": ..., "content": ...}]
    slots           : thông tin đã thu thập được
    intent_history  : danh sách intent qua các lượt
    turn_count      : số lượt hội thoại
    stage           : giai đoạn hội thoại (greeting|collecting|advising|done)
    last_intent     : intent của lượt gần nhất
"""

import uuid
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from slot_extractor import empty_slots

# ── Cấu hình ──────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS  = 10   # Số lượt tối đa giữ trong history (mỗi lượt = 2 message)
SESSION_TTL_SECS   = 3600 # Session hết hạn sau 1 tiếng không hoạt động

# Các stage của hội thoại
STAGE_GREETING   = "greeting"
STAGE_COLLECTING = "collecting"   # Đang thu thập thông tin
STAGE_ADVISING   = "advising"     # Đang tư vấn
STAGE_DONE       = "done"         # Kết thúc


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ConversationState:
    session_id:     str                  = field(default_factory=lambda: str(uuid.uuid4()))
    slots:          Dict[str, Any]       = field(default_factory=empty_slots)
    history:        List[Dict[str, str]] = field(default_factory=list)
    intent_history: List[str]            = field(default_factory=list)
    stage:          str                  = STAGE_GREETING
    last_intent:    Optional[str]        = None
    turn_count:     int                  = 0
    created_at:     float                = field(default_factory=time.time)
    updated_at:     float                = field(default_factory=time.time)

    # ── Slot helpers ──────────────────────────────────────────────────────────

    def update_slots(self, new_slots: Dict[str, Any]) -> None:
        """Merge slots mới vào state. Không ghi đè slot đã có."""
        from slot_extractor import merge_slots
        self.slots = merge_slots(self.slots, new_slots)

    def get_filled_slots(self) -> Dict[str, Any]:
        return {k: v for k, v in self.slots.items() if v is not None}

    def get_missing_slots(self) -> List[str]:
        return [k for k, v in self.slots.items() if v is None]

    def has_enough_info(self) -> bool:
        """
        Đủ thông tin để tư vấn khi có ít nhất 1 trong các slot chính:
        budget, seats, purpose — không cần đủ tất cả.
        """
        key_slots = ["budget", "seats", "purpose"]
        return any(self.slots.get(s) is not None for s in key_slots)

    # ── History helpers ───────────────────────────────────────────────────────

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Thêm 1 lượt hội thoại (user + assistant) vào history."""
        self.history.append({"role": "user",      "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        self.turn_count += 1
        self.updated_at = time.time()

        # Trim history — giữ tối đa MAX_HISTORY_TURNS lượt gần nhất
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def get_recent_history(self, n_turns: int = 3) -> List[Dict[str, str]]:
        """Lấy n lượt hội thoại gần nhất."""
        return self.history[-(n_turns * 2):]

    def get_history_text(self, n_turns: int = 3) -> str:
        """Lịch sử hội thoại dạng text để đưa vào prompt."""
        recent = self.get_recent_history(n_turns)
        lines  = []
        for msg in recent:
            role   = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    # ── Stage helpers ─────────────────────────────────────────────────────────

    def update_stage(self, intent: str) -> None:
        """Cập nhật stage dựa theo intent mới nhất."""
        self.last_intent    = intent
        self.intent_history.append(intent)
        self.updated_at     = time.time()

        if intent == "greeting" and self.turn_count == 0:
            self.stage = STAGE_GREETING
        elif intent in ("car_advice", "car_info", "budget_filter",
                        "seat_filter", "usage_filter"):
            if self.has_enough_info():
                self.stage = STAGE_ADVISING
            else:
                self.stage = STAGE_COLLECTING

    def is_expired(self) -> bool:
        return (time.time() - self.updated_at) > SESSION_TTL_SECS

    def summary(self) -> Dict[str, Any]:
        """Tóm tắt state — dùng để debug / logging."""
        return {
            "session_id":    self.session_id,
            "stage":         self.stage,
            "turn_count":    self.turn_count,
            "last_intent":   self.last_intent,
            "filled_slots":  self.get_filled_slots(),
            "missing_slots": self.get_missing_slots(),
            "history_len":   len(self.history),
        }


# ── Manager ───────────────────────────────────────────────────────────────────

class ConversationStateManager:
    """
    Quản lý nhiều session đồng thời (in-memory).

    Dùng trong single-user CLI:
        mgr = ConversationStateManager()
        state = mgr.get_or_create("session-abc")

    Dùng trong multi-user server (FastAPI, v.v.):
        mgr = ConversationStateManager()   # 1 instance toàn cục
        state = mgr.get_or_create(user_id)
    """

    def __init__(self):
        self._sessions: Dict[str, ConversationState] = {}

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(self, session_id: Optional[str] = None) -> ConversationState:
        sid   = session_id or str(uuid.uuid4())
        state = ConversationState(session_id=sid)
        self._sessions[sid] = state
        return state

    def get(self, session_id: str) -> Optional[ConversationState]:
        state = self._sessions.get(session_id)
        if state and state.is_expired():
            self.delete(session_id)
            return None
        return state

    def get_or_create(self, session_id: str) -> ConversationState:
        state = self.get(session_id)
        if state is None:
            state = self.create(session_id)
        return state

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def reset(self, session_id: str) -> ConversationState:
        """Xoá state cũ và tạo mới cho session."""
        self.delete(session_id)
        return self.create(session_id)

    # ── Maintenance ───────────────────────────────────────────────────────────

    def purge_expired(self) -> int:
        """Xoá tất cả session hết hạn. Trả về số session đã xoá."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def active_count(self) -> int:
        return len(self._sessions)

    def all_summaries(self) -> List[Dict]:
        return [s.summary() for s in self._sessions.values()]


# Singleton dùng chung toàn app
state_manager = ConversationStateManager()


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from slot_extractor import extract_slots

    mgr     = ConversationStateManager()
    state   = mgr.create("test-session-001")

    queries = [
        ("Xin chào!", "greeting"),
        ("Tôi muốn mua xe 7 chỗ khoảng 1 tỷ", "car_advice"),
        ("Chủ yếu đi trong thành phố", "usage_filter"),
        ("Tôi thích xe chạy xăng", "car_info"),
    ]

    for user_msg, intent in queries:
        slots = extract_slots(user_msg)
        state.update_slots(slots)
        state.update_stage(intent)
        state.add_turn(user_msg, "[assistant response placeholder]")
        print(f"Turn {state.turn_count}: {state.summary()}\n")