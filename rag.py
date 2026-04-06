"""
rag.py  —  RAG pipeline đầy đủ
Flow:
    query
      ↓ Intent Classifier
      ↓ Business Rules
      ↓ Slot Extractor  (cập nhật state)
      ↓ Conversation State Manager
      ↓ Smart Car Consultant  (quyết định prompt + skip_rag)
      ↓ [RAG Retrieve nếu cần]
      ↓ LLM Generate  (Groq → fallback Gemini)
      ↓ Compose final response
"""

import os
from typing import List, Dict, Any, Optional

from groq import Groq
from google import genai
from google.genai import types

from embed import embed_query
from vector_database import search
from intent_classifier import classify_intent
from business_rules import rules_engine
from slot_extractor import extract_slots
from conversation_state_manager import state_manager, ConversationState
from logic_smart_car_consultant import smart_consultant

from dotenv import load_dotenv
load_dotenv()

# ── Cấu hình ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",  "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GROQ_API_KEY:
    raise EnvironmentError("❌ Thiếu GROQ_API_KEY trong .env")
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ Thiếu GEMINI_API_KEY trong .env")

groq_client  = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL   = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-2.0-flash"
TOP_K        = 5

SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn mua xe chuyên nghiệp của TOYOTA Việt Nam.

Nhiệm vụ:
- Tư vấn xe Toyota phù hợp với nhu cầu người dùng
- Trả lời dựa trên ngữ cảnh tài liệu và thông tin khách hàng đã cung cấp
- Không bịa giá, thông số, phiên bản
- Chỉ tư vấn xe có trong tài liệu

Khi tư vấn, trình bày theo cấu trúc:
1. Nhận định nhu cầu (dựa trên thông tin đã thu thập)
2. Gợi ý xe phù hợp
3. Lý do chọn
4. Gợi ý thêm nếu có

Luôn trả lời bằng tiếng Việt. Giọng văn: chuyên nghiệp, thân thiện.
""".strip()


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _ask_groq(messages: List[Dict]) -> str:
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def _ask_gemini(messages: List[Dict]) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages if m["role"] != "system"
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
    )
    return resp.text


def _generate(messages: List[Dict]) -> tuple[str, str]:
    """Gọi Groq, fallback Gemini. Trả về (text, model_used)."""
    try:
        return _ask_groq(messages), GROQ_MODEL
    except Exception as e:
        print(f"⚠️  Groq lỗi ({e}), fallback Gemini...")
        return _ask_gemini(messages), GEMINI_MODEL


def _build_context(retrieved: List[Dict[str, Any]]) -> str:
    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(
            f"[{i}] (Trang {r['page']}, score: {r['score']:.2f})\n{r['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Pipeline chính ────────────────────────────────────────────────────────────

def answer(
    query:        str,
    session_id:   str = "default",
    chat_history: Optional[List[Dict]] = None,  # deprecated — giữ tương thích ngược
) -> Dict[str, Any]:
    """
    RAG pipeline đầy đủ với state management.

    Args:
        query       : câu hỏi của người dùng
        session_id  : ID session (mỗi user 1 ID riêng)
        chat_history: (deprecated) không dùng nữa, state quản lý bởi ConversationStateManager

    Returns:
        dict: answer, sources, model_used, intent,
              rule_triggered, stage, slots, session_id
    """

    # ── Lấy / tạo conversation state ─────────────────────────────────────────
    state: ConversationState = state_manager.get_or_create(session_id)

    # ── Bước 1: Classify intent ───────────────────────────────────────────────
    intent_result = classify_intent(query)
    intent        = intent_result["intent"]
    print(f"🔍 Intent: {intent} | confidence: {intent_result.get('confidence', '?')}")

    # ── Bước 2: Business rules ────────────────────────────────────────────────
    blocked, rule_name, rule_response = rules_engine.check(query, intent_result)

    if blocked:
        print(f"🚫 Blocked: {rule_name}")
        state.add_turn(query, rule_response)
        return _make_result(
            answer     = rule_response,
            sources    = [],
            model_used = "business_rules",
            intent     = intent,
            rule_name  = rule_name,
            state      = state,
            session_id = session_id,
        )

    # Rule warning — không chặn, prepend vào response cuối
    warning_prefix = f"{rule_response}\n\n---\n\n" if rule_response else ""

    # ── Bước 3: Slot extraction — cập nhật vào state ─────────────────────────
    new_slots = extract_slots(query)
    state.update_slots(new_slots)
    state.update_stage(intent)
    print(f"📦 Slots filled: {state.get_filled_slots()}")
    print(f"📋 Stage: {state.stage}")

    # ── Bước 4: Smart consultant — sơ bộ quyết định (chưa có RAG context) ────
    prelim_decision = smart_consultant.decide(query, state, rag_context="")

    # ── Bước 5: RAG retrieve ──────────────────────────────────────────────────
    sources     = []
    rag_context = ""

    if not prelim_decision.skip_rag:
        query_vec = embed_query(query)
        retrieved = search(query_vec, top_k=TOP_K, score_threshold=0.35)
        if not retrieved:
            retrieved = search(query_vec, top_k=TOP_K, score_threshold=0.2)

        if retrieved:
            rag_context = _build_context(retrieved)
            sources     = [{"page": r["page"], "score": round(r["score"], 3)} for r in retrieved]
        else:
            # Không có tài liệu liên quan → trả lời ngay với thông tin state
            no_data = (
                warning_prefix
                + "Tôi chưa có thông tin chính xác về nội dung này trong dữ liệu hiện tại."
            )
            final_msg = smart_consultant.compose_final_response(no_data, prelim_decision)
            state.add_turn(query, final_msg)
            return _make_result(
                answer     = final_msg,
                sources    = [],
                model_used = "none",
                intent     = intent,
                rule_name  = rule_name,
                state      = state,
                session_id = session_id,
            )

    # ── Bước 6: Build prompt cuối với RAG context đầy đủ ─────────────────────
    final_decision = smart_consultant.decide(query, state, rag_context=rag_context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state.get_recent_history(n_turns=3),
        {"role": "user", "content": final_decision.prompt},
    ]

    # ── Bước 7: Generate ──────────────────────────────────────────────────────
    llm_response, model_used = _generate(messages)

    final_answer = warning_prefix + smart_consultant.compose_final_response(
        llm_response, final_decision
    )

    # ── Cập nhật state ────────────────────────────────────────────────────────
    state.add_turn(query, final_answer)

    return _make_result(
        answer     = final_answer,
        sources    = sources,
        model_used = model_used,
        intent     = intent,
        rule_name  = rule_name,
        state      = state,
        session_id = session_id,
    )


def _make_result(
    answer:     str,
    sources:    list,
    model_used: str,
    intent:     str,
    rule_name:  str,
    state:      ConversationState,
    session_id: str,
) -> Dict[str, Any]:
    return {
        "answer":          answer,
        "sources":         sources,
        "model_used":      model_used,
        "intent":          intent,
        "rule_triggered":  rule_name,
        "stage":           state.stage,
        "slots":           state.get_filled_slots(),
        "session_id":      session_id,
    }


# ── CLI loop ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uuid
    SESSION_ID = str(uuid.uuid4())
    print(f"🤖 Toyota RAG Chatbot  |  session: {SESSION_ID}\n")
    print("Lệnh đặc biệt: 'reset' | 'state' | 'exit'\n")

    while True:
        query = input("Bạn: ").strip()
        if not query:
            continue

        if query.lower() in ("exit", "quit", "thoát"):
            break

        if query.lower() == "reset":
            state_manager.reset(SESSION_ID)
            print("🔄 Đã reset hội thoại.\n")
            continue

        if query.lower() == "state":
            s = state_manager.get(SESSION_ID)
            print(f"📊 {s.summary() if s else 'Không có state'}\n")
            continue

        result = answer(query, session_id=SESSION_ID)

        print(f"\n🤖 ({result['model_used']}) "
              f"[intent={result['intent']} | stage={result['stage']}]:")
        print(result["answer"])
        print(f"\n📦 Slots    : {result['slots']}")
        print(f"📚 Nguồn    : {result['sources']}")
        print(f"🔒 Rule     : {result['rule_triggered']}\n")
        print("-" * 60)