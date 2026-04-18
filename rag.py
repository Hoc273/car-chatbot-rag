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
      ↓ LLM Generate  (Groq → Ollama → Gemini)
      ↓ Compose final response

LLM_PROVIDER (trong .env):
    groq    → Groq API (default)
    ollama  → Ollama local (gemma3:4b hoặc model bất kỳ)
    gemini  → Google Gemini
    auto    → Groq → fallback Ollama → fallback Gemini
"""

import os
import httpx
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
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Chọn provider: groq | ollama | gemini | auto
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "auto").lower()

GROQ_MODEL     = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/chat")

TOP_K = 5

# Validate API keys theo provider
if LLM_PROVIDER in ("groq", "auto") and not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY chưa set — Groq sẽ không dùng được")
if LLM_PROVIDER in ("gemini",) and not GEMINI_API_KEY:
    raise EnvironmentError("❌ Thiếu GEMINI_API_KEY trong .env")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn mua xe chuyên nghiệp của TOYOTA Việt Nam.

Nhiệm vụ:
- Chỉ tư vấn xe có trong tài liệu
- Tư vấn xe Toyota phù hợp với nhu cầu người dùng
- Trả lời dựa trên ngữ cảnh tài liệu và thông tin khách hàng đã cung cấp
- Không bịa giá, thông số, phiên bản, chỉ dựa trên dữ liệu đã có
- Nếu không đủ thông tin, hãy hỏi thêm người dùng (ưu tiên slot quan trọng trước)


Khi tư vấn, trình bày theo cấu trúc:
1. Nhận định nhu cầu (dựa trên thông tin đã thu thập)
2. Gợi ý xe phù hợp
3. Lý do chọn
4. Gợi ý thêm nếu có

Luôn trả lời bằng tiếng Việt. Giọng văn: chuyên nghiệp, thân thiện.
""".strip()


def _ollama_base_url() -> str:
    base_url = OLLAMA_URL.rstrip("/")
    for suffix in ("/api/chat", "/api/generate"):
        if base_url.endswith(suffix):
            return base_url[: -len(suffix)]
    return base_url


def _resolve_ollama_model(preferred_model: str) -> str:
    """Return the preferred Ollama model if installed, otherwise a local fallback."""
    try:
        resp = httpx.get(f"{_ollama_base_url()}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return preferred_model

    models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
    if not models:
        return preferred_model

    if preferred_model in models:
        return preferred_model

    fallback_model = models[0]
    print(f"⚠️ Ollama model '{preferred_model}' not found, using '{fallback_model}' instead")
    return fallback_model


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _ask_groq(messages: List[Dict]) -> str:
    if not groq_client:
        raise RuntimeError("Groq client chưa được khởi tạo (thiếu API key)")
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def _ask_ollama(messages: List[Dict]) -> str:
    """
    Gọi Ollama local qua REST API.
    Ollama nhận messages theo chuẩn OpenAI (role + content).
    System prompt được inject vào đầu messages nếu chưa có.
    """
    # Đảm bảo có system message
    if not messages or messages[0]["role"] != "system":
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        full_messages = messages

    model_name = _resolve_ollama_model(OLLAMA_MODEL)

    payload = {
        "model":   model_name,
        "messages": full_messages,
        "stream":  False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
            "num_ctx":     4096,   # giới hạn context để tiết kiệm RAM
        },
    }
    try:
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        status = None
        try:
            status = e.response.status_code if e.response is not None else None
        except Exception:
            status = None
        if status == 404:
            # Try common alternate endpoint (/api/generate)
            alt_url = f"{_ollama_base_url()}/api/generate"
            print(f"⚠️ Ollama endpoint returned 404, retrying {alt_url}")
            resp = httpx.post(alt_url, json=payload, timeout=120)
            resp.raise_for_status()
        else:
            body = ""
            try:
                body = e.response.text.strip() if e.response is not None else ""
            except Exception:
                body = ""
            detail = body or str(e)
            raise RuntimeError(f"Ollama HTTP {status}: {detail}") from e

    # Parse known response shapes: /api/chat -> {"message": {"content": ...}}
    # /api/generate  -> {"text": "..."} or similar. Fall back to raw text.
    j = None
    try:
        j = resp.json()
    except Exception:
        return resp.text

    if isinstance(j, dict):
        if "message" in j and isinstance(j["message"], dict) and "content" in j["message"]:
            return j["message"]["content"]
        if "text" in j:
            return j["text"]

    return resp.text


def _ask_gemini(messages: List[Dict]) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
        if m["role"] != "system"
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
    )
    return resp.text


def _generate(messages: List[Dict]) -> tuple[str, str]:
    """
    Gọi LLM theo LLM_PROVIDER.

    Provider modes:
        groq   → chỉ Groq, không fallback
        ollama → chỉ Ollama local, không fallback
        gemini → chỉ Gemini, không fallback
        auto   → Groq → Ollama → Gemini (fallback theo thứ tự)
    """
    if LLM_PROVIDER == "ollama":
        try:
            return _ask_ollama(messages), OLLAMA_MODEL
        except Exception as e:
            print(f"⚠️  Ollama lỗi ({e}), thử provider dự phòng...")
            if GROQ_API_KEY:
                try:
                    return _ask_groq(messages), GROQ_MODEL
                except Exception as groq_error:
                    print(f"⚠️  Groq lỗi ({groq_error}), thử Gemini...")
            if GEMINI_API_KEY:
                return _ask_gemini(messages), GEMINI_MODEL
            raise RuntimeError(
                "Ollama không khả dụng và không có provider dự phòng hợp lệ (GROQ_API_KEY/GEMINI_API_KEY)."
            ) from e

    if LLM_PROVIDER == "groq":
        return _ask_groq(messages), GROQ_MODEL

    if LLM_PROVIDER == "gemini":
        return _ask_gemini(messages), GEMINI_MODEL

    # auto — fallback chain
    if GROQ_API_KEY:
        try:
            return _ask_groq(messages), GROQ_MODEL
        except Exception as e:
            print(f"⚠️  Groq lỗi ({e}), fallback Ollama...")

    try:
        return _ask_ollama(messages), OLLAMA_MODEL
    except Exception as e:
        print(f"⚠️  Ollama lỗi ({e}), fallback Gemini...")

    return _ask_gemini(messages), GEMINI_MODEL


def _build_context(retrieved: List[Dict[str, Any]]) -> str:
    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(
            f"Nguồn {i}:\n"
            f"Trang: {r['page']} | Độ tin cậy: {r['score']:.2f}\n\n"
            f"{r['content']}"
        )
    return "\n\n==========\n\n".join(parts)


# ── Pipeline chính ────────────────────────────────────────────────────────────

def answer(
    query:        str,
    session_id:   str = "default",
) -> Dict[str, Any]:

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

    warning_prefix = f"{rule_response}\n\n---\n\n" if rule_response else ""

    # ── Bước 3: Slot extraction ───────────────────────────────────────────────
    new_slots = extract_slots(query)
    state.update_slots(new_slots)
    state.update_stage(intent)
    print(f"📦 Slots filled: {state.get_filled_slots()}")
    print(f"📋 Stage: {state.stage}")

    # ── Bước 4: Smart consultant (sơ bộ, chưa có RAG context) ────────────────
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
            no_data   = warning_prefix + "Tôi chưa có thông tin chính xác về nội dung này trong dữ liệu hiện tại."
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
        {
            "role": "user",
            "content": (
                f"Dữ liệu tham khảo:\n{rag_context}\n\n"
                f"---\n\n"
                f"Câu hỏi người dùng:\n{final_decision.prompt}"
            ),
        },
    ]

    # ── Bước 7: Generate ──────────────────────────────────────────────────────
    llm_response, model_used = _generate(messages)

    final_answer = warning_prefix + smart_consultant.compose_final_response(
        llm_response, final_decision
    )

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
    print(f"🤖 Toyota RAG Chatbot  |  session: {SESSION_ID}")
    print(f"🧠 LLM provider: {LLM_PROVIDER.upper()}", end="")
    if LLM_PROVIDER == "ollama":
        print(f"  (model: {OLLAMA_MODEL})")
    elif LLM_PROVIDER == "groq":
        print(f"  (model: {GROQ_MODEL})")
    elif LLM_PROVIDER == "gemini":
        print(f"  (model: {GEMINI_MODEL})")
    else:
        print(f"  (Groq → Ollama:{OLLAMA_MODEL} → Gemini)")
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
        print(f"\n🤖 ({result['model_used']}) [intent={result['intent']} | stage={result['stage']}]:")
        print(result["answer"])
        print(f"\n📦 Slots : {result['slots']}")
        print(f"📚 Nguồn : {result['sources']}")
        print(f"🔒 Rule  : {result['rule_triggered']}\n")
        print("-" * 60)