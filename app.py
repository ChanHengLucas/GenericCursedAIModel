import os
import json
import uuid
import time
import re
import unicodedata
import numpy as np
import openai
from flask import Flask, request, send_from_directory, jsonify
from dotenv import load_dotenv

app = Flask(__name__, static_folder=".", static_url_path="")

# Keys
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# ===== In-memory stores =====
conversations = {}                         # conversations[cid] = [{"role":..., "content":...}, ...]
chaos_meta    = {}                         # chaos_meta[cid]   = {"fixed_prompt": str|None}
list_meta     = {}                         # list_meta[cid]    = ["item1", ...]  (can be [])
challenge_cache = {}                       # challenge_cache[cid] = {"question": str, "ts": float}

# --------- helpers ---------
def to_decimal(x, default):
    try:
        v = float(str(x).strip())
        if v > 1:
            v = v / 100.0
        return max(0.0, min(1.0, v))
    except:
        return default

def context_kept(text, keep_decimal):
    words = text.split()
    if not words:
        return text
    keep_n = max(1, int(round(len(words) * keep_decimal)))
    idxs = np.arange(len(words))
    keep_idxs = np.random.choice(idxs, size=min(keep_n, len(words)), replace=False)
    keep_idxs.sort()
    return " ".join(words[i] for i in keep_idxs)

def convo_history_as_text(messages, max_turns=20):
    out = []
    for m in messages[-max_turns:]:
        role = m.get("role","")
        content = m.get("content","")
        out.append(f"{role.upper()}: {content}")
    return "\n".join(out)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9%\-+.:,_ ]", "", s)
    return s

def soft_match(a: str, b: str) -> bool:
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return False
    return na == nb or (na in nb) or (nb in na)

def safe_json_loads(s: str):
    try:
        return json.loads(s), True
    except:
        return None, False

# --------- routes ---------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/new_conversation", methods=["POST"])
def new_conversation():
    cid = uuid.uuid4().hex
    conversations[cid] = []
    chaos_meta[cid] = {"fixed_prompt": None}
    list_meta[cid] = []
    return jsonify({"ok": True, "conversation_id": cid})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)

    conversation_id = data.get("conversation_id")
    if not conversation_id:
        conversation_id = uuid.uuid4().hex
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    if conversation_id not in chaos_meta:
        chaos_meta[conversation_id] = {"fixed_prompt": None}
    if conversation_id not in list_meta:
        list_meta[conversation_id] = []

    user_prompt = data.get("user_prompt", "")
    context_meter = data.get("context_meter", 1.0)
    nonsense_meter = data.get("nonsense_meter", 0.0)
    system_prompt_user = data.get("system_prompt", "")

    # The List (NO default)
    items = []
    the_list_items = data.get("the_list_items", None)
    if isinstance(the_list_items, list):
        items = [str(s).strip() for s in the_list_items if str(s).strip()]
    else:
        the_list = data.get("the_list", "")
        raw = (the_list or "").strip()
        items = [s.strip() for s in raw.split(",") if s.strip()]
    list_meta[conversation_id] = items[:]

    # Chaos
    chaos_mode = data.get("chaos_mode", False)
    chaos_list = data.get("chaos_list", [])
    if not isinstance(chaos_list, list):
        chaos_list = []

    keep_decimal = to_decimal(context_meter, 1.0)
    nonsense_decimal = to_decimal(nonsense_meter, 0.0)
    degraded_user_prompt = context_kept(user_prompt, keep_decimal)

    fixed = chaos_meta[conversation_id].get("fixed_prompt")
    chaos_locked = fixed is not None
    if chaos_locked:
        chaos_prompt = fixed
    else:
        chaos_prompt = ""
        if chaos_mode and len(chaos_list) > 0:
            chaos_prompt = str(np.random.choice(chaos_list))
            chaos_meta[conversation_id]["fixed_prompt"] = chaos_prompt
            chaos_locked = True

    context_clause = (
        f"Assume the user's message may be missing about {int((1 - keep_decimal) * 100)}% "
        f"of its original context; infer intent and answer helpfully."
    )
    nonsense_clause = (
        f"Maintain coherence but inject approximately {int(nonsense_decimal * 100)}% playful "
        f"nonsense, side remarks, or surreal asides without contradicting hard facts."
    )
    list_clause = ""
    if items:
        list_clause = (
            "Please include very specific details of elements within The List in your response even if it's irrelevant. "
            f"The List of elements includes {items}. Don't acknowledge or mention the words or the existence of \"The List\" at all."
        )

    parts = [p for p in [system_prompt_user.strip(), context_clause, nonsense_clause, list_clause, chaos_prompt] if p]
    final_system_prompt = "\n".join(parts)

    prior = conversations.get(conversation_id, [])
    messages = [{"role": "system", "content": final_system_prompt}] + prior + [
        {"role": "user", "content": degraded_user_prompt}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        output = response.choices[0].message.content

        conversations[conversation_id].append({"role": "user", "content": user_prompt})
        conversations[conversation_id].append({"role": "assistant", "content": output})

        return jsonify({
            "ok": True,
            "output": output,
            "conversation_id": conversation_id,
            "chaos_locked": chaos_locked,
            "chaos_prompt": chaos_meta[conversation_id].get("fixed_prompt") or ""
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ===== Personality-aware challenge & grading =====
@app.route("/api/challenge", methods=["POST"])
def challenge():
    data = request.get_json(force=True)
    conversation_id = data.get("conversation_id", "")
    hist = conversations.get(conversation_id, [])
    history_text = convo_history_as_text(hist)

    chaos_prompt = (chaos_meta.get(conversation_id) or {}).get("fixed_prompt") or ""
    items = list_meta.get(conversation_id, [])  # can be []

    # Keep question factual/concise, but let tone peek through
    system = (
        "You generate EXACTLY ONE concise factual recall question about the chat so far. "
        "It must be answerable by a short phrase/word/number that appears in or is clearly implied by the history. "
        "If history is empty, ask an ultra-basic math or common knowledge question. "
        "Keep the sentence short and natural; do NOT use quotes or code fences.\n"
        "Use the provided personality strictly as TONE, not content drift.\n"
        "Return ONLY strict JSON: {\"question\": \"...\"}"
    )

    tone_bits = [f"PERSONALITY_TONE: {chaos_prompt or '(none)'}"]
    if items:
        tone_bits.append(f"REFERENCE_LIST (tone only, do NOT mention as 'list'): {items}")
    tone = "\n".join(tone_bits)

    try:
        res = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{tone}\n\nCHAT HISTORY:\n{history_text}"}
            ],
            temperature=0.3
        )
        js = (res.choices[0].message.content or "").strip()
        obj, ok = safe_json_loads(js)
        q = ""
        if ok:
            q = str(obj.get("question", "")).strip()
        if not q:
            q = "What is 2 + 2?"

        challenge_cache[conversation_id] = {"question": q, "ts": time.time()}
        return jsonify({"ok": True, "question": q})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/verify_challenge", methods=["POST"])
def verify_challenge():
    data = request.get_json(force=True)
    conversation_id = data.get("conversation_id", "")
    user_answer = (data.get("answer","") or "").strip()

    ch = challenge_cache.get(conversation_id)
    hist = conversations.get(conversation_id, [])
    chaos_prompt = (chaos_meta.get(conversation_id) or {}).get("fixed_prompt") or ""
    items = list_meta.get(conversation_id, [])

    if not ch:
        return jsonify({"ok": False, "error": "No active challenge."}), 400

    question = ch["question"]
    history_text = convo_history_as_text(hist)

    # ---- Step 1: normalize & produce expected answer (tone-neutral) ----
    system_1 = (
        "You are an answer normalizer. "
        "Given a question and chat history, produce a short canonical 'expected answer' string "
        "that a strict grader would accept, and also normalize the user's attempt.\n"
        "Normalization: lowercase, trim, remove obvious punctuation, map clear synonyms, words to numerals if unambiguous "
        "(e.g., 'twenty four'->'24'), expand common abbreviations (e.g., 'nyc'->'new york city').\n"
        "Return strict JSON only: {\"expected\":\"...\",\"normalized_user\":\"...\"}"
    )

    try:
        res1 = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content": system_1},
                {"role":"user","content": f"QUESTION: {question}\nUSER_RAW: {user_answer}\nCHAT HISTORY (reference):\n{history_text}"}
            ],
            temperature=0
        )
        js1 = (res1.choices[0].message.content or "").strip()
        obj1, ok1 = safe_json_loads(js1)
        if not ok1:
            # graceful tonal fallback
            return jsonify({"ok": True, "allow": False, "feedback": "Couldn’t verify—my brain did a backflip. Try again?"})

        expected = str(obj1.get("expected","")).strip()
        normalized_user = str(obj1.get("normalized_user","")).strip()
        if not expected:
            return jsonify({"ok": True, "allow": False, "feedback": "Hazy vibes—I can’t pin the answer yet. Try once more?"})

        prelim_ok = soft_match(normalized_user, expected)

        # ---- Step 2: personality-aware strict grader ----
        rubric = (
            "You are a strict but fair grader with STYLE. Decide if the user's answer shows knowledge of the asked item.\n"
            f"PERSONALITY_TONE (for FEEDBACK wording only): {chaos_prompt or '(neutral)'}\n"
            f"OPTIONAL_REFERENCES (tone flavor only): {items if items else '(none)'}\n"
            "Rules:\n"
            " - If empty, unrelated, or contradicts meaning → no.\n"
            " - Typos/variants OK if unmistakably the same entity/quantity.\n"
            " - If question says 'one of', any single correct item is OK.\n"
            "Output STRICT JSON ONLY:\n"
            "{\"allow\":\"yes\"|\"no\",\"feedback\":\"<=120 chars, keep the specified tone\",\"confidence\":0..1}"
        )

        res2 = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content": rubric},
                {"role":"user","content":
                    f"QUESTION: {question}\n"
                    f"EXPECTED_NORMALIZED: {expected}\n"
                    f"USER_NORMALIZED: {normalized_user}\n"
                    f"CHAT HISTORY (reference):\n{history_text}"}
            ],
            temperature=0.2
        )
        js2 = (res2.choices[0].message.content or "").strip()
        obj2, ok2 = safe_json_loads(js2)
        if not ok2:
            return jsonify({"ok": True, "allow": False, "feedback": "My judge wig slipped—give it another go."})

        allow_flag = str(obj2.get("allow","no")).lower() == "yes"
        feedback = str(obj2.get("feedback","")).strip()[:120]
        try:
            confidence = float(obj2.get("confidence", 0))
        except:
            confidence = 0.0

        # Blend strictness and fuzzy match
        final_allow = (allow_flag and confidence >= 0.72) or prelim_ok

        # Numeric sanity override: if numbers differ noticeably, be stricter
        if re.search(r"\b(\d+)\b", expected) and not soft_match(expected, normalized_user):
            if not (allow_flag and confidence >= 0.88):
                final_allow = False

        return jsonify({
            "ok": True,
            "allow": bool(final_allow),
            "feedback": feedback if feedback else ("Accepted." if final_allow else "Close, but not quite—try again.")
        })
    except Exception as e:
        # Toneful hard-error message
        return jsonify({"ok": False, "error": f"Grader face-planted: {e}"}), 400

# Running
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
