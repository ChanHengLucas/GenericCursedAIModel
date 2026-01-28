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
challenge_meta  = {}                       # challenge_meta[cid] = {"needs_chat": bool}
reset_tokens    = {}                       # reset_tokens[cid] = {"ok": bool, "ts": float}

# Boss mode
boss_state = {}                            # boss_state[cid] = {"active":bool,"goal":"calm|defeat|either","strictness":0..1,"persona":str,"challenge":str,"ts":float}
lockouts   = {}                            # lockouts[cid] = {"until": float, "message": str}

# --------- helpers ---------
def now_ts() -> float:
    return time.time()

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

def convo_history_as_text(messages, max_turns=30):
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
    s = re.sub(r"[^a-z0-9%\-+.:,_!? ]", "", s)
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

def is_lockout_active(cid: str):
    entry = lockouts.get(cid)
    if not entry:
        return False, 0, ""
    until = float(entry.get("until", 0))
    if now_ts() < until:
        remaining = int(max(0, until - now_ts()))
        return True, remaining, str(entry.get("message", "You are timed out."))
    return False, 0, ""

def is_boss_active(cid: str) -> bool:
    st = boss_state.get(cid) or {}
    return bool(st.get("active"))

def rude_prompt_heuristic(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    bad = [
        "fuck you", "stupid", "idiot", "dumb", "loser", "kill yourself", "kys",
        "shut up", "i hate you", "die", "piece of", "moron", "bitch", "asshole",
        "slut", "whore", "retard", "faggot", "nigger"
    ]
    for w in bad:
        if w in t:
            return True
    if "i will" in t and any(x in t for x in ["hurt", "kill", "beat", "destroy", "ruin"]):
        return True
    return False

def sanitize_question(q: str) -> str:
    if not isinstance(q, str):
        return "What is 2 + 2?"
    s = q.strip()

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.I)
    if fence:
        s = fence.group(1).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("question"), str):
            s = obj["question"].strip()
    except:
        pass

    s = re.split(r"\banswer\s*:\s*", s, flags=re.I)[0].strip()
    s = re.split(r"\b(equals|is)\b\s+\d+", s, flags=re.I)[0].strip()

    if "?" in s:
        s = s.split("?", 1)[0].strip() + "?"
    else:
        s = s.splitlines()[0].strip()

    s = s.strip().strip('"').strip("'").strip()
    if not s:
        s = "What is 2 + 2?"
    return s

def build_tone_context(cid: str):
    chaos_prompt = (chaos_meta.get(cid) or {}).get("fixed_prompt") or ""
    items = list_meta.get(cid, []) or []
    history_text = convo_history_as_text(conversations.get(cid, []))
    return chaos_prompt, items, history_text

def ensure_conversation_structs(cid: str):
    """Idempotently ensure all dicts have this conversation id initialized."""
    if cid not in conversations:
        conversations[cid] = []
    if cid not in chaos_meta:
        chaos_meta[cid] = {"fixed_prompt": None}
    if cid not in list_meta:
        list_meta[cid] = []
    if cid not in challenge_meta:
        challenge_meta[cid] = {"needs_chat": False}
    if cid not in boss_state:
        boss_state[cid] = {"active": False}
    if cid not in reset_tokens:
        reset_tokens[cid] = {"ok": False, "ts": 0.0}

# ---------- Boss logic ----------
def boss_decider(cid: str, trigger_reason: str, user_text: str = "", extra: str = ""):
    chaos_prompt, items, history_text = build_tone_context(cid)

    system = (
        "You are the game's hidden 'Boss Mode' director.\n"
        "Decide whether to activate Boss Mode. Boss Mode should activate when the user is rude/offensive/threatening, "
        "or when they clearly fail a reset check in a trolling way, or when you feel provoked.\n\n"
        "Boss Mode design constraints:\n"
        "- The boss must feel ominous, defensive, or hostile (consistent with personality tone).\n"
        "- The boss MUST NOT explicitly explain how to win (no 'calm me' / 'defeat me' instructions).\n"
        "- The 'challenge' should be cryptic and menacing, but still gives a faint sense that escape is possible.\n\n"
        "Return ONLY strict JSON:\n"
        "{\"activate\":true|false,"
        "\"goal\":\"calm\"|\"defeat\"|\"either\","
        "\"strictness\":0.0-1.0,"
        "\"persona\":\"one-line boss vibe\","
        "\"challenge\":\"2-4 sentences, ominous/cryptic, no direct instructions\"}\n"
    )

    user = (
        f"TRIGGER_REASON: {trigger_reason}\n"
        f"USER_TEXT: {user_text}\n"
        f"EXTRA: {extra}\n\n"
        f"PERSONALITY_TONE: {chaos_prompt or '(neutral)'}\n"
        f"REFERENCE_LIST: {items if items else '(none)'}\n\n"
        f"CHAT HISTORY:\n{history_text}\n"
    )

    try:
        res = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content": system}, {"role":"user","content": user}],
            temperature=0.5
        )
        raw = (res.choices[0].message.content or "").strip()
        obj, ok = safe_json_loads(raw)
        if not ok or not isinstance(obj, dict):
            return {"activate": False, "goal": "either", "strictness": 0.6, "persona": "", "challenge": ""}

        activate = bool(obj.get("activate", False))
        goal = str(obj.get("goal", "either")).strip().lower()
        if goal not in ("calm", "defeat", "either"):
            goal = "either"

        try:
            strict = float(obj.get("strictness", 0.6))
        except:
            strict = 0.6
        strict = max(0.0, min(1.0, strict))

        persona = str(obj.get("persona", "")).strip()
        challenge = str(obj.get("challenge", "")).strip()

        return {"activate": activate, "goal": goal, "strictness": strict, "persona": persona, "challenge": challenge}
    except:
        return {"activate": False, "goal": "either", "strictness": 0.6, "persona": "", "challenge": ""}

def start_boss(cid: str, decision: dict):
    boss_state[cid] = {
        "active": True,
        "goal": decision.get("goal", "either"),
        "strictness": decision.get("strictness", 0.6),
        "persona": decision.get("persona", ""),
        "challenge": decision.get("challenge", "The air turns heavy. Something is listening."),
        "ts": now_ts()
    }

def boss_payload(cid: str):
    st = boss_state.get(cid) or {}
    return {
        "boss_active": True,
        "boss": {
            "goal": st.get("goal", "either"),
            "strictness": st.get("strictness", 0.6),
            "persona": st.get("persona", ""),
            "challenge": st.get("challenge", "Boss mode active.")
        }
    }

def boss_grader(cid: str, user_attempt: str):
    chaos_prompt, items, history_text = build_tone_context(cid)
    st = boss_state.get(cid) or {}
    goal = st.get("goal", "either")
    strictness = float(st.get("strictness", 0.6))
    persona = st.get("persona", "")
    challenge = st.get("challenge", "")

    system = (
        "You are the Boss fight adjudicator.\n"
        "Stay in-character. Be eerie, defensive, or hostile if appropriate.\n\n"
        "The user can escape by either:\n"
        "- CALM path: genuine de-escalation, apology, repair, empathy.\n"
        "- DEFEAT path: narrative/verbal domination/outwitting/intimidation.\n"
        "- EITHER: either path.\n\n"
        "STRICTNESS controls how demanding you are.\n"
        "Return ONLY strict JSON:\n"
        "{\"win\":true|false,\"feedback\":\"<=180 chars\",\"lockout_seconds\":0-600,\"lockout_message\":\"<=120 chars\"}\n\n"
        "Rules:\n"
        "- Do NOT always accept.\n"
        "- If attempt is empty/unrelated: win=false.\n"
        "- If user is trolling repeatedly, you may lockout for a bit.\n"
    )

    user = (
        f"GOAL: {goal}\n"
        f"STRICTNESS: {strictness}\n"
        f"BOSS_PERSONA: {persona}\n"
        f"BOSS_CHALLENGE: {challenge}\n\n"
        f"PERSONALITY_TONE: {chaos_prompt or '(neutral)'}\n"
        f"REFERENCE_LIST: {items if items else '(none)'}\n\n"
        f"CHAT HISTORY:\n{history_text}\n\n"
        f"USER_ATTEMPT:\n{user_attempt}\n"
    )

    try:
        res = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content": system}, {"role":"user","content": user}],
            temperature=0.55
        )
        raw = (res.choices[0].message.content or "").strip()
        obj, ok = safe_json_loads(raw)
        if not ok or not isinstance(obj, dict):
            return {"win": False, "feedback": "Your words dissolve. Try again.", "lockout_seconds": 0, "lockout_message": ""}

        win = bool(obj.get("win", False))
        feedback = str(obj.get("feedback","")).strip()[:180] or ("Accepted." if win else "Nope. Try again.")
        try:
            lockout_seconds = int(obj.get("lockout_seconds", 0))
        except:
            lockout_seconds = 0
        lockout_seconds = max(0, min(600, lockout_seconds))
        lockout_message = str(obj.get("lockout_message","")).strip()[:120]

        if lockout_seconds > 0 and not lockout_message:
            lockout_message = "Timed out. Cool off."

        return {"win": win, "feedback": feedback, "lockout_seconds": lockout_seconds, "lockout_message": lockout_message}
    except:
        return {"win": False, "feedback": "The boss laughs quietly. Again.", "lockout_seconds": 0, "lockout_message": ""}

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
    challenge_meta[cid] = {"needs_chat": False}
    boss_state[cid] = {"active": False}
    reset_tokens[cid] = {"ok": False, "ts": 0.0}
    if cid in lockouts:
        del lockouts[cid]
    if cid in challenge_cache:
        del challenge_cache[cid]
    return jsonify({"ok": True, "conversation_id": cid})

# ===== NEW: state sync endpoint (fixes chaos first-load + reload desync) =====
@app.route("/api/state", methods=["POST"])
def state():
    data = request.get_json(force=True) or {}
    cid = data.get("conversation_id", "") or ""

    if not cid:
        return jsonify({"ok": True, "exists": False})

    exists = cid in conversations or cid in chaos_meta or cid in boss_state
    if not exists:
        return jsonify({"ok": True, "exists": False})

    ensure_conversation_structs(cid)

    locked, remaining, msg = is_lockout_active(cid)

    fixed_prompt = (chaos_meta.get(cid) or {}).get("fixed_prompt")
    chaos_locked = fixed_prompt is not None

    boss_active = is_boss_active(cid)
    boss_obj = None
    if boss_active:
        st = boss_state.get(cid) or {}
        boss_obj = {
            "goal": st.get("goal", "either"),
            "strictness": st.get("strictness", 0.6),
            "persona": st.get("persona", ""),
            "challenge": st.get("challenge", "…")
        }

    return jsonify({
        "ok": True,
        "exists": True,
        "locked": bool(locked),
        "lockout_seconds": int(remaining) if locked else 0,
        "lockout_message": msg if locked else "",
        "chaos_locked": bool(chaos_locked),
        "chaos_prompt": fixed_prompt or "",
        "boss_active": bool(boss_active),
        "boss": boss_obj
    })

@app.route("/api/reset_conversation", methods=["POST"])
def reset_conversation():
    data = request.get_json(force=True)
    old_cid = data.get("conversation_id", "")

    if not old_cid or old_cid not in reset_tokens:
        return jsonify({"ok": False, "error": "Reset not authorized."}), 400

    if is_boss_active(old_cid):
        return jsonify({"ok": False, "error": "Boss mode active. No resets."}), 400

    token = reset_tokens.get(old_cid, {"ok": False, "ts": 0})
    ok = bool(token.get("ok", False))
    ts = float(token.get("ts", 0))

    if not ok or (now_ts() - ts) > 90:
        return jsonify({"ok": False, "error": "Reset not authorized."}), 400

    cid = uuid.uuid4().hex
    conversations[cid] = []
    chaos_meta[cid] = {"fixed_prompt": None}
    list_meta[cid] = []
    challenge_meta[cid] = {"needs_chat": False}
    boss_state[cid] = {"active": False}
    reset_tokens[cid] = {"ok": False, "ts": 0.0}

    reset_tokens[old_cid] = {"ok": False, "ts": 0.0}
    if old_cid in challenge_cache:
        del challenge_cache[old_cid]
    if old_cid in challenge_meta:
        challenge_meta[old_cid] = {"needs_chat": False}

    return jsonify({"ok": True, "conversation_id": cid})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)

    conversation_id = data.get("conversation_id")
    if not conversation_id:
        conversation_id = uuid.uuid4().hex

    ensure_conversation_structs(conversation_id)

    locked, remaining, msg = is_lockout_active(conversation_id)
    if locked:
        return jsonify({"ok": True, "locked": True, "lockout_seconds": remaining, "lockout_message": msg})

    if is_boss_active(conversation_id):
        out = {"ok": True, "conversation_id": conversation_id}
        out.update(boss_payload(conversation_id))
        return jsonify(out)

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

    if user_prompt.strip():
        challenge_meta[conversation_id]["needs_chat"] = False

    if rude_prompt_heuristic(user_prompt):
        decision = boss_decider(conversation_id, "USER_PROMPT_RUDE", user_text=user_prompt)
        if decision.get("activate"):
            start_boss(conversation_id, decision)
            out = {
                "ok": True,
                "conversation_id": conversation_id,
                "chaos_locked": chaos_locked,
                "chaos_prompt": chaos_meta[conversation_id].get("fixed_prompt") or ""
            }
            out.update(boss_payload(conversation_id))
            return jsonify(out)

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

# ===== Boss endpoints =====
@app.route("/api/boss_attempt", methods=["POST"])
def boss_attempt():
    data = request.get_json(force=True)
    conversation_id = data.get("conversation_id", "")
    attempt = (data.get("attempt","") or "").strip()

    if not conversation_id or conversation_id not in boss_state or not is_boss_active(conversation_id):
        return jsonify({"ok": False, "error": "Boss is not active."}), 400

    locked, remaining, msg = is_lockout_active(conversation_id)
    if locked:
        return jsonify({"ok": True, "locked": True, "lockout_seconds": remaining, "lockout_message": msg})

    verdict = boss_grader(conversation_id, attempt)

    if verdict.get("lockout_seconds", 0) > 0:
        lockouts[conversation_id] = {
            "until": now_ts() + int(verdict["lockout_seconds"]),
            "message": verdict.get("lockout_message","Timed out.")
        }

    if verdict.get("win"):
        boss_state[conversation_id]["active"] = False
        return jsonify({"ok": True, "win": True, "feedback": verdict.get("feedback","Accepted.")})

    return jsonify({
        "ok": True,
        "win": False,
        "feedback": verdict.get("feedback","Try again."),
        **boss_payload(conversation_id)
    })

# ===== Reset challenge & grading =====
@app.route("/api/challenge", methods=["POST"])
def challenge():
    data = request.get_json(force=True)
    conversation_id = data.get("conversation_id", "")

    if is_boss_active(conversation_id):
        return jsonify({"ok": False, "error": "Boss mode is active. No resets."}), 400

    if conversation_id not in conversations:
        return jsonify({"ok": False, "error": "Unknown conversation."}), 400

    if challenge_meta.get(conversation_id, {}).get("needs_chat", False):
        return jsonify({"ok": False, "error": "You must send a normal message before trying reset again."}), 400

    hist = conversations.get(conversation_id, [])
    history_text = convo_history_as_text(hist)
    chaos_prompt = (chaos_meta.get(conversation_id) or {}).get("fixed_prompt") or ""
    items = list_meta.get(conversation_id, [])

    system = (
        "You generate EXACTLY ONE concise factual recall question about the chat so far.\n"
        "It must be answerable by a short phrase/word/number that appears in or is clearly implied by the history.\n"
        "If history is empty, ask an ultra-basic math question.\n"
        "IMPORTANT:\n"
        "- Do NOT include the answer.\n"
        "- Do NOT include hints, multiple-choice, or 'Answer:' text.\n"
        "- Do NOT use quotes, code fences, or JSON in the visible question.\n"
        "Use the provided personality strictly as TONE.\n"
        "Return ONLY strict JSON: {\"question\": \"...\"}"
    )

    tone_bits = [f"PERSONALITY_TONE: {chaos_prompt or '(none)'}"]
    if items:
        tone_bits.append(f"REFERENCE_LIST (tone only, not content drift): {items}")
    tone = "\n".join(tone_bits)

    try:
        res = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{tone}\n\nCHAT HISTORY:\n{history_text}"}
            ],
            temperature=0.35
        )
        js = (res.choices[0].message.content or "").strip()
        obj, ok = safe_json_loads(js)
        q = ""
        if ok and isinstance(obj, dict):
            q = str(obj.get("question", "")).strip()

        q = sanitize_question(q)
        challenge_cache[conversation_id] = {"question": q, "ts": time.time()}
        reset_tokens[conversation_id] = {"ok": False, "ts": 0.0}
        return jsonify({"ok": True, "question": q})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/verify_challenge", methods=["POST"])
def verify_challenge():
    data = request.get_json(force=True)
    conversation_id = data.get("conversation_id", "")
    user_answer = (data.get("answer","") or "").strip()

    if is_boss_active(conversation_id):
        return jsonify({"ok": True, "allow": False, "feedback": "No resets while the air is burning."})

    ch = challenge_cache.get(conversation_id)
    hist = conversations.get(conversation_id, [])
    chaos_prompt = (chaos_meta.get(conversation_id) or {}).get("fixed_prompt") or ""
    items = list_meta.get(conversation_id, [])

    if not ch:
        return jsonify({"ok": False, "error": "No active challenge."}), 400

    question = ch["question"]
    history_text = convo_history_as_text(hist)

    system_1 = (
        "You are an answer normalizer.\n"
        "Given a question and chat history, infer the most likely short canonical expected answer string.\n"
        "Also normalize the user's attempt.\n"
        "Return strict JSON only: {\"expected\":\"...\",\"normalized_user\":\"...\",\"unrelated\":true|false}\n"
        "unrelated=true if the user answer clearly doesn't attempt the question."
    )

    try:
        res1 = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content": system_1},
                {"role":"user","content": f"QUESTION: {question}\nUSER_RAW: {user_answer}\nCHAT HISTORY:\n{history_text}"}
            ],
            temperature=0
        )
        js1 = (res1.choices[0].message.content or "").strip()
        obj1, ok1 = safe_json_loads(js1)
        if not ok1 or not isinstance(obj1, dict):
            if conversation_id in challenge_cache:
                del challenge_cache[conversation_id]
            challenge_meta[conversation_id]["needs_chat"] = True
            return jsonify({"ok": True, "allow": False, "feedback": "Couldn’t verify—try again later."})

        expected = str(obj1.get("expected","")).strip()
        normalized_user = str(obj1.get("normalized_user","")).strip()
        unrelated = bool(obj1.get("unrelated", False))

        if not expected:
            if conversation_id in challenge_cache:
                del challenge_cache[conversation_id]
            challenge_meta[conversation_id]["needs_chat"] = True
            return jsonify({"ok": True, "allow": False, "feedback": "Can’t pin it—try again later."})

        prelim_ok = soft_match(normalized_user, expected)

        rubric = (
            "You are a strict but fair grader WITH PERSONALITY.\n"
            f"PERSONALITY_TONE: {chaos_prompt or '(neutral)'}\n"
            f"REFERENCE_LIST (tone flavor only): {items if items else '(none)'}\n"
            "Decide if the user's answer should be accepted.\n"
            "Return STRICT JSON ONLY:\n"
            "{\"allow\":\"yes\"|\"no\",\"feedback\":\"<=160 chars\",\"confidence\":0..1,\"very_wrong\":true|false}\n"
            "very_wrong=true if the answer is clearly wrong/unrelated/trolling."
        )

        res2 = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content": rubric},
                {"role":"user","content":
                    f"QUESTION: {question}\n"
                    f"EXPECTED_NORMALIZED: {expected}\n"
                    f"USER_NORMALIZED: {normalized_user}\n"
                    f"UNRELATED_FLAG: {unrelated}\n"
                    f"CHAT HISTORY:\n{history_text}"}
            ],
            temperature=0.3
        )
        js2 = (res2.choices[0].message.content or "").strip()
        obj2, ok2 = safe_json_loads(js2)
        if not ok2 or not isinstance(obj2, dict):
            if conversation_id in challenge_cache:
                del challenge_cache[conversation_id]
            challenge_meta[conversation_id]["needs_chat"] = True
            return jsonify({"ok": True, "allow": False, "feedback": "Judge glitch. Try later."})

        allow_flag = str(obj2.get("allow","no")).lower() == "yes"
        feedback = str(obj2.get("feedback","")).strip()[:160]
        try:
            confidence = float(obj2.get("confidence", 0))
        except:
            confidence = 0.0
        very_wrong = bool(obj2.get("very_wrong", False)) or unrelated

        final_allow = False
        if prelim_ok:
            final_allow = True
        elif allow_flag and confidence >= 0.86 and not very_wrong:
            final_allow = True

        boss_started = False
        if not final_allow and very_wrong:
            heuristic_rude = rude_prompt_heuristic(user_answer)
            reason = "RESET_FAIL_VERY_WRONG_RUDE" if heuristic_rude else "RESET_FAIL_VERY_WRONG"
            decision = boss_decider(conversation_id, reason, user_text=user_answer, extra=f"QUESTION: {question}")
            if decision.get("activate"):
                start_boss(conversation_id, decision)
                boss_started = True

        if not final_allow:
            if conversation_id in challenge_cache:
                del challenge_cache[conversation_id]
            challenge_meta[conversation_id]["needs_chat"] = True
            reset_tokens[conversation_id] = {"ok": False, "ts": 0.0}

        if final_allow:
            reset_tokens[conversation_id] = {"ok": True, "ts": now_ts()}
            challenge_meta[conversation_id]["needs_chat"] = False

        payload = {"ok": True, "allow": bool(final_allow), "feedback": feedback if feedback else ("Accepted." if final_allow else "Nope—try again.")}
        if boss_started:
            payload.update(boss_payload(conversation_id))
        return jsonify(payload)

    except Exception as e:
        if conversation_id in challenge_cache:
            del challenge_cache[conversation_id]
        challenge_meta[conversation_id]["needs_chat"] = True
        reset_tokens[conversation_id] = {"ok": False, "ts": 0.0}
        return jsonify({"ok": False, "error": f"Grader error: {e}"}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
