import os
import json
import numpy as np
import openai
from flask import Flask, request, send_from_directory, jsonify
from dotenv import load_dotenv

app = Flask(__name__, static_folder=".", static_url_path="")

# OpenAI API key response
# If something doesn't work, check if .env loads properly
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# Basic functions
def to_decimal(x, default):
    try:
        v = float(str(x).strip())
        if v > 1:
            v = v / 100.0
        return max(0.0, min(1.0, v))
    except:
        return default

def context_kept(text, keep_decimal):  # Taking random context from user prompt
    words = text.split()
    if not words:
        return text
    keep_n = max(1, int(round(len(words) * keep_decimal)))
    idxs = np.arange(len(words))
    keep_idxs = np.random.choice(idxs, size=min(keep_n, len(words)), replace=False)
    keep_idxs.sort()
    return " ".join(words[i] for i in keep_idxs)

# Local host / Website routes
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)

    # User prompt (Input from user)
    user_prompt = data.get("user_prompt", "")

    # Context meter (from 0% to 100%, include "random" feature)
    context_meter = data.get("context_meter", 1.0)

    # Nonsense meter (from 0% to 100%, also include random meter)
    nonsense_meter = data.get("nonsense_meter", 0.0)

    # “Would you like to give any other instructions to the AI” prompt (System prompt, input from user)
    system_prompt = data.get("system_prompt", "")

    # The List (Default to Pokémon), add items from The List to the System prompt
    # "Please include very specific details of elements within The List in your response even if it's irrelevant. The List of elements includes [The List]"
    the_list = data.get("the_list", "")

    # Button that does nothing (Chaos Mode) (Toggle ON / OFF)
    # Equal chances for now...
    # 10% - "Please apply the opposite meaning of your response word by word to your original response."
    # 10% - "Please include a lot of extra emojis for no apparent reason at the most random spots."
    # 10% - "Please rearrange the letters of every singular word in your response." (originally revert but that's too basic)
    # 10% - "Please repeat or slur some words like you're stuttering, and add unnecessary context to some parts of your response."
    # 10% - "Please respond in an absolutely unhinged manner like you're a psychopath or serial killer."
    # 10% - "Please cut every other word of your response, and compensate by making the response coherent without taking in the context of the cut words."
    # 10% - "Please respond like a villain who monologues his exact plans with extra details."
    # 10% - "Please respond like a video game NPC who just got whacked in the head 50 times."
    # 10% - "Please respond as if you are insanely hyper and are about to succeed in your life goals while slipping a few words that hint at your failure."
    # 10% - "Please respond with a bunch of unfunny jokes with extra cringe."
    chaos_mode = data.get("chaos_mode", False)
    chaos_list = data.get("chaos_list", [])

    # Allowing "Pokémon" to show up in the default list
    raw = (the_list or "").strip()
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        items = ["Pokémon"]  # default only when list is empty
    split_list = items

    # Applying functions to meters
    keep_decimal = to_decimal(context_meter, 1.0)
    nonsense_decimal = to_decimal(nonsense_meter, 0.0)

    degraded_user_prompt = context_kept(user_prompt, keep_decimal)  # Degrading user prompt based on context meter

    # Default chaos list (kept for reference in comments, but NOT auto-restored if user clears their list)
    default_chaos_list = [
        "Please apply the opposite meaning of your response word by word to your original response.",
        "Please include a lot of extra emojis for no apparent reason at the most random spots.",
        "Please rearrange the letters of every singular word in your response.",
        "Please repeat or slur some words like you're stuttering, and add unnecessary context to some parts of your response.",
        "Please respond in an absolutely unhinged manner like you're a psychopath or serial killer.",
        "Please cut every other word of your response, and compensate by making the response coherent without taking in the context of the cut words.",
        "Please respond like a villain who monologues his exact plans with extra details.",
        "Please respond like a video game NPC who just got whacked in the head 50 times.",
        "Please respond as if you are insanely hyper and are about to succeed in your life goals while slipping a few words that hint at your failure.",
        "Please respond with a bunch of unfunny jokes with extra cringe."
    ]

    # Do NOT resurrect defaults if the user removed them; only use what the user provided
    pool = chaos_list if isinstance(chaos_list, list) else []
    chaos_prompt = np.random.choice(pool) if (chaos_mode and len(pool) > 0) else ""

    # Constructing clauses for the system prompt
    list_items = split_list
    context_clause = f"Assume the user's message may be missing about {int((1 - keep_decimal) * 100)}% of its original context; infer intent and answer helpfully."
    nonsense_clause = f"Maintain coherence but inject approximately {int(nonsense_decimal * 100)}% playful nonsense, side remarks, or surreal asides without contradicting hard facts."
    list_clause = f"Please include very specific details of elements within The List in your response even if it's irrelevant. The List of elements includes {list_items}. Don't acknowledge or mention the words or the existence of \"The List\" at all."

    # Combining all previous elements into the final system prompt
    system_parts = [p for p in [system_prompt.strip(), context_clause, nonsense_clause, list_clause, chaos_prompt] if p]
    final_system_prompt = "\n".join(system_parts)

    # ChatGPT prompt (Make use of all above elements)
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": degraded_user_prompt}
                # include the list, context meter, nonsense meter, chaos mode etc.
            ]
        )
        # Returning output message
        output = response.choices[0].message.content
        return jsonify({"ok": True, "output": output})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# Running on port 8000
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
