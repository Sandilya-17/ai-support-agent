"""
app.py
------
REST API for the AI support agent.

Endpoints:
  POST /chat     -> {"message": "...", "history": [...]}  run one agent turn
  GET  /health   -> liveness check

Run:
  export ANTHROPIC_API_KEY=sk-...   (or use a .env file)
  python app.py
"""

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

_agent = None


def get_agent():
    """Lazy-init the agent so /health works even without an API key set."""
    global _agent
    if _agent is None:
        from agent import SupportAgent  # imported here to defer the API-key check
        _agent = SupportAgent()
    return _agent


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Field 'message' is required."}), 400

    try:
        agent = get_agent()
        reply, updated_history, tool_calls = agent.chat(message, history=history)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            "reply": reply,
            "history": updated_history,
            "tool_calls": tool_calls,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
