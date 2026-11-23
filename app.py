"""
app.py
------
REST API for the AI support agent.

Endpoints:
  POST /chat                      run one agent turn
  GET  /health                    liveness check
  GET  /approvals                 list pending approvals        (admin)
  POST /approvals/<id>/approve    approve and execute the action (admin)
  POST /approvals/<id>/reject     reject the action              (admin)

Admin endpoints require the X-API-Key header to match ADMIN_API_KEY and are
disabled (503) when ADMIN_API_KEY is not set, so they fail closed.

Run:
  export ANTHROPIC_API_KEY=sk-...   (or use a .env file)
  python app.py
"""

import hmac
import logging
import os
import time
import uuid

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv

from agent import AgentConfigError, AgentUpstreamError
from approvals import ApprovalAlreadyDecided, ApprovalNotFound, ApprovalStore
from tools import create_support_ticket

load_dotenv()

MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_MESSAGES = 40
ALLOWED_ROLES = {"user", "assistant"}

logger = logging.getLogger("support_api")


def validate_chat_payload(data):
    """Return (message, history) or raise ValueError with a client-safe message."""
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Field 'message' is required and must be a non-empty string.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Field 'message' must be at most {MAX_MESSAGE_CHARS} characters.")

    history = data.get("history", [])
    if not isinstance(history, list) or len(history) > MAX_HISTORY_MESSAGES:
        raise ValueError(f"Field 'history' must be a list of at most {MAX_HISTORY_MESSAGES} messages.")
    for item in history:
        if (
            not isinstance(item, dict)
            or item.get("role") not in ALLOWED_ROLES
            or not isinstance(item.get("content"), (str, list))
        ):
            raise ValueError("Each history item needs a 'role' (user/assistant) and 'content'.")
    return message.strip(), history


def create_app(agent=None, approvals: ApprovalStore = None) -> Flask:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = Flask(__name__)
    approvals = approvals if approvals is not None else ApprovalStore()
    state = {"agent": agent}

    def get_agent():
        """Lazy-init so /health works even without an API key set."""
        if state["agent"] is None:
            from agent import SupportAgent  # deferred to delay the API-key check

            state["agent"] = SupportAgent(approvals=approvals)
        return state["agent"]

    @app.before_request
    def _start():
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        g.started = time.perf_counter()

    @app.after_request
    def _finish(response):
        response.headers["X-Request-ID"] = g.get("request_id", "-")
        logger.info(
            "request id=%s method=%s path=%s status=%s duration_ms=%.1f",
            g.get("request_id", "-"),
            request.method,
            request.path,
            response.status_code,
            (time.perf_counter() - g.get("started", time.perf_counter())) * 1000,
        )
        return response

    def _admin_error():
        expected = os.environ.get("ADMIN_API_KEY")
        if not expected:
            return jsonify({"error": "Approval endpoints are disabled: ADMIN_API_KEY is not set."}), 503
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided.encode(), expected.encode()):
            return jsonify({"error": "Unauthorized."}), 401
        return None

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/chat", methods=["POST"])
    def chat():
        try:
            message, history = validate_chat_payload(request.get_json(silent=True))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        try:
            reply, updated_history, tool_calls = get_agent().chat(message, history=history)
        except AgentConfigError as e:
            logger.error("agent_not_configured")
            return jsonify({"error": str(e)}), 503
        except AgentUpstreamError as e:
            return jsonify({"error": str(e)}), 502

        return jsonify({"reply": reply, "history": updated_history, "tool_calls": tool_calls})

    @app.route("/approvals", methods=["GET"])
    def list_approvals():
        denied = _admin_error()
        if denied:
            return denied
        return jsonify({"pending": approvals.list("pending")})

    def _decide(approval_id: str, approve: bool):
        denied = _admin_error()
        if denied:
            return denied
        try:
            item = approvals.decide(
                approval_id,
                approve,
                executor=(lambda tool_input: create_support_ticket(**tool_input)),
            )
        except ApprovalNotFound:
            return jsonify({"error": "Approval not found."}), 404
        except ApprovalAlreadyDecided:
            return jsonify({"error": "Approval was already decided."}), 409
        logger.info("approval_decided id=%s status=%s", approval_id, item["status"])
        return jsonify(item)

    @app.route("/approvals/<approval_id>/approve", methods=["POST"])
    def approve(approval_id):
        return _decide(approval_id, True)

    @app.route("/approvals/<approval_id>/reject", methods=["POST"])
    def reject(approval_id):
        return _decide(approval_id, False)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(Exception)
    def unexpected(e):
        if isinstance(e, HTTPException):
            return jsonify({"error": e.description}), e.code
        logger.exception("unhandled_error id=%s", g.get("request_id", "-"))
        return jsonify({"error": "Internal server error."}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="127.0.0.1" if debug else "0.0.0.0", port=port, debug=debug)
