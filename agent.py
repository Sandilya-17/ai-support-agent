"""
agent.py
--------
The agentic core: a loop that sends the conversation to Claude, lets the
model decide whether to call a tool, executes that tool, feeds the result
back, and repeats until the model produces a final answer.

Safety and reliability features:
  * Human-in-the-loop: side-effecting tools (ticket creation) are queued for
    human approval instead of running immediately (see approvals.py).
  * Tool input from the model is validated; bad input is returned to the model
    as an error result instead of crashing the request.
  * Upstream API failures are wrapped in typed errors the API layer can map.
  * Structured logging of tool calls (names and timings only, never customer text).
"""

import os
import json
import time
import logging

import anthropic
from anthropic import Anthropic

from approvals import ApprovalStore
from rag import TfidfRetriever
from tools import (
    TOOL_SCHEMAS,
    search_knowledge_base,
    create_support_ticket,
    get_order_status,
    validate_ticket_input,
)

logger = logging.getLogger("support_agent")

DEFAULT_KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce
company. Use the search_knowledge_base tool for any question about policy,
shipping, refunds, or accounts before answering. Use get_order_status when a
customer gives an order ID. Use create_support_ticket when an issue needs
human follow-up. If a tool result has status "pending_approval", tell the
customer a human agent will review the request and share the reference ID;
never say a ticket was created until it is approved. Keep answers short,
friendly, and accurate to what the tools return - never invent policy details."""


class AgentConfigError(RuntimeError):
    """The agent is not configured (e.g. missing API key)."""


class AgentUpstreamError(RuntimeError):
    """The model API call failed."""


def _block_to_dict(block) -> dict:
    """Convert an SDK content block into a plain, JSON-serializable dict."""
    if isinstance(block, dict):
        return block
    return block.model_dump(exclude_none=True)


class SupportAgent:
    def __init__(
        self,
        knowledge_dir: str = DEFAULT_KB_DIR,
        model: str = None,
        approvals: ApprovalStore = None,
        client=None,
        require_approval: bool = None,
    ):
        if client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise AgentConfigError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            client = Anthropic(api_key=api_key)
        self.client = client
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
        self.retriever = TfidfRetriever(knowledge_dir)
        self.approvals = approvals if approvals is not None else ApprovalStore()
        if require_approval is None:
            require_approval = os.environ.get("REQUIRE_TICKET_APPROVAL", "true").lower() != "false"
        self.require_approval = require_approval

    def _dispatch_tool(self, name: str, tool_input: dict):
        """Route a tool_use block to the matching Python function.

        Never raises for bad model input: errors are returned as a result so
        the model can correct itself or explain the problem.
        """
        try:
            if name == "search_knowledge_base":
                return search_knowledge_base(self.retriever, **tool_input)
            if name == "get_order_status":
                return get_order_status(**tool_input)
            if name == "create_support_ticket":
                clean = validate_ticket_input(**tool_input)
                if self.require_approval:
                    item = self.approvals.submit(name, clean)
                    return {
                        "status": "pending_approval",
                        "approval_id": item["id"],
                        "message": "Ticket request queued for human approval.",
                    }
                return create_support_ticket(**clean)
            return {"error": f"Unknown tool: {name}"}
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid input for {name}: {e}"}

    def chat(self, user_message: str, history: list = None, max_turns: int = 5):
        """
        Run the agent loop for a single user message.

        `history` is a list of prior {"role": ..., "content": ...} turns so
        the endpoint can support multi-turn conversations.
        Returns (final_text, updated_history, tool_calls_made); the history is
        plain JSON-serializable data.
        """
        messages = list(history) if history else []
        messages.append({"role": "user", "content": user_message})

        tool_calls_made = []

        for turn in range(max_turns):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
            except anthropic.APIError as e:
                logger.error("model_call_failed turn=%d error=%s", turn, type(e).__name__)
                raise AgentUpstreamError("The model service is unavailable.") from e

            content = [_block_to_dict(b) for b in response.content]
            messages.append({"role": "assistant", "content": content})

            if response.stop_reason != "tool_use":
                final_text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
                logger.info("agent_done turns=%d tool_calls=%d", turn + 1, len(tool_calls_made))
                return final_text, messages, tool_calls_made

            # Model wants to call one or more tools.
            tool_results = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue
                started = time.perf_counter()
                result = self._dispatch_tool(block["name"], block["input"])
                logger.info(
                    "tool_call name=%s ok=%s duration_ms=%.1f",
                    block["name"],
                    "error" not in result,
                    (time.perf_counter() - started) * 1000,
                )
                tool_calls_made.append(
                    {"tool": block["name"], "input": block["input"], "output": result}
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        logger.warning("max_turns_reached max_turns=%d", max_turns)
        return (
            "I'm sorry, I wasn't able to resolve this in the allotted steps - "
            "let me escalate this to a human agent.",
            messages,
            tool_calls_made,
        )
