"""
agent.py
--------
The agentic core: a loop that sends the conversation to Claude, lets the
model decide whether to call a tool, executes that tool, feeds the result
back, and repeats until the model produces a final answer.

This is the same "tool-use loop" pattern used by real agent frameworks
(LangChain agents, CrewAI, custom MCP clients) - kept intentionally
dependency-light here so the mechanics are easy to read end to end.
"""

import os
import json
from anthropic import Anthropic

from rag import TfidfRetriever
from tools import (
    TOOL_SCHEMAS,
    search_knowledge_base,
    create_support_ticket,
    get_order_status,
)

SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce
company. Use the search_knowledge_base tool for any question about policy,
shipping, refunds, or accounts before answering. Use get_order_status when a
customer gives an order ID. Use create_support_ticket when an issue needs
human follow-up. Keep answers short, friendly, and accurate to what the
tools return - never invent policy details."""


class SupportAgent:
    def __init__(self, knowledge_dir: str = "knowledge_base", model: str = None):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.client = Anthropic(api_key=api_key)
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
        self.retriever = TfidfRetriever(knowledge_dir)

    def _dispatch_tool(self, name: str, tool_input: dict):
        """Route a tool_use block to the matching Python function."""
        if name == "search_knowledge_base":
            return search_knowledge_base(self.retriever, **tool_input)
        if name == "create_support_ticket":
            return create_support_ticket(**tool_input)
        if name == "get_order_status":
            return get_order_status(**tool_input)
        return {"error": f"Unknown tool: {name}"}

    def chat(self, user_message: str, history: list = None, max_turns: int = 5):
        """
        Run the agent loop for a single user message.

        `history` is a list of prior {"role": ..., "content": ...} turns so
        the endpoint can support multi-turn conversations.
        Returns (final_text, updated_history, tool_calls_made).
        """
        messages = list(history) if history else []
        messages.append({"role": "user", "content": user_message})

        tool_calls_made = []

        for _ in range(max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                messages.append({"role": "assistant", "content": response.content})
                return final_text, messages, tool_calls_made

            # Model wants to call one or more tools.
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self._dispatch_tool(block.name, block.input)
                tool_calls_made.append({"tool": block.name, "input": block.input, "output": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return (
            "I'm sorry, I wasn't able to resolve this in the allotted steps - "
            "let me escalate this to a human agent.",
            messages,
            tool_calls_made,
        )
