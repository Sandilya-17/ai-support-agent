"""Agent loop tests using a scripted fake model client (no API key or network)."""
import copy
import json
import os

import anthropic
import httpx
import pytest

from agent import AgentUpstreamError, SupportAgent
from approvals import ApprovalStore
from tools import _TICKETS

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")


class FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, **_):
        return dict(self.__dict__)


class FakeResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


def text(t):
    return FakeResponse([FakeBlock(type="text", text=t)], "end_turn")


def tool_use(name, tool_input, block_id="toolu_1"):
    return FakeResponse(
        [FakeBlock(type="tool_use", id=block_id, name=name, input=tool_input)], "tool_use"
    )


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))  # snapshot: the agent mutates messages
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(script, **kw):
    return SupportAgent(knowledge_dir=KB_DIR, client=FakeClient(script), **kw)


def test_selects_knowledge_base_tool_and_returns_answer():
    agent = make_agent(
        [tool_use("search_knowledge_base", {"query": "refund policy"}), text("Refunds within 30 days.")]
    )
    reply, history, calls = agent.chat("Can I get a refund?")
    assert reply == "Refunds within 30 days."
    assert [c["tool"] for c in calls] == ["search_knowledge_base"]
    assert calls[0]["output"]["found"] is True
    json.dumps(history)  # history must be JSON-serializable for the REST API


def test_tool_result_is_sent_back_to_the_model():
    agent = make_agent([tool_use("get_order_status", {"order_id": "ORD1"}), text("Done")])
    agent.chat("Where is ORD1?")
    second_call_messages = agent.client.calls[1]["messages"]
    last = second_call_messages[-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "toolu_1"


def test_ticket_requires_human_approval_by_default():
    store = ApprovalStore()
    before = len(_TICKETS)
    agent = make_agent(
        [
            tool_use("create_support_ticket", {"subject": "Refund", "description": "Wants refund"}),
            text("A human agent will review it."),
        ],
        approvals=store,
    )
    _, _, calls = agent.chat("Please escalate")
    assert calls[0]["output"]["status"] == "pending_approval"
    assert len(_TICKETS) == before  # nothing created yet
    pending = store.list("pending")
    assert len(pending) == 1 and pending[0]["input"]["subject"] == "Refund"


def test_ticket_created_immediately_when_approval_disabled():
    before = len(_TICKETS)
    agent = make_agent(
        [tool_use("create_support_ticket", {"subject": "S", "description": "D"}), text("ok")],
        require_approval=False,
    )
    _, _, calls = agent.chat("escalate")
    assert calls[0]["output"]["id"].startswith("TCK-")
    assert len(_TICKETS) == before + 1


def test_invalid_tool_input_is_returned_to_model_not_raised():
    store = ApprovalStore()
    agent = make_agent(
        [tool_use("create_support_ticket", {"subject": "", "description": "D"}), text("sorry")],
        approvals=store,
    )
    _, _, calls = agent.chat("escalate")
    assert "error" in calls[0]["output"]
    assert store.list("pending") == []


def test_unexpected_tool_arguments_are_handled():
    agent = make_agent([tool_use("get_order_status", {"order": "x"}), text("ok")])
    _, _, calls = agent.chat("status?")
    assert "error" in calls[0]["output"]


def test_unknown_tool_returns_error_result():
    agent = make_agent([tool_use("delete_database", {}), text("I can't do that.")])
    _, _, calls = agent.chat("do it")
    assert calls[0]["output"] == {"error": "Unknown tool: delete_database"}


def test_max_turns_falls_back_to_escalation_message():
    script = [tool_use("get_order_status", {"order_id": "A"}, f"t{i}") for i in range(3)]
    reply, _, calls = make_agent(script).chat("loop", max_turns=3)
    assert "escalate" in reply
    assert len(calls) == 3


def test_upstream_api_error_is_wrapped():
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(AgentUpstreamError):
        make_agent([err]).chat("hello")
