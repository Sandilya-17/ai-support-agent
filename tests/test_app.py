"""REST API tests with a fake agent (no API key or network)."""
import pytest

from agent import AgentConfigError, AgentUpstreamError
from app import create_app
from approvals import ApprovalStore


class FakeAgent:
    def __init__(self, error=None):
        self.error = error

    def chat(self, message, history=None):
        if self.error:
            raise self.error
        return f"echo: {message}", (history or []) + [{"role": "user", "content": message}], []


@pytest.fixture
def store():
    return ApprovalStore()


@pytest.fixture
def client(store):
    app = create_app(agent=FakeAgent(), approvals=store)
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "ok"}
    assert r.headers["X-Request-ID"]


def test_chat_success(client):
    r = client.post("/chat", json={"message": "  hi  "})
    assert r.status_code == 200
    assert r.get_json()["reply"] == "echo: hi"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": 5},
        {"message": "x" * 2001},
        {"message": "hi", "history": "nope"},
        {"message": "hi", "history": [{"role": "system", "content": "x"}]},
        {"message": "hi", "history": [{"role": "user"}]},
        {"message": "hi", "history": [{"role": "user", "content": "x"}] * 41},
        ["not", "an", "object"],
    ],
)
def test_chat_rejects_invalid_payload(client, payload):
    assert client.post("/chat", json=payload).status_code == 400


def test_chat_rejects_non_json_body(client):
    assert client.post("/chat", data="hello", content_type="text/plain").status_code == 400


def test_upstream_error_maps_to_502(store):
    c = create_app(agent=FakeAgent(AgentUpstreamError("down")), approvals=store).test_client()
    assert c.post("/chat", json={"message": "hi"}).status_code == 502


def test_missing_config_maps_to_503(store):
    c = create_app(agent=FakeAgent(AgentConfigError("no key")), approvals=store).test_client()
    assert c.post("/chat", json={"message": "hi"}).status_code == 503


def test_unknown_route_is_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.get_json() == {"error": "Not found."}


# ---- approvals ----

def test_approvals_disabled_without_admin_key(client, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    assert client.get("/approvals").status_code == 503


def test_approvals_reject_wrong_or_missing_key(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    assert client.get("/approvals").status_code == 401
    assert client.get("/approvals", headers={"X-API-Key": "wrong"}).status_code == 401


def test_approve_flow_creates_ticket(client, store, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    h = {"X-API-Key": "secret"}
    item = store.submit(
        "create_support_ticket", {"subject": "S", "description": "D", "priority": "normal"}
    )
    listed = client.get("/approvals", headers=h).get_json()["pending"]
    assert [i["id"] for i in listed] == [item["id"]]

    r = client.post(f"/approvals/{item['id']}/approve", headers=h)
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "approved" and body["result"]["id"].startswith("TCK-")
    assert client.get("/approvals", headers=h).get_json()["pending"] == []


def test_reject_flow_and_double_decision(client, store, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    h = {"X-API-Key": "secret"}
    item = store.submit("create_support_ticket", {"subject": "S", "description": "D"})
    r = client.post(f"/approvals/{item['id']}/reject", headers=h)
    assert r.status_code == 200 and r.get_json()["status"] == "rejected"
    assert client.post(f"/approvals/{item['id']}/approve", headers=h).status_code == 409


def test_unknown_approval_id_is_404(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    r = client.post("/approvals/APR-NOPE/approve", headers={"X-API-Key": "secret"})
    assert r.status_code == 404
