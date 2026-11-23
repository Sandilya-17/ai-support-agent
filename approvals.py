"""
approvals.py
------------
Human-in-the-loop approval queue.

Side-effecting tool calls requested by the model (e.g. creating a support
ticket) are not executed immediately. They are queued here as "pending" and
only run when a human approves them through the /approvals API.

The store is in-memory and guarded by a lock. For multiple server processes,
replace it with a database table; the interface can stay the same.
"""

import threading
import uuid
import datetime

PENDING, APPROVED, REJECTED = "pending", "approved", "rejected"


class ApprovalNotFound(KeyError):
    pass


class ApprovalAlreadyDecided(Exception):
    pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ApprovalStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._items = {}

    def submit(self, tool: str, tool_input: dict) -> dict:
        approval_id = f"APR-{uuid.uuid4().hex[:8].upper()}"
        item = {
            "id": approval_id,
            "tool": tool,
            "input": dict(tool_input),
            "status": PENDING,
            "created_at": _now(),
            "decided_at": None,
            "result": None,
        }
        with self._lock:
            self._items[approval_id] = item
        return dict(item)

    def get(self, approval_id: str) -> dict:
        with self._lock:
            if approval_id not in self._items:
                raise ApprovalNotFound(approval_id)
            return dict(self._items[approval_id])

    def list(self, status: str = PENDING) -> list:
        with self._lock:
            return [dict(i) for i in self._items.values() if i["status"] == status]

    def decide(self, approval_id: str, approve: bool, executor=None) -> dict:
        """Approve (running `executor(tool_input)`) or reject a pending item."""
        with self._lock:
            item = self._items.get(approval_id)
            if item is None:
                raise ApprovalNotFound(approval_id)
            if item["status"] != PENDING:
                raise ApprovalAlreadyDecided(approval_id)
            if approve:
                item["result"] = executor(item["input"]) if executor else None
                item["status"] = APPROVED
            else:
                item["status"] = REJECTED
            item["decided_at"] = _now()
            return dict(item)
