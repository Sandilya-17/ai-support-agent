"""
tools.py
--------
Defines the tools (functions) the agent can call, plus their JSON schemas
in the format expected by the Anthropic Messages API tool-use interface.

In a real deployment these would call out to actual systems (a ticketing
API, an order database, a CRM). Here they are simulated so the project
runs standalone, but the schema/dispatch pattern is production-shaped -
swap the function body for a real API/DB call and nothing else changes.

All tool input comes from the model, so it is validated before use.
"""

import uuid
import datetime

# In-memory "database" of tickets created during this process's lifetime.
_TICKETS = {}

VALID_PRIORITIES = ("low", "normal", "high", "urgent")
MAX_SUBJECT_CHARS = 120
MAX_DESCRIPTION_CHARS = 2000
MAX_ORDER_ID_CHARS = 40
MAX_QUERY_CHARS = 500


def _require_text(name: str, value, max_len: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string.")
    value = value.strip()
    if not value:
        raise ValueError(f"'{name}' must not be empty.")
    if len(value) > max_len:
        raise ValueError(f"'{name}' must be at most {max_len} characters.")
    return value


def validate_ticket_input(subject, description, priority="normal") -> dict:
    """Return cleaned ticket fields or raise ValueError."""
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"'priority' must be one of {', '.join(VALID_PRIORITIES)}.")
    return {
        "subject": _require_text("subject", subject, MAX_SUBJECT_CHARS),
        "description": _require_text("description", description, MAX_DESCRIPTION_CHARS),
        "priority": priority,
    }


def search_knowledge_base(retriever, query: str, top_k: int = 2):
    """Retrieve relevant support-doc snippets for a query (RAG tool)."""
    query = _require_text("query", query, MAX_QUERY_CHARS)
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return {"found": False, "results": []}
    return {"found": True, "results": results}


def create_support_ticket(subject: str, description: str, priority: str = "normal"):
    """Create a support ticket and return its record (simulated CRM/ticket tool)."""
    fields = validate_ticket_input(subject, description, priority)
    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    _TICKETS[ticket_id] = {
        "id": ticket_id,
        **fields,
        "status": "open",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return _TICKETS[ticket_id]


def get_order_status(order_id: str):
    """Look up an order's status (simulated order-management tool)."""
    order_id = _require_text("order_id", order_id, MAX_ORDER_ID_CHARS)
    # Deterministic fake data keyed off the order id so demos are repeatable.
    fake_statuses = ["Processing", "Shipped", "Out for Delivery", "Delivered"]
    status = fake_statuses[sum(map(ord, order_id)) % len(fake_statuses)]
    return {"order_id": order_id, "status": status}


# Tool schemas in Anthropic Messages API "tools" format.
TOOL_SCHEMAS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the company's internal knowledge base (refund policy, "
            "shipping FAQ, account help) for information relevant to a "
            "customer's question. Always use this before answering policy "
            "or process questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question or topic to search for.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Request a support ticket for issues that cannot be resolved "
            "immediately (e.g. escalations, refund approvals, complaints). "
            "Tickets may need human approval before they are created."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Short ticket title."},
                "description": {
                    "type": "string",
                    "description": "Full details of the customer's issue.",
                },
                "priority": {
                    "type": "string",
                    "enum": list(VALID_PRIORITIES),
                    "description": "Ticket priority.",
                },
            },
            "required": ["subject", "description"],
        },
    },
    {
        "name": "get_order_status",
        "description": "Look up the current shipping/delivery status of an order by its order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The customer's order ID."}
            },
            "required": ["order_id"],
        },
    },
]
