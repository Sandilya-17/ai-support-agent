"""
tools.py
--------
Defines the tools (functions) the agent can call, plus their JSON schemas
in the format expected by the Anthropic Messages API tool-use interface.

In a real deployment these would call out to actual systems (a ticketing
API, an order database, a CRM). Here they are simulated so the project
runs standalone, but the schema/dispatch pattern is production-shaped -
swap the function body for a real API/DB call and nothing else changes.
"""

import uuid
import datetime

# In-memory "database" of tickets created during this process's lifetime.
_TICKETS = {}


def search_knowledge_base(retriever, query: str, top_k: int = 2):
    """Retrieve relevant support-doc snippets for a query (RAG tool)."""
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return {"found": False, "results": []}
    return {"found": True, "results": results}


def create_support_ticket(subject: str, description: str, priority: str = "normal"):
    """Create a support ticket and return its ID (simulated CRM/ticket tool)."""
    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    _TICKETS[ticket_id] = {
        "id": ticket_id,
        "subject": subject,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    return _TICKETS[ticket_id]


def get_order_status(order_id: str):
    """Look up an order's status (simulated order-management tool)."""
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
            "Create a support ticket for issues that cannot be resolved "
            "immediately (e.g. escalations, refund approvals, complaints)."
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
                    "enum": ["low", "normal", "high", "urgent"],
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
