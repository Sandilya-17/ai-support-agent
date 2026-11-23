import pytest

from tools import (
    _TICKETS,
    create_support_ticket,
    get_order_status,
    validate_ticket_input,
)


def test_create_ticket_returns_record():
    ticket = create_support_ticket("Late order", "Package has not arrived", "high")
    assert ticket["id"].startswith("TCK-")
    assert ticket["status"] == "open"
    assert ticket["priority"] == "high"
    assert ticket["id"] in _TICKETS


@pytest.mark.parametrize(
    "subject,description,priority",
    [
        ("", "details", "normal"),
        ("   ", "details", "normal"),
        ("subject", "", "normal"),
        ("subject", "details", "critical"),
        (123, "details", "normal"),
        ("x" * 121, "details", "normal"),
        ("subject", "y" * 2001, "normal"),
    ],
)
def test_ticket_validation_rejects_bad_input(subject, description, priority):
    with pytest.raises(ValueError):
        validate_ticket_input(subject, description, priority)


def test_order_status_is_deterministic():
    assert get_order_status("ORD1234") == get_order_status("ORD1234")


def test_order_status_rejects_empty_and_non_string():
    with pytest.raises(ValueError):
        get_order_status("")
    with pytest.raises(ValueError):
        get_order_status(42)
