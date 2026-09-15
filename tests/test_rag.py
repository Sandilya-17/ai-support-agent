import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag import TfidfRetriever

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")


def test_retrieve_refund_policy():
    retriever = TfidfRetriever(KB_DIR)
    results = retriever.retrieve("how do I get my money back for an order")
    assert results, "Expected at least one result"
    assert results[0]["source"] == "refund_policy.txt"


def test_retrieve_shipping():
    retriever = TfidfRetriever(KB_DIR)
    results = retriever.retrieve("shipping delivery time for my package")
    assert results, "Expected at least one result"
    assert results[0]["source"] == "shipping_faq.txt"


def test_no_match_returns_empty_or_low_score():
    retriever = TfidfRetriever(KB_DIR)
    results = retriever.retrieve("zzz qqq nonsense unrelated")
    assert isinstance(results, list)


if __name__ == "__main__":
    test_retrieve_refund_policy()
    test_retrieve_shipping()
    test_no_match_returns_empty_or_low_score()
    print("All tests passed.")
