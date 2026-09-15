# AI Support Agent — RAG + Tool-Calling REST API

A small, self-contained AI agent that answers customer-support questions using
**retrieval-augmented generation (RAG)** and can take real actions via
**tool calling** (Anthropic's Messages API tool-use), served over a Flask
**REST API**.

Built to be a compact, readable reference for the core mechanics behind
agentic AI systems: tool-use loops, RAG retrieval, and multi-turn state -
the same primitives used by frameworks like LangChain/LangGraph, CrewAI, and
MCP-based agents.

## Why this project

This started as a quick demo aligned with an AI Agent Developer role
(agentic workflows, RAG assistants, tool/API orchestration, rapid Flask/REST
prototyping) — good scope for a resume project or an interview walkthrough.

## How it works

```
 User message
      │
      ▼
 Flask /chat endpoint
      │
      ▼
 agent.py: tool-use loop (Anthropic Messages API)
      │
      ├─ search_knowledge_base  ──► rag.py (TF-IDF retriever over knowledge_base/*.txt)
      ├─ get_order_status       ──► tools.py (simulated order lookup)
      └─ create_support_ticket  ──► tools.py (simulated ticket creation)
      │
      ▼
 Final answer + full conversation history + list of tool calls made
```

The agent loop (`agent.py`) sends the conversation to Claude with a list of
tool schemas. If Claude decides a tool is needed (e.g. "what's your refund
policy?" → `search_knowledge_base`), the loop executes the matching Python
function, feeds the result back to the model, and repeats until Claude
produces a final natural-language answer. This is the same pattern real
agent frameworks use under the hood.

RAG is implemented with TF-IDF + cosine similarity (`rag.py`) instead of a
hosted embedding/vector-DB service, so the whole project runs offline with
zero external dependencies beyond the LLM call itself. Swapping in a real
vector DB (Chroma, Pinecone, pgvector) means changing `rag.py` only — nothing
else in the agent or API layer needs to change.

## Project structure

```
ai-support-agent/
├── app.py                # Flask REST API (/chat, /health)
├── agent.py               # Tool-use agent loop
├── tools.py                # Tool implementations + Anthropic tool schemas
├── rag.py                  # TF-IDF retriever
├── knowledge_base/         # Sample support docs the agent retrieves from
│   ├── refund_policy.txt
│   ├── shipping_faq.txt
│   └── account_help.txt
├── tests/
│   └── test_rag.py         # Retriever tests (no API key required)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd ai-support-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-...

python app.py
```

## Usage

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want a refund for order ORD1234, is that possible?"}'
```

Example response shape:

```json
{
  "reply": "Yes — refunds are available within 30 days if the item is unused ...",
  "history": [ ... ],
  "tool_calls": [
    {"tool": "search_knowledge_base", "input": {"query": "refund policy"}, "output": {...}}
  ]
}
```

## Running tests

The retriever tests don't need an API key:

```bash
python -m pytest tests/
# or: python tests/test_rag.py
```

## Extending this

- **Add a tool**: write a function in `tools.py`, add its JSON schema to
  `TOOL_SCHEMAS`, and add a branch in `SupportAgent._dispatch_tool`.
- **Swap the retriever**: implement the same `.retrieve(query, top_k)`
  interface backed by a real vector DB.
- **Multi-agent**: `agent.py`'s loop can be composed — e.g. a router agent
  that hands off to specialized sub-agents (billing, shipping, technical)
  each with their own tool set.
- **MCP**: the tool-dispatch pattern here (`name` + JSON `input` → function
  call → result back to the model) is the same shape an MCP client uses when
  calling tools exposed by an MCP server — this project can be extended by
  routing `_dispatch_tool` to call out to an MCP server's tools instead of
  local Python functions.

## Tech stack

Python · Flask · REST API · Anthropic Claude (tool use / agents) ·
scikit-learn (TF-IDF retrieval / RAG) · pytest
