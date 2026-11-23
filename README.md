# AI Support Agent — RAG + Tool-Calling REST API

A small, self-contained AI agent that answers customer-support questions using
**retrieval-augmented generation (RAG)** and can take real actions via
**tool calling** (Anthropic's Messages API tool-use), served over a Flask
**REST API**.

Built to be a compact, readable reference for the core mechanics behind
agentic AI systems: tool-use loops, RAG retrieval, and multi-turn state -
the same primitives used by frameworks like LangChain/LangGraph, CrewAI, and
MCP-based agents.

## Features

- **Tool-use agent loop** over Anthropic's Messages API (knowledge search, order lookup, ticket creation).
- **RAG** with TF-IDF and cosine similarity behind a swappable retriever interface.
- **Human-in-the-loop approvals:** ticket creation is queued and only runs after a human approves it via the API.
- **Validation and safe failure:** request payloads and model tool inputs are validated, upstream errors map to 502/503, and bad tool input is returned to the model as an error instead of crashing.
- **Observability:** structured logs with request IDs, status codes, latencies and per-tool timings (customer text is never logged).
- **Tests and CI:** 40+ tests run without an API key (fake model client); GitHub Actions runs tests and builds the Docker image.

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
      └─ create_support_ticket  ──► approvals.py queue ──► human approves ──► tools.py (simulated ticket)
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
├── app.py                  # Flask REST API (chat, approvals, health), validation, logging
├── agent.py                # Tool-use agent loop with approval gating
├── approvals.py            # Human-approval queue
├── tools.py                # Validated tool implementations + Anthropic tool schemas
├── rag.py                  # TF-IDF retriever
├── knowledge_base/         # Sample support docs the agent retrieves from
├── tests/                  # Retriever, tools, agent loop and API tests (no API key needed)
├── Dockerfile              # Non-root image served by gunicorn
├── .github/workflows/ci.yml
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## Setup

```bash
git clone <your-repo-url>
cd ai-support-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (and ADMIN_API_KEY to use /approvals)

python app.py
```

## Usage

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want a refund for order ORD1234, is that possible?"}'
```

Response shape: `{"reply": "...", "history": [...], "tool_calls": [...]}`.
Send `history` back with the next message for multi-turn conversations.

### API reference

| Endpoint | Auth | Description |
|---|---|---|
| `POST /chat` | none | Run one agent turn. `message` is required (max 2000 chars); `history` is optional (max 40 messages). Invalid input returns 400, model outage 502, missing API key 503. |
| `GET /health` | none | Liveness check. |
| `GET /approvals` | `X-API-Key` | List pending approvals. |
| `POST /approvals/<id>/approve` | `X-API-Key` | Approve and execute the queued action (creates the ticket). |
| `POST /approvals/<id>/reject` | `X-API-Key` | Reject the queued action. |

Approval endpoints are disabled (503) unless `ADMIN_API_KEY` is set, and reject any request with a wrong key (401).

### Human-in-the-loop example

```bash
# 1. The agent decides a ticket is needed; it is queued, not created.
curl -X POST localhost:5000/chat -H "Content-Type: application/json" \
  -d '{"message": "My package arrived broken, please escalate this"}'

# 2. A human reviews the queue and approves (or rejects) it.
curl localhost:5000/approvals -H "X-API-Key: $ADMIN_API_KEY"
curl -X POST localhost:5000/approvals/APR-XXXXXXXX/approve -H "X-API-Key: $ADMIN_API_KEY"
```

Set `REQUIRE_TICKET_APPROVAL=false` to create tickets immediately.

## Running tests

No API key or network is needed (the model client is faked):

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Docker

```bash
docker build -t ai-support-agent .
docker run -p 5000:5000 --env-file .env ai-support-agent
```

The image runs as a non-root user under gunicorn with a single worker, because
the approval queue and ticket store are in memory. To scale out, move both into
a database.

## Logging

Logs go to stdout (`LOG_LEVEL` controls verbosity) and include request IDs
(also returned in the `X-Request-ID` header), status codes, latencies and tool
call timings. Customer messages and tool inputs are not logged.

## Extending this

- **Add a tool**: write a validated function in `tools.py`, add its JSON schema to
  `TOOL_SCHEMAS`, and add a branch in `SupportAgent._dispatch_tool`. Route any
  side-effecting tool through the approval queue.
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
scikit-learn (TF-IDF retrieval / RAG) · pytest · Docker · GitHub Actions
