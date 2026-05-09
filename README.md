# SHL Assessment Recommender

A conversational FastAPI agent that takes a hiring manager from a vague intent to a grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue.

Built for the **SHL Labs AI Intern Take-Home Assignment**.

---

## Architecture

```
POST /chat (full history)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 1 – Classifier (Groq LLM)            │
│  Determines: CLARIFY / RECOMMEND /          │
│              COMPARE / REFUSE               │
│  Extracts: query, job_level, type filter    │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│  TF-IDF Retriever (catalog.py)              │
│  377 SHL assessments, bigram index          │
│  Multi-query expansion + level/type boost   │
│  Returns top-10 catalog items               │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│  Stage 2 – Responder (Groq LLM)             │
│  Generates reply text only                  │
│  Recommendations come from retriever        │
│  (never from LLM → no hallucinated URLs)    │
└─────────────────────────────────────────────┘
```

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments suited for a mid-level Java developer…",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Agent Behaviors

| Behavior | Description |
|---|---|
| **Clarify** | Asks focused questions when context is vague |
| **Recommend** | Returns 1–10 catalog assessments once context is sufficient |
| **Compare** | Grounds comparison in catalog data only |
| **Refuse** | Declines off-topic queries (salary, legal, prompt injection) |
| **Turn cap** | Forces final recommendation at turn 8 (per spec) |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# Run locally
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (get one at console.groq.com) |
| `GROQ_MODEL` | Model to use (default: `llama-3.3-70b-versatile`) |
| `PORT` | Port to listen on (set automatically by Railway) |

## Deploy on Railway

1. Connect this repo in Railway
2. Add env vars: `GROQ_API_KEY`, `GROQ_MODEL`
3. Railway auto-detects `railway.toml` and deploys

## Stack

- **FastAPI** — API framework
- **Groq** (`llama-3.3-70b-versatile`) — LLM backbone
- **scikit-learn TF-IDF** — Catalog retrieval with multi-query expansion
- **httpx** — Catalog download
