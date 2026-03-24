# Newscast AI

Newscast AI generates a personalised daily audio news briefing for each user.
It pulls articles from RSS feeds and live SearxNG web searches, scores and
ranks them against per-user topic preferences, summarises the top stories with
a local or API-served LLM, writes a broadcast-style script, and synthesises an
MP3 episode served over a private RSS feed — all on infrastructure you run
yourself.  No cloud AI APIs required; a single consumer GPU is enough.

---

## Architecture

```
[User] → API (FastAPI :8000)
           │
           │ POST /users/{id}/kick  (or APScheduler at schedule_time)
           ▼
       Worker (Celery :8001)
         ├─ ingestion.py   — RSS fetch, SearxNG search, facet scoring
         ├─ agent_search.py — agentic single-article path (breaking news)
         ├─ tasks.py        — orchestration: ingest → summarise → TTS → store
         └─ HTTP → MCP (:7000)
                    ├─ /hostify      — LangGraph script pipeline (plan→draft→validate→compress)
                    ├─ /summarize_batch — BART summarisation
                    └─ /tts             — gTTS / Kokoro synthesis
                                ↓
                         audio/*.mp3  (nginx-served volume)
                                ↓
                       nginx :8080  ← RSS feed + audio delivery
```

**Services:**

| Service | Technology | Port | Role |
|---------|-----------|------|------|
| `api` | FastAPI | 8000 | User preferences, episode retrieval, RSS |
| `worker` | Celery + APScheduler | 8001 | Episode generation orchestration |
| `mcp` | FastAPI | 7000 | LLM inference, TTS, summarisation |
| `nginx` | nginx:alpine | 8080 | Reverse proxy + audio file serving |
| `postgres` | postgres:16 | 5432 | Primary database |
| `redis` | redis:7 | 6379 | Celery broker + result backend |
| `searxng` | searxng/searxng | — | Web search for agentic ingestion |
| `minio` | minio | 9000 | S3-compatible audio object storage |

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2.20
- A `.env` file in the project root (copy `.env.example` and fill in values)

### Environment variables

```bash
cp .env.example .env
```

Minimum required values in `.env`:

```bash
POSTGRES_PASSWORD=changeme
REDIS_URL=redis://redis:6379/0
MCP_URL=http://mcp:7000
SECRET_KEY=<random 32-byte hex>

# LLM — pick one of the three provider modes (see Model Configuration below)
HOSTIFY_PROVIDER=api
OPENAI_API_KEY=sk-...          # or "EMPTY" when pointing at a local vLLM server
HOSTIFY_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### Start all services

```bash
docker compose up --build
```

This starts: **postgres**, **redis**, **searxng**, **minio**, **api** (port 8000),
**worker** (port 8001), **mcp** (port 7000), and **nginx** (port 8080).

### Create a user and trigger the first episode

```bash
# Register preferences
curl -s -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"schedule_time":"07:30","topics":["AI","Canada","Finance"],"max_duration_min":7,"voice":"en_US"}' \
  | jq .

# Poll until the episode is ready (user_id returned above)
curl -s http://localhost:8000/episodes/1/latest | jq .

# Fetch the RSS feed
curl -s http://localhost:8080/feed/1.rss
```

The worker generates an episode immediately on user creation.  Subsequent
episodes fire automatically at `schedule_time` each day.

### Stop and clean up

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop containers and delete volumes (wipes DB + audio)
```

---

## Model Configuration

Model selection is controlled by environment variables read by
[services/mcp/settings.py](services/mcp/settings.py).

### Deployment tiers

| Tier | `HOSTIFY_PROVIDER` | Model | VRAM | Use case |
|------|--------------------|-------|------|----------|
| **Production** | `api` | `Qwen/Qwen2.5-7B-Instruct` (vLLM) | ~14 GB | Server GPU, full quality |
| **Large** | `api` | `Qwen/Qwen2.5-14B-Instruct` (vLLM) | ~28 GB | Two-GPU or A10G |
| **Local dev** | `local` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | ~6 GB | Single consumer GPU |
| **CPU fallback** | `cpu` | `facebook/bart-large-cnn` | None | No GPU; pipeline smoke-tests only |

### Switching tiers

Set `HOSTIFY_PROVIDER` in your `.env`:

```bash
HOSTIFY_PROVIDER=api        # OpenAI-compatible endpoint (default)
HOSTIFY_PROVIDER=local      # load model in-process (needs GPU)
HOSTIFY_PROVIDER=cpu        # CPU-only BART pipeline (not suitable for production)
```

To point `api` mode at a local vLLM server instead of OpenAI:

```bash
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://localhost:8000/v1
HOSTIFY_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### Why Qwen2.5?

Qwen2.5-7B-Instruct ranked first in the 7–8B weight class on the Open LLM
Leaderboard (January 2026) for instruction following and structured JSON output
— both critical for the `CritiqueAgent` and `HumanificationAgent` nodes in the
LangGraph hostify pipeline.  The full rationale and benchmark citations are in
[services/mcp/settings.py](services/mcp/settings.py).

---

## Design Decisions

**Intentional service duplication over a shared package.**
`base_agent.py` exists in both `services/mcp/` and `services/worker/` with an
explicit sync contract rather than a `services/common/` package.  A shared
package requires coordinated Docker build changes and introduces import path
fragility for only two consumers.  The refactor trigger is clear: a third
service needs `BaseAgent`, or the class exceeds ~50 lines.

**Synchronous Celery tasks, async MCP calls.**
`summarizer_client.py` uses blocking `requests` because Celery workers are
synchronous processes.  `agent_search.py` uses `aiohttp` because it runs
inside an `asyncio` event loop.  Mixing them would block the event loop or
require a full Celery async migration — neither is warranted at current load.

**Single LLM round-trip for agentic search.**
`choose_and_summarize()` collapses topic ranking and article summarisation into
one structured prompt.  A two-step approach would re-send the same candidate
articles twice, doubling latency and token cost.  The strict JSON output schema
makes the result directly usable downstream without a parsing step.

**Progressive time-window fallback in RSS ingestion.**
`NewsAgent` retries with expanding windows (7d → 30d → 1y) before returning
`no_news_today`.  The narrow default catches fresh articles first; widening on
retry keeps the episode non-empty during slow news cycles without permanently
degrading freshness for active topics.

---

## Development

### Run a single service without Docker

Each service can be started independently against a locally-running postgres
and redis (or use `docker compose up postgres redis` to start just the
infrastructure).

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/newscast"
export REDIS_URL="redis://localhost:6379/0"
export MCP_URL="http://localhost:7000"
export OPENAI_API_KEY="sk-..."

# API service
uvicorn services.api.app:app --reload --port 8000

# MCP service (separate terminal)
uvicorn services.mcp.server:app --reload --port 7000

# Worker scheduler server (separate terminal)
uvicorn services.worker.server:app --port 8001

# Celery worker (separate terminal)
celery -A services.worker.tasks.celery worker --loglevel=info
```

### Code style

```bash
black services/           # format
isort services/           # sort imports
mypy services/            # type check
```

Configuration for all three tools lives in [pyproject.toml](pyproject.toml).

### Running ingestion standalone

```bash
python -m services.worker.ingestion
```

Runs the CLI entry point in `ingestion.py` and logs scored articles — useful
for verifying feed connectivity and topic scoring without starting the full
stack.
