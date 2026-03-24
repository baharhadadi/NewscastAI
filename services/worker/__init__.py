"""
services/worker
---------------
Celery worker tier for the NewscastAI backend.

Responsibilities
----------------
This package owns everything that runs inside the Celery worker process:

- **Ingestion** (``ingestion.py``) — async RSS polling, article scoring,
  multi-window fallback search, and full-text extraction via newspaper3k.
- **Agentic search** (``agent_search.py``) — SearxNG-backed web search with
  LLM-driven topic selection as an alternative / complement to RSS feeds.
- **Task orchestration** (``tasks.py``) — the ``generate_episode`` Celery task
  that drives the full pipeline: fetch → summarise → TTS → assemble → persist.
- **MCP clients** (``tts_client.py``, ``summarizer_client.py``) — thin HTTP
  wrappers that call the MCP service tier for TTS synthesis and summarisation.
- **Audio assembly** (``assembler.py``) — stitches per-segment MP3 clips into
  a single episode file using pydub.
- **Configuration** (``settings.py``) — pydantic-settings ``BaseSettings``
  instance; reads all tuneable values from environment variables / ``.env``.
- **Constants** (``constants.py``) — ``TOPIC_EXPANSIONS`` and
  ``DOMAIN_AUTHORITY`` look-up tables shared across ingestion and search.
- **Base agent** (``base_agent.py``) — ``AgentResult`` dataclass and
  ``BaseAgent`` ABC that standardise agent return envelopes.

Architecture note
-----------------
The worker tier communicates with the MCP tier exclusively over HTTP (via the
MCP client modules above).  It never imports MCP internals directly, so the
two tiers can be scaled and deployed independently.
"""
