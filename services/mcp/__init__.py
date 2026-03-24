"""
services/mcp
-------------
Model-Control-Plane (MCP) service tier for the NewscastAI backend.

Responsibilities
----------------
This package owns all LLM and ML inference, and exposes results to the worker
tier via an internal HTTP API:

- **Episode pipeline** (``hostify_graph.py``) — LangGraph directed graph that
  drives: plan → draft → validate → critique → compress.  The critique node
  runs a quality-gate LLM evaluation and may route back to draft for up to
  ``settings.critique_max_iterations`` revision passes before force-approving.
- **Humanification** (``humanification_agent.py``) — ``HumanificationAgent``
  inserts speech markers (<pause>, <breath>, <emm>, <emphasis>) calibrated to
  the requested delivery tone.
- **Summarisation** (``summarize.py``) — lazy-loaded HuggingFace pipelines for
  abstractive summarisation (BART-large-CNN) and style rewriting (flan-t5-base).
- **TTS** (``tts.py``) — gTTS synthesis to per-segment MP3 files.
- **Schema** (``hostify_schema.py``) — Pydantic models (``Episode``,
  ``Section``) that enforce the structured output contract between the LLM and
  the assembly pipeline.
- **Configuration** (``settings.py``) — pydantic-settings ``BaseSettings``
  instance with model registry, critique thresholds, and audio directory.
- **Base agent** (``base_agent.py``) — ``AgentResult`` dataclass and
  ``BaseAgent`` ABC shared with the worker tier.

Architecture note
-----------------
The MCP tier is stateless between requests and is designed to be horizontally
scalable.  It holds no database connections; persistence is the worker tier's
responsibility.  Model weights are loaded lazily on first use and held as
module-level singletons so they survive across requests within a single process.
"""
