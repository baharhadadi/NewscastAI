"""
services/worker/settings.py
----------------------------
Central configuration for the Celery worker tier.

All values are read from environment variables.  Set them in your .env file
or docker-compose environment block; the defaults below are safe for local dev.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # --- LLM / Hostify ---
    # Base URL of the internal MCP service that runs the Hostify/summarizer pipeline.
    mcp_url: str = "http://mcp:7000"

    # --- Audio ---
    # Filesystem path where generated MP3 files are stored (shared with MCP tier).
    audio_dir: str = "/mnt/audio"

    # --- Infrastructure ---
    # SQLAlchemy-compatible database connection string.
    database_url: Optional[str] = None

    # Redis connection string used as both Celery broker and result backend.
    redis_url: Optional[str] = None

    # IANA timezone string used for scheduling and timestamp display.
    tz: str = "America/Toronto"

    # SearxNG instance base URL for web search.
    searxng_url: str = "http://searxng:8080"

    # SearxNG time_range parameter: "day" | "week" | "month" | "year".
    news_time_range: str = "day"

    # Maximum total search results fetched per episode generation run.
    news_max_results: int = 50

    # BCP-47 language code passed to SearxNG.
    news_lang: str = "en"

    # ISO 3166-1 alpha-2 region code passed to SearxNG.
    news_region: str = "ca"

    # --- Scoring weights (ingestion.py score_facet_items) ---
    # Weights were tuned empirically; see score_facet_items() for the full formula.
    # Recency is double-weighted because stale news is the primary failure mode.
    score_weight_facet: float = 1.0      # raw keyword-hit relevance contribution
    score_weight_recency: float = 2.0    # exponential-decay recency signal
    score_weight_authority: float = 1.2  # source domain authority (DOMAIN_AUTHORITY table)

    # --- Audio assembly ---
    # Leading silence (ms) prepended to every episode MP3.
    # Prevents aggressive players from clipping the first word of the intro, and
    # gives listeners a brief cognitive pause before content begins.
    audio_padding_ms: int = 250

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
