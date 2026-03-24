"""
services/worker/tasks.py
-------------------------
Celery task definitions for the newscast worker tier.

The single exported task, ``generate_episode``, orchestrates the full pipeline
from article ingestion through TTS synthesis and audio assembly to database
persistence.  It is enqueued by the API server when a user's scheduled
generation window is reached.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .assembler import stitch
from .ingestion import agentic_fetch_articles as fetch_articles
from .settings import settings
from .summarizer_client import summarize_many
from .tts_client import tts

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)

celery = Celery(__name__, broker=settings.redis_url, backend=settings.redis_url)

AUDIO_DIR = settings.audio_dir
os.makedirs(AUDIO_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Private pipeline helpers
# ---------------------------------------------------------------------------

def _fetch_and_filter_articles(
    topics: List[str],
    max_n: int = 12,
) -> List[Dict[str, str]]:  # TODO: replace with TypedDict {"title": str, "text": str}
    """Fetch articles via the RSS ingestion pipeline and filter to usable items.

    Extracted from ``generate_episode`` to isolate the fetch-and-filter concern
    from task orchestration.  Two invariants are maintained:

    1. Only articles with both a non-empty ``title`` and non-empty body text
       (``text`` preferred over ``summary`` fallback) are returned.
    2. Article text is clamped to 15,000 characters to protect the MCP
       summariser against abnormally long inputs.

    Args:
        topics: User topic strings forwarded to ``fetch_articles()``.
        max_n: Maximum number of articles to keep after fetching.  A slightly
            larger initial fetch (``limit=max_n + 8``) is requested so that
            articles without text can be dropped without leaving the slate empty.

    Returns:
        List of ``{"title": str, "text": str}`` dicts ready for
        ``summarize_many()``.  Empty list when no qualifying articles are found.
    """
    raw = fetch_articles(topics, limit=max_n + 8)[:max_n]
    items = []
    for a in raw:
        title = a.get("title") or ""
        # Prefer full text; fallback to RSS summary; skip if neither exists
        text = (a.get("text") or a.get("summary") or "").strip()
        if not title or not text:
            continue
        items.append({"title": title, "text": text[:15000]})
    return items


def _build_no_content_episode(user: Any, db: Session, stamp: str) -> str:
    """Generate and persist a graceful empty episode when no articles match.

    Extracted from ``generate_episode`` to isolate the no-content path, which
    has a distinct TTS script ("No fresh articles…") and skips the summarise
    step entirely.  Commits the ``Episode`` row before returning so the caller
    can immediately return the audio URL without additional DB work.

    Invariant: always persists an ``Episode`` row with ``ready=True`` and
    returns a valid ``/audio/…`` path, so the API never serves a missing
    episode to the client.

    Args:
        user: ``User`` ORM instance whose ``id`` and ``voice`` are used.
        db: Open SQLAlchemy session; the caller must close it after this call.
        stamp: ``"%Y%m%d"``-formatted date string used in the filename.

    Returns:
        Relative audio URL string (e.g. ``"/audio/user_1_20240315.mp3"``).
    """
    from services.api.models import Episode  # circular import — intentional

    intro = tts("Good morning. No fresh articles matched your preferences today.", voice=user.voice)
    outro = tts("That's all for now. Have a great day.", voice=user.voice)
    out_rel = f"/audio/user_{user.id}_{stamp}.mp3"
    out_abs = os.path.join(AUDIO_DIR, f"user_{user.id}_{stamp}.mp3")
    stitch(intro, [], outro, out_abs)
    ep = Episode(user_id=user.id, audio_path=out_rel, manifest=[], ready=True)
    db.add(ep)
    db.commit()
    return out_rel


def _run_tts_pipeline(
    summaries: List[str],
    voice: str,
) -> Tuple[str, List[str], str]:
    """Synthesise all script lines to MP3 clips via the MCP TTS service.

    Extracted from ``generate_episode`` to isolate TTS I/O from orchestration
    logic.  Two invariants are maintained:

    1. Per-segment failures (network blip, quota error) are silently skipped
       with a warning log — a partial episode is far better than no episode.
    2. Intro and outro synthesis is kept in this helper so all TTS calls are
       co-located; callers receive a complete ``(intro, clips, outro)`` triple
       ready to pass straight to ``stitch()``.

    Args:
        summaries: Ordered list of broadcast-script strings from
            ``summarize_many()``, one per article.
        voice: BCP-47 locale hint forwarded to each ``tts()`` call.

    Returns:
        ``(intro_path, body_clip_paths, outro_path)`` where each path is the
        absolute filesystem path of the generated MP3 clip.
    """
    clips: List[str] = []
    for s in summaries:
        try:
            clips.append(tts(s, voice=voice))
        except Exception as exc:
            logger.warning("TTS failed for segment, skipping: %s", exc)

    intro = tts("Good morning. Here is your personalized news briefing.", voice=voice)
    outro = tts("That's all for today. Have a great day.", voice=voice)
    return intro, clips, outro


def _persist_episode(
    user_id: int,
    audio_path: str,
    manifest: List[str],
    db: Session,
) -> str:
    """Write an Episode row to the database and return the audio path.

    Extracted from ``generate_episode`` to isolate the persistence concern.
    Sets ``ready=True`` immediately so the API can serve the episode as soon
    as this function returns — the caller must ensure the audio file has
    already been written to disk before calling this helper.

    Args:
        user_id: Primary key of the owning ``User`` row.
        audio_path: Relative ``/audio/…`` URL stored in the DB column and
            returned to the API.
        manifest: List of broadcast-script strings (one per article) stored
            as the episode transcript manifest.
        db: Open SQLAlchemy session; the caller must close it after this call.

    Returns:
        ``audio_path`` unchanged, so the Celery task can return it directly.
    """
    from services.api.models import Episode  # circular import — intentional

    ep = Episode(user_id=user_id, audio_path=audio_path, manifest=manifest, ready=True)
    db.add(ep)
    db.commit()
    return audio_path


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery.task(name="generate_episode")
def generate_episode(user_id: int) -> Optional[str]:
    """Orchestrate the full newscast pipeline for one user and persist the result.

    This is the top-level entry point for episode generation.  A new engineer
    reading this function can follow the pipeline top-to-bottom:

    1. **FETCH** — ``_fetch_and_filter_articles()`` polls RSS feeds via
       ``NewsAgent``, scoring candidates by topic relevance, recency, and
       source authority across up to three progressive time windows (7 d → 30 d
       → 1 yr).

    2. **FILTER** — Handled inside ``_fetch_and_filter_articles()``: full
       article text is preferred over RSS summaries; articles with neither are
       dropped; a 15,000-char cap protects the summariser.

    3. **SUMMARIZE** — ``summarize_many()`` compresses each article into a
       2–3 sentence broadcast-style script line via the MCP summariser.

    4. **TTS** — ``_run_tts_pipeline()`` synthesises each script line plus
       fixed intro/outro strings to uniquely-named MP3 clips.

    5. **ASSEMBLE** — ``stitch()`` concatenates ``[intro + clips + outro]``
       into a single episode MP3 written to ``AUDIO_DIR``.

    6. **PERSIST** — ``_persist_episode()`` commits an ``Episode`` row with
       the relative audio URL and summary manifest.

    Empty-results path: if no articles survive the filter step,
    ``_build_no_content_episode()`` produces a minimal "no news today" episode
    so the user always receives a valid, playable audio file.

    Args:
        user_id: Primary key of the ``User`` row whose topic preferences,
            voice setting, and schedule drive this generation run.

    Returns:
        Relative audio URL string (e.g. ``"/audio/user_1_20240315.mp3"``) on
        success, or ``None`` if the user record was not found in the database.
    """
    from services.api.models import User  # circular import — intentional

    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            return None

        items = _fetch_and_filter_articles(user.topics)
        stamp = datetime.utcnow().strftime("%Y%m%d")

        if not items:
            return _build_no_content_episode(user, db, stamp)

        summaries = summarize_many(items)
        intro, clips, outro = _run_tts_pipeline(summaries, user.voice)

        out_rel = f"/audio/user_{user.id}_{stamp}.mp3"
        out_abs = os.path.join(AUDIO_DIR, f"user_{user.id}_{stamp}.mp3")
        stitch(intro, clips, outro, out_abs)
        logger.info("Generated audio record: %s", out_rel)
        return _persist_episode(user.id, out_rel, summaries, db)
    finally:
        db.close()
