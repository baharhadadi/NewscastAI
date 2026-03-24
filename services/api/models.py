"""
services/api/models.py
-----------------------
SQLAlchemy ORM models for the NewscastAI API tier.

These models define the persistent data schema that links user preferences to
the episode pipeline.  The flow is:

  ``User`` (preferences) → Celery task → ``Episode`` (generated audio + manifest)

The ``Article`` model is reserved for future use as a deduplication cache so
the worker can skip re-fetching articles that have already been processed in a
recent window.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from .db import Base


class User(Base):
    """Registered listener with personalised newscast preferences.

    Each ``User`` row drives one or more ``Episode`` rows: the Celery worker
    reads ``topics``, ``voice``, and ``max_duration_min`` when generating each
    episode, and APScheduler checks ``schedule_time`` every minute to decide
    when to enqueue the generation task.

    Attributes:
        id: Auto-incrementing primary key.
        email: Optional contact address; not currently used by the pipeline.
        schedule_time: ``"HH:MM"`` string (24-hour, local time) at which a
            new episode is generated daily.
        max_duration_min: Soft upper bound on episode audio length in minutes.
            Passed to the Hostify graph's ``max_audio_seconds`` parameter.
        topics: JSON list of topic strings (e.g. ``["AI", "Canada", "Finance"]``).
            Passed verbatim to ``NewsAgent.run()`` and ``TOPIC_EXPANSIONS``
            look-up in ``constants.py``.
        voice: BCP-47 locale hint forwarded to the TTS service (e.g.
            ``"en_US"``).  Controls accent/language of the synthesised audio.
        created_at: UTC timestamp of account creation.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=True)
    schedule_time = Column(String, nullable=False)  # HH:MM
    max_duration_min = Column(Integer, default=7)
    topics = Column(JSON, default=list)
    voice = Column(String, default="en_US")
    created_at = Column(DateTime, default=datetime.utcnow)


class Article(Base):
    """Cached article record for deduplication (reserved for future use).

    Attributes:
        id: Auto-incrementing primary key.
        url: Canonical article URL used as the uniqueness key.
        title: Article headline.
        text: Full article body text extracted by newspaper3k.
        tags: JSON list of topic/category tag strings from the RSS feed.
    """

    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    url = Column(Text, unique=True)
    title = Column(Text)
    text = Column(Text)
    tags = Column(JSON, default=list)


class Episode(Base):
    """A generated newscast episode for one user on one day.

    An ``Episode`` row is written by ``generate_episode`` in ``tasks.py``
    after the full pipeline (fetch → summarise → TTS → assemble) completes.
    The ``ready`` flag is set to ``True`` only after the audio file has been
    successfully written to disk, so the API never serves a partially-built
    episode.

    Attributes:
        id: Auto-incrementing primary key.
        user_id: Foreign key to ``users.id``.
        date: UTC timestamp when the episode was generated.
        audio_path: Logical path to the assembled MP3, always prefixed with
            ``/audio/`` (e.g. ``"/audio/user_1_20240315.mp3"``).  nginx serves
            files under this prefix from the shared ``audio`` Docker volume.
        manifest: JSON payload describing the episode content.  Currently
            stored as a list of broadcast-script strings (one per article) or
            a dict with ``script_path`` and ``stories`` keys when the Hostify
            graph is used.  Drives the ``/latest_details`` transcript view.
        ready: ``True`` once audio assembly is complete and the file exists on
            disk.  ``False`` while the pipeline is running or if it failed.
        user: SQLAlchemy relationship back to the owning ``User`` row.
    """

    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime, default=datetime.utcnow)
    audio_path = Column(Text)
    manifest = Column(JSON)  # list of summaries or structured dict
    ready = Column(Boolean, default=False)
    user = relationship("User")
