"""
services/api
-------------
FastAPI application tier for the NewscastAI backend.

Responsibilities
----------------
This package is the public-facing HTTP layer of the system:

- **App** (``app.py``) — FastAPI application with endpoints for user
  preference management, episode retrieval, and RSS feed generation.
  Delegates episode generation to the Celery worker tier via a ``/kick``
  HTTP call rather than importing worker internals directly.
- **Models** (``models.py``) — SQLAlchemy ORM models (``User``, ``Article``,
  ``Episode``).  Imported by the worker tier's scheduler to query user records.
- **Schemas** (``schemas.py``) — Pydantic request/response models
  (``PrefsIn``, ``PrefsOut``) used for input validation and OpenAPI docs.
- **RSS** (``rss.py``) — ``APIRouter`` that generates per-user Atom/RSS feeds
  pointing to episode audio files.
- **Database** (``db.py``) — SQLAlchemy engine and ``SessionLocal`` factory.
- **Configuration** (``settings.py``) — pydantic-settings ``BaseSettings``
  instance; reads all tuneable values from environment variables / ``.env``.

Architecture note
-----------------
The API tier never imports from ``services.worker`` or ``services.mcp``
internals at runtime; communication is strictly over HTTP.  The one exception
is ``services.worker.server`` which imports ``services.api.models`` directly
to run the scheduler in the same process — this is a deliberate architectural
shortcut documented in the worker ``__init__.py``.
"""
