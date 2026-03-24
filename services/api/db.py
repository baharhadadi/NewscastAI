"""
services/api/db.py
-------------------
SQLAlchemy engine and session-factory setup for the API tier.

This module is intentionally kept separate from ``models.py`` to break the
circular import that would otherwise occur: ``models.py`` needs ``Base`` (from
here) to define its ORM classes, and ``app.py`` needs both the models and the
``engine`` object to call ``Base.metadata.create_all()``.  Keeping the engine
and declarative base in their own module lets both consumers import from one
authoritative place without creating a dependency cycle.

``pool_pre_ping=True`` causes SQLAlchemy to check each connection is alive
before using it, so stale connections are recycled automatically after a
postgres restart (common in Docker Compose development workflows).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from .settings import settings

engine = create_engine(settings.database_url, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
