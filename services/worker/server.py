"""
services/worker/server.py
--------------------------
FastAPI application and APScheduler daemon for the worker tier.

Exposes a ``POST /kick`` endpoint that enqueues an episode generation task
immediately (used by the API service when a new user is created).  Also runs
a background scheduler that checks every minute whether any user's
``schedule_time`` matches the current wall-clock time and enqueues their task
accordingly.
"""

import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.api.models import User
from services.worker.settings import settings
from .tasks import generate_episode

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)

tz = pytz.timezone(settings.tz)
sched = BackgroundScheduler(timezone=tz)

app = FastAPI()


class Kick(BaseModel):
    """Request body for the ``POST /kick`` endpoint.

    Attributes:
        user_id: Primary key of the ``User`` whose episode should be generated
            immediately.  Must match an existing ``users.id`` row; the Celery
            task will return ``None`` without error if the user is not found.
    """

    user_id: int


@app.post("/kick")
def kick(k: Kick):
    """Enqueue an episode generation task for the given user immediately.

    Called by the API service (``services/api/app.py``) via an HTTP POST right
    after a new ``User`` row is created, so the first episode is generated
    without waiting for the next scheduled tick.  Can also be triggered
    manually (e.g. via ``curl``) to force an out-of-schedule regeneration.

    The task is dispatched asynchronously via ``generate_episode.delay()``;
    this endpoint returns as soon as the task is enqueued in Redis, not when
    the episode is complete.  Clients should poll
    ``GET /episodes/{user_id}/latest`` for completion.

    Args:
        k: ``Kick`` payload containing the target ``user_id``.

    Returns:
        ``{"queued": True}`` immediately upon successful enqueue.
    """
    generate_episode.delay(k.user_id)
    return {"queued": True}


def tick():
    """Scheduled job: enqueue episode generation for every user whose
    ``schedule_time`` matches the current wall-clock minute.

    Runs on a one-minute interval via APScheduler (see ``sched.add_job``
    below).  Compares ``datetime.now(tz).strftime("%H:%M")`` against each
    user's ``schedule_time`` string; users whose time matches the current
    minute have their episode task enqueued.

    Only users with a ``schedule_time`` exactly matching the current minute
    receive a task — there is no catch-up logic for missed minutes (e.g.
    if the worker was down).  This is intentional: a missed daily briefing
    is less disruptive than generating a backlog of stale episodes.
    """
    now = datetime.now(tz).strftime("%H:%M")
    db = SessionLocal()
    try:
        for u in db.query(User).all():
            if u.schedule_time == now:
                generate_episode.delay(u.id)
    finally:
        db.close()


sched.add_job(tick, "interval", minutes=1)
sched.start()
