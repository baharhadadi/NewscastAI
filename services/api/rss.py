"""
services/api/rss.py
--------------------
RSS 2.0 feed endpoint — the primary delivery mechanism for generated episodes.

Each user gets a personal RSS feed at ``/feed/{user_id}.rss``.  Podcast
clients (Apple Podcasts, Spotify, Overcast, etc.) subscribe to this URL and
poll it periodically.  When a new episode is ready they download the MP3 via
the ``<enclosure>`` element, which points to the audio file served by nginx.

RSS is used rather than a proprietary push protocol because:

- Every podcast client understands it natively.
- No client SDK or custom app is needed — the same feed works in any player.
- The feed doubles as a human-readable history of all generated episodes.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import Episode, User

router = APIRouter()


def get_db():
    """Yield a database session and guarantee it is closed after the request.

    Implements the FastAPI dependency-injection pattern for SQLAlchemy
    sessions.  The ``try / finally`` block ensures the session is closed and
    returned to the connection pool even when the route handler raises an
    exception, preventing connection leaks under load.

    Yields:
        An open ``sqlalchemy.orm.Session`` bound to ``SessionLocal``.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/feed/{user_id}.rss", response_class=Response)
def feed(user_id: int, db: Session = Depends(get_db)):
    """Generate an RSS 2.0 feed containing the user's last 10 episodes.

    Each ``<item>`` in the feed represents one episode and includes an
    ``<enclosure>`` element pointing to the MP3 file served by nginx.  Podcast
    clients use the ``<enclosure>`` to discover and download audio; the
    ``<pubDate>`` field drives playback ordering within the client.

    Returns a ``404`` response (not a JSON error) when the user does not exist,
    because RSS clients expect an HTTP status code rather than a JSON body.

    Args:
        user_id: Primary key of the ``User`` whose feed is requested.
        db: Injected database session from ``get_db()``.

    Returns:
        ``application/rss+xml`` response containing the feed XML, or a
        plain-text ``404`` response when the user is not found.
    """
    user = db.get(User, user_id)
    if not user:
        return Response(status_code=404, content="No user")
    eps = (
        db.query(Episode)
        .filter_by(user_id=user_id, ready=True)
        .order_by(Episode.date.desc())
        .limit(10)
        .all()
    )
    items = []
    for e in eps:
        pubdate = e.date.strftime("%a, %d %b %Y %H:%M:%S GMT")
        items.append(f"""        <item>
          <title>Newscast {e.date.date()}</title>
          <link>http://localhost:8080{e.audio_path}</link>
          <guid>{e.id}</guid>
          <pubDate>{pubdate}</pubDate>
          <enclosure url="http://localhost:8080{e.audio_path}" type="audio/mpeg"/>
        </item>
        """)
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0"><channel>
      <title>Newscast AI — {user_id}</title>
      <link>http://localhost:8080/feed/{user_id}.rss</link>
      <description>Daily personalized news</description>
      {''.join(items)}
    </channel></rss>"""
    return Response(content=xml, media_type="application/rss+xml")
