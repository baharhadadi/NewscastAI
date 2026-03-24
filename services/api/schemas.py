"""
services/api/schemas.py
------------------------
Pydantic request/response schemas for the API tier.

These are kept separate from the SQLAlchemy models in ``models.py`` to
maintain a clean boundary between the database representation and the HTTP
contract.  Pydantic handles input validation and OpenAPI documentation
generation automatically from these class definitions.
"""

from typing import List

from pydantic import BaseModel


class PrefsIn(BaseModel):
    """User preference payload for the ``POST /users`` endpoint.

    Defines the newscast parameters that drive episode generation.  All fields
    except ``max_duration_min`` and ``voice`` are required.

    Attributes:
        schedule_time: Daily generation time in ``"HH:MM"`` 24-hour format
            (e.g. ``"07:30"``).  The worker scheduler checks this value every
            minute against the current wall-clock time in the user's timezone.
        topics: Ordered list of topic strings the newscast should cover
            (e.g. ``["AI", "Canada", "Finance"]``).  Each string is looked up
            in ``TOPIC_EXPANSIONS`` in ``constants.py`` to produce a broader
            keyword set; arbitrary strings are also accepted and treated as
            literal search keywords.  Ordering does not affect article scoring
            but is preserved as a priority hint to the LLM agent.
        max_duration_min: Soft upper bound on episode audio duration in
            minutes.  Converted to ``max_audio_seconds`` when passed to the
            Hostify graph's compress node.  Defaults to ``7``.
        voice: BCP-47 locale hint forwarded to the TTS service (e.g.
            ``"en_US"``, ``"en_GB"``).  Controls accent and language of the
            synthesised audio.  Defaults to ``"en_US"``.
    """

    schedule_time: str
    topics: List[str]
    max_duration_min: int = 7
    voice: str = "en_US"


class PrefsOut(PrefsIn):
    """Response payload returned after a successful ``POST /users`` call.

    Extends ``PrefsIn`` with the server-assigned ``id`` so the client can
    immediately use it to poll for episodes and subscribe to the RSS feed
    without a second round-trip.

    Attributes:
        id: Auto-assigned primary key of the newly created ``User`` row.
            Used in ``GET /episodes/{user_id}/latest`` and
            ``GET /feed/{user_id}.rss``.
    """

    id: int
