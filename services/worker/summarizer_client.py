"""
services/worker/summarizer_client.py
--------------------------------------
Thin HTTP client for the MCP batch-summarisation endpoint.

Only one function is intentionally exposed here: ``summarize_many()``.
The former ``chat_json()`` helper has been removed — it was declared
``async def`` but used blocking ``requests.post`` internally, which would
have stalled the event loop.  The only caller (``agent_search.py``) now
makes its MCP call inline using ``aiohttp``, which is already a dependency
of that module.
"""

from typing import Any, Dict, List

import requests
import requests.exceptions

from .settings import settings

# Per-request timeout (seconds).  Batch summarisation can be slow when the
# HuggingFace model is cold-starting; 120 s accommodates up to ~10 articles.
_SUMMARIZE_TIMEOUT_S: int = 120


# Synchronous HTTP call — intentional.
# This runs inside a Celery worker task which is synchronous.
# Do not convert to async without migrating the entire
# Celery task to an async worker (e.g. aiohttp + asyncio).
def summarize_many(
    items: List[Dict[str, Any]],
    voice_style: str = "host",
) -> List[str]:
    """Submit a batch of articles to the MCP summariser and return script lines.

    Args:
        items: List of article dicts, each with at least ``title`` and ``text``.
        voice_style: Narration style hint forwarded to the MCP service.

    Returns:
        Ordered list of broadcast-style script strings, one per input article.

    Raises:
        requests.exceptions.HTTPError: On a non-2xx MCP response.
        requests.exceptions.ConnectionError: If the MCP service is unreachable.
        requests.exceptions.Timeout: If the request exceeds ``_SUMMARIZE_TIMEOUT_S``.
    """
    r = requests.post(
        f"{settings.mcp_url}/summarize_batch",
        json={"items": items, "style": voice_style},
        timeout=_SUMMARIZE_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()["summaries"]
