"""
services/worker/tts_client.py
------------------------------
Thin HTTP client for the MCP text-to-speech endpoint.

Converts a text string to an MP3 file via the MCP service and returns the
absolute filesystem path of the generated clip.
"""

import logging

import requests
import requests.exceptions

from .settings import settings

_logger = logging.getLogger(__name__)

# Per-request timeout (seconds).  TTS synthesis is CPU/network-bound;
# 30 s is generous for a single sentence but prevents hung workers.
_TTS_TIMEOUT_S: int = 30


def tts(text: str, voice: str = "en_US") -> str:
    """Send *text* to the MCP TTS service and return the MP3 file path.

    Args:
        text: Narration string to synthesise.
        voice: BCP-47 locale hint forwarded to the MCP service (e.g. ``"en_US"``).

    Returns:
        Absolute filesystem path of the generated MP3 clip.

    Raises:
        requests.exceptions.HTTPError: On a non-2xx response from the MCP service.
        requests.exceptions.ConnectionError: If the MCP service is unreachable.
        requests.exceptions.Timeout: If the request exceeds ``_TTS_TIMEOUT_S``.
        KeyError: If the MCP response JSON is missing the ``audio_path`` field.
    """
    r = requests.post(
        f"{settings.mcp_url}/tts",
        json={"text": text, "voice": voice},
        timeout=_TTS_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()["audio_path"]
