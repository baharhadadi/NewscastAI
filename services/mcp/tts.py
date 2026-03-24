"""
services/mcp/tts.py
--------------------
Text-to-speech synthesis via gTTS (Google Text-to-Speech).  Each call
synthesises one string, writes a uniquely-named MP3 to the audio directory,
and returns the absolute path for downstream stitching.
"""

import uuid, os
from gtts import gTTS
from .settings import settings

def speak(text: str, voice: str = "en_US") -> str:
    """Synthesise *text* to an MP3 file and return its absolute path.

    A UUID4 hex string is used as the filename so concurrent calls never
    collide, even for identical input text.

    Args:
        text: Narration string to synthesise.
        voice: BCP-47-style locale hint (e.g. ``"en_US"``).  Currently only
            the language component (``"en"``) is forwarded to gTTS; the region
            suffix is accepted for API compatibility but has no effect.

    Returns:
        Absolute filesystem path of the generated MP3 file.

    Raises:
        gTTSError: If the Google TTS API call fails (network error, quota, etc.).
        OSError: If the configured audio directory is not writable.
    """
    fname = f"{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(settings.audio_dir, fname)
    tts = gTTS(text=text, lang="en")
    tts.save(out_path)
    return out_path
