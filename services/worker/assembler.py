"""
services/worker/assembler.py
-----------------------------
Audio assembly: stitches TTS-generated MP3 clips into a single episode file
using pydub.  Handles path resolution between logical audio URLs stored in the
database and the actual filesystem paths used by the audio directory.
"""

import logging
from typing import List, Optional

from pydub import AudioSegment
import os

from .settings import settings

_logger = logging.getLogger(__name__)

def _realize_path(p: str | None) -> str | None:
    """Resolve a logical audio reference to an absolute filesystem path.

    Path resolution is centralised here rather than inlined in ``stitch()``
    because the TTS service may return either an absolute OS path or a
    URL-style logical path (``/audio/foo.mp3``), while the database stores
    logical paths.  A single resolution function makes it trivial to add new
    path conventions (e.g. S3 URIs, CDN prefix mapping) without touching
    the assembly logic.

    Handles three input forms:

    - Absolute path that already exists on disk → returned as-is.
    - Logical URL like ``/audio/<name>.mp3`` → mapped to
      ``<settings.audio_dir>/<name>.mp3`` if that file exists.
    - Bare filename like ``name.mp3`` → looked up in ``settings.audio_dir``.

    Args:
        p: Path string in any of the above forms, or ``None``.

    Returns:
        Resolved absolute path if the file exists, otherwise ``None`` so
        the caller can skip missing clips without raising.
    """
    if not p:
        return None

    # Already a real absolute path — return directly without remapping.
    if os.path.isabs(p) and os.path.exists(p):
        return p

    # Logical paths like "/audio/foo.mp3" store only the basename in the DB;
    # resolve against the configured audio directory.
    name = os.path.basename(p)
    candidate = os.path.join(settings.audio_dir, name)
    if os.path.exists(candidate):
        return candidate

    return None

def stitch(
    intro_path: Optional[str],
    segments_paths: List[Optional[str]],
    outro_path: Optional[str],
    out_path: str,
) -> str:
    """Concatenate audio clips into a single episode MP3.

    Output layout: ``[settings.audio_padding_ms silence] + intro + body clips + outro``.

    Missing clips are silently skipped (``_realize_path`` returns ``None``),
    so a partial episode (e.g. intro-only when no articles were found) is
    still assembled cleanly without raising.

    Args:
        intro_path: Logical or absolute path to the intro MP3 clip.
        segments_paths: Ordered list of paths to body-segment clips.
            May be empty for no-news episodes.
        outro_path: Path to the outro MP3 clip.
        out_path: Absolute filesystem path where the output MP3 is written.
            Parent directory is created if it does not already exist.

    Returns:
        The ``out_path`` string, confirming where the file was written.

    Raises:
        OSError: If the output directory cannot be created or the MP3 cannot
            be written.
        pydub.exceptions.CouldntDecodeError: If any input clip is corrupt.
    """
    # Leading silence duration is configurable via settings.audio_padding_ms (default 250 ms).
    # It serves two purposes:
    # (1) Prevents audio players that aggressively trim leading MP3 silence
    #     from clipping the very first word of the intro on some devices.
    # (2) Gives listeners a brief cognitive pause before content begins —
    #     the audio equivalent of a radio host's lead-in moment of quiet.
    audio = AudioSegment.silent(duration=settings.audio_padding_ms)

    # intro
    ip = _realize_path(intro_path)
    if ip:
        try:
            audio += AudioSegment.from_file(ip)
        except OSError as exc:
            _logger.warning("Could not load intro clip %s: %s", ip, exc)

    # body
    for p in (segments_paths or []):
        rp = _realize_path(p)
        if rp:
            try:
                audio += AudioSegment.from_file(rp)
            except OSError as exc:
                _logger.warning("Could not load segment clip %s: %s", rp, exc)

    # outro
    op = _realize_path(outro_path)
    if op:
        try:
            audio += AudioSegment.from_file(op)
        except OSError as exc:
            _logger.warning("Could not load outro clip %s: %s", op, exc)

    # ensure output dir exists
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except OSError as exc:
        _logger.error("Cannot create audio output directory for %s: %s", out_path, exc)
        raise

    try:
        audio.export(out_path, format="mp3")
    except OSError as exc:
        _logger.error("Failed to write episode MP3 to %s: %s", out_path, exc)
        raise

    return out_path
