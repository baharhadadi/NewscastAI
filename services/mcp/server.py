"""
services/mcp/server.py
-----------------------
FastAPI application for the MCP (Model-Control-Plane) service tier.

Exposes internal HTTP endpoints consumed by the worker tier:

- ``POST /summarize_batch`` — batch article summarisation
- ``POST /tts``             — text-to-speech synthesis
"""

import logging
import os
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from .settings import settings
from .summarize import get_summarizer, write_script
from .tts import speak

os.makedirs(settings.audio_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

app = FastAPI(title="MCP")


class Item(BaseModel):
    """A single article to be summarised.

    Attributes:
        title: Article headline; used as context when rewriting the summary
            into a broadcast-style script line via ``write_script()``.
        text: Full article body text (or RSS summary as fallback).  Fed
            directly into the BART summarisation pipeline.
    """

    title: str
    text: str


class BatchIn(BaseModel):
    """Request body for the ``POST /summarize_batch`` endpoint.

    Attributes:
        items: Ordered list of articles to summarise.  Results are returned
            in the same order so the caller can correlate them with the input.
        style: Narration style hint (currently unused by the BART pipeline;
            reserved for future style-conditioned models).  Defaults to
            ``"host"``.
    """

    items: List[Item]
    style: str = "host"


class TTSIn(BaseModel):
    """Request body for the ``POST /tts`` endpoint.

    Attributes:
        text: Narration string to synthesise into audio.  Should be a single
            sentence or short paragraph; very long strings may be truncated by
            gTTS.
        voice: BCP-47 locale hint (e.g. ``"en_US"``, ``"en_GB"``).  Forwarded
            to ``speak()`` in ``tts.py``; currently only the language component
            (``"en"``) is used by gTTS.  Defaults to ``"en_US"``.
    """

    text: str
    voice: str = "en_US"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[\.\!\?])\s+", (text or "").strip())


def _chunk_by_tokens(sents: List[str], tokenizer: Any, max_tokens: int) -> List[str]:
    chunks, buf, buf_tok = [], [], 0
    for s in sents:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
            s = tokenizer.decode(ids, skip_special_tokens=True)
        if buf and (buf_tok + len(ids)) > max_tokens:
            chunks.append(" ".join(buf))
            buf, buf_tok = [], 0
        buf.append(s)
        buf_tok += len(ids)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def summarize_long_text(
    text: str,
    summarizer: Any,
    max_input_tokens: int = 900,
    chunk_summary_max: int = 110,
    chunk_summary_min: int = 40,
    final_summary_max: int = 110,
    final_summary_min: int = 60,
) -> str:
    """Summarise arbitrarily long text using a map-reduce chunking strategy.

    BART-large-CNN accepts at most ~1024 tokens.  Articles that exceed
    ``max_input_tokens`` are split into sentences, packed into token-bounded
    chunks, each chunk summarised independently, and the chunk summaries
    recombined for a final summarisation pass.

    Use this function (rather than calling the pipeline directly) whenever the
    input text length is unknown.  ``summarize_batch`` calls the pipeline
    directly with a fixed 110-token output budget because the worker pre-clips
    article text to 15,000 characters, which the pipeline handles without
    chunking in the typical case.

    Args:
        text: Raw article body text to summarise.
        summarizer: Loaded HuggingFace ``transformers.pipeline`` instance
            (from ``get_summarizer()``).
        max_input_tokens: Maximum token count before map-reduce is triggered.
        chunk_summary_max: Maximum token count per intermediate chunk summary.
        chunk_summary_min: Minimum token count per intermediate chunk summary.
        final_summary_max: Maximum token count for the final summary.
        final_summary_min: Minimum token count for the final summary.

    Returns:
        Summarised text string.  Returns an empty string when ``text`` is
        empty or whitespace-only.
    """
    text = (text or "").strip()
    if not text:
        return ""
    tok = summarizer.tokenizer
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_input_tokens:
        return summarizer(
            text,
            max_length=final_summary_max,
            min_length=final_summary_min,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]
    # map-reduce over chunks
    sents = _split_into_sentences(text)
    chunks = _chunk_by_tokens(sents, tok, max_input_tokens)
    chunk_summaries = []
    for ch in chunks:
        cs = summarizer(
            ch,
            max_length=chunk_summary_max,
            min_length=chunk_summary_min,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]
        chunk_summaries.append(cs)
    combined = " ".join(chunk_summaries)
    return summarizer(
        combined,
        max_length=final_summary_max,
        min_length=final_summary_min,
        do_sample=False,
        truncation=True,
    )[0]["summary_text"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/summarize_batch")
def summarize_batch(payload: BatchIn) -> Dict[str, Any]:
    """Summarise a batch of articles and rewrite each as a broadcast script line.

    This is the primary entry point called by the worker tier's
    ``summarizer_client.summarize_many()``.  For each article it runs two
    sequential steps: (1) BART abstractive summarisation of the full article
    text, and (2) flan-t5 style rewriting of the summary into a podcast-style
    sentence via ``write_script()``.

    Args:
        payload: ``BatchIn`` containing the list of articles and an optional
            style hint.

    Returns:
        JSON object with three parallel lists — all in input order:

        - ``summaries`` — final broadcast-style script strings (one per item).
        - ``raw_summaries`` — raw BART output before style rewriting.
        - ``items`` — combined dicts with ``title``, ``raw_summary``, and
          ``script`` keys, useful for debugging the rewriting step.
    """
    summarizer = get_summarizer()
    scripts, raw_sums, packed = [], [], []
    for it in payload.items:
        s = summarizer(
            it.text, max_length=110, min_length=20, do_sample=False
        )[0]["summary_text"]
        raw_sums.append(s)
        script = write_script(it.title, s)
        scripts.append(script)
        packed.append({"title": it.title, "raw_summary": s, "script": script})
    return {"summaries": scripts, "raw_summaries": raw_sums, "items": packed}


@app.post("/tts")
def tts(inp: TTSIn) -> Dict[str, str]:
    """Synthesise a text string to an MP3 file and return its logical path.

    Calls ``speak()`` from ``tts.py`` (gTTS), which writes a UUID-named MP3
    to ``settings.audio_dir``.  Returns a ``/audio/``-prefixed logical path
    rather than an absolute filesystem path so the caller can construct a
    public URL by prepending the nginx base URL.

    Args:
        inp: ``TTSIn`` containing ``text`` and optional ``voice`` hint.

    Returns:
        JSON ``{"audio_path": "/audio/<uuid>.mp3"}`` where the path is
        directly appendable to the nginx base URL to produce a playable link.
    """
    path = speak(inp.text, voice=inp.voice)
    rel = "/audio/" + os.path.basename(path)
    return {"audio_path": rel}
