"""
services/mcp/summarize.py
--------------------------
HuggingFace pipeline wrappers for article summarisation and podcast-style
script rewriting.  Both pipelines are lazily loaded on first use so the MCP
service starts fast and only allocates GPU/CPU memory on demand.

Exports for hostify_graph.py
-----------------------------
  get_summarizer          — lazily-loaded BART summarisation pipeline
  summarize_text_block    — single-block wrapper with sentence fallback
  _shorten                — character-budget trimmer for audio compression
  HOSTIFY_PROMPT_GUIDELINES — canonical system-prompt constant
  generate_hostify_struct — provider-aware structured generation entry point
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from transformers import pipeline

from .settings import settings

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline singletons — lazily loaded to keep startup time fast
# ---------------------------------------------------------------------------
_summarizer = None
_stylist = None

# Lazy-loaded local model cache (only populated when hostify_provider == "local").
_local_tokenizer: Optional[Any] = None
_local_model: Optional[Any] = None


def get_summarizer() -> Any:
    """Return a lazy-initialised abstractive summarisation pipeline.

    The pipeline is a module-level singleton so model weights are loaded only
    once per process, regardless of how many times this function is called.
    Uses the model named in ``settings.summary_model``
    (default: ``facebook/bart-large-cnn``).

    Returns:
        HuggingFace ``transformers.pipeline`` configured for summarisation.
    """
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=settings.summary_model)
    return _summarizer


def get_stylist() -> Optional[Any]:
    """Return a lazy-initialised text2text rewriting pipeline, or ``None``.

    Returns ``None`` when ``settings.style_enabled`` is ``False``, so callers
    can check for ``None`` instead of reading the settings flag themselves.

    Uses the model named in ``settings.style_model``
    (default: ``google/flan-t5-base``).

    Returns:
        HuggingFace ``transformers.pipeline`` for text2text generation, or
        ``None`` if style rewriting is disabled.
    """
    global _stylist
    if not settings.style_enabled:
        return None
    if _stylist is None:
        _stylist = pipeline("text2text-generation", model=settings.style_model)
    return _stylist


# ---------------------------------------------------------------------------
# Style-rewriting (simple path used by the /summarize_batch route)
# ---------------------------------------------------------------------------
STYLE_PROMPT = """Rewrite the news for a short podcast in 2–3 sentences.
Tone: calm, neutral, friendly radio host. Avoid speculation or extra facts.
Start with the key point, then one detail, then why it matters.
Do not invent sources. Keep names and numbers as given.

Headline: {title}
Summary: {summary}

Script:"""


def write_script(title: str, summary: str) -> str:
    """Rewrite a headline and summary into a podcast-style broadcast sentence.

    Falls back to ``"{title}. {summary}"`` concatenation when the stylist is
    unavailable (disabled or failed to load), so the pipeline never blocks on
    a missing model.

    Args:
        title: Article headline.
        summary: Short article summary (typically 1–3 sentences).

    Returns:
        2–3 sentence broadcast-style narration string.
    """
    styler = get_stylist()
    if styler is None:
        return f"{title}. {summary}"
    out = styler(STYLE_PROMPT.format(title=title, summary=summary), max_new_tokens=128)[0]["generated_text"]
    return out.strip()


# ---------------------------------------------------------------------------
# Text utilities used by the Hostify graph pipeline
# ---------------------------------------------------------------------------

def _shorten(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, appending an ellipsis when cut.

    Used by the ``compress`` node in ``hostify_graph.py`` to enforce per-section
    character budgets derived from ``max_audio_seconds``.

    Args:
        text: Input string (script section, intro, or outro text).
        max_chars: Maximum character count for the output string including
            the appended ellipsis character when truncation occurs.

    Returns:
        Original text unchanged when it fits within *max_chars*.
        Truncated text ending with ``"…"`` when it exceeds the budget.
    """
    if not text or len(text) <= max_chars:
        return text or ""
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def summarize_text_block(
    text: str,
    summarizer: Any,
    max_len: int = 110,
    min_len: int = 60,
) -> str:
    """Summarise a single text block using the seq2seq pipeline.

    Used by ``_make_briefs()`` in ``hostify_graph.py`` to compress each
    article into a brief that fits the LLM's context window.

    Falls back to returning the first three sentences of the input when the
    model call fails (network blip, OOM, malformed input), so ``_make_briefs``
    always returns something even if the pipeline is unhealthy.

    Args:
        text: Raw article body text to summarise.
        summarizer: Loaded HuggingFace ``transformers.pipeline`` instance
            (from ``get_summarizer()``).
        max_len: Maximum token count for the generated summary.
        min_len: Minimum token count; avoids one-word outputs.

    Returns:
        Summarised text string, or first 3 sentences of input on failure.
    """
    text = (text or "").strip()
    if not text:
        return ""
    try:
        return summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]
    except Exception:
        # Graceful fallback: return the first three sentences so the pipeline
        # always has something to work with rather than an empty brief.
        sents = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sents[:3])


# ---------------------------------------------------------------------------
# Hostify structured generation — provider-aware LLM routing
# ---------------------------------------------------------------------------

# System prompt injected into every hostify LLM call.
# Centralised here so prompt changes propagate to both
# the graph-based and direct hostify_agent paths.
HOSTIFY_PROMPT_GUIDELINES = """
You are Hostify, an expert podcast script writer for a
personalized daily news briefing service.

VOICE AND TONE:
- Warm, authoritative, and conversational — like NPR but faster
- Address the listener directly ("you") sparingly
- Vary sentence length: mix short punchy sentences with longer
  explanatory ones to create natural audio rhythm

STRUCTURE RULES:
- intro: 2-3 sentences, sets the day's theme, no "welcome back"
- sections: each covers ONE story, opens with the headline
  implication not the headline itself
- outro: 1-2 sentences, forward-looking, never "that's all"

FACTUAL STANDARDS:
- Every claim must be traceable to the source briefs provided
- Do not infer causation beyond what the source states
- If a date is not in the source, do not mention one

OUTPUT:
- Return ONLY valid JSON matching the provided schema
- No preamble, no markdown, no explanation outside the JSON
""".strip()


def _generate_with_provider(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    json_schema: Optional[dict],
) -> str:
    """Route a generation request to the configured LLM provider.

    Provider selection is controlled by ``settings.hostify_provider``:

    - ``"api"``   — OpenAI-compatible HTTP API (vLLM, Together, OpenAI).
                    Uses ``settings.hostify_model`` and ``settings.openai_api_key``.
    - ``"local"`` — 4-bit quantised HuggingFace model loaded in-process.
                    Uses ``settings.hostify_local_model``; requires ~6 GB VRAM.
                    The tokenizer and model are cached as module-level singletons
                    so weights are only loaded once per process.
    - ``"cpu"``   — CPU-only BART fallback (slow; dev/test use only).
                    Will not produce valid structured JSON — use only to verify
                    the pipeline plumbing end-to-end without a GPU.

    Args:
        system_prompt: Role and formatting instructions for the model.
        user_prompt: Article briefs and output schema for this specific call.
        max_new_tokens: Maximum tokens to generate in the response.
        json_schema: JSON Schema dict for constrained decoding hints (passed
            as ``response_format`` to OpenAI-compatible APIs).
            ``None`` disables schema-guided generation.

    Returns:
        Raw text response from the model (not yet JSON-parsed).

    Raises:
        RuntimeError: If the provider call fails for any reason.
    """
    global _local_tokenizer, _local_model

    provider = settings.hostify_provider

    # ------------------------------------------------------------------
    # API path: OpenAI-compatible HTTP endpoint (vLLM / Together / OpenAI)
    # ------------------------------------------------------------------
    if provider == "api":
        from openai import OpenAI  # imported here to avoid hard dep at startup

        client = OpenAI(api_key=settings.openai_api_key or "EMPTY")
        call_kwargs: Dict[str, Any] = dict(
            model=settings.hostify_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.7,
        )
        if json_schema:
            # Ask the model to return a JSON object; schema used as a hint in
            # the prompt (true constrained decoding requires vLLM grammar mode).
            call_kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**call_kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"API provider call failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Local path: 4-bit quantised model loaded in-process
    # ------------------------------------------------------------------
    if provider == "local":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

            if _local_tokenizer is None or _local_model is None:
                _logger.info(
                    "Loading local model %s (first call only)…",
                    settings.hostify_local_model,
                )
                _local_tokenizer = AutoTokenizer.from_pretrained(
                    settings.hostify_local_model
                )
                _local_model = AutoModelForCausalLM.from_pretrained(
                    settings.hostify_local_model,
                    load_in_4bit=True,
                    device_map="auto",
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = _local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = _local_tokenizer([text], return_tensors="pt").to(
                _local_model.device
            )
            out_ids = _local_model.generate(**inputs, max_new_tokens=max_new_tokens)
            # Decode only the newly generated tokens (skip the prompt)
            new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            return _local_tokenizer.decode(new_ids, skip_special_tokens=True)
        except Exception as exc:
            raise RuntimeError(f"Local provider call failed: {exc}") from exc

    # ------------------------------------------------------------------
    # CPU fallback: BART summariser — returns plain text, not JSON
    # Use only for pipeline smoke-tests without GPU access.
    # ------------------------------------------------------------------
    _logger.warning(
        "hostify_provider=%r — using CPU BART fallback.  "
        "Output will NOT be valid JSON for structured generation.",
        provider,
    )
    summarizer = get_summarizer()
    combined = f"{system_prompt}\n\n{user_prompt}"[:3000]
    try:
        out = summarizer(
            combined,
            max_length=min(max_new_tokens, 300),
            min_length=60,
            do_sample=False,
        )[0]["summary_text"]
        return out
    except Exception as exc:
        raise RuntimeError(f"CPU provider call failed: {exc}") from exc


def generate_hostify_struct(
    system_prompt: str,
    user_prompt: str,
    json_schema: Optional[dict],
    max_new_tokens: int = 1200,
) -> Dict[str, Any]:
    """Generate a structured episode dict via the configured LLM provider.

    This is the single public entry point for all structured generation in the
    MCP service.  Provider routing (api / local / cpu) is handled internally
    by ``_generate_with_provider()``.

    Attempts generation once; on JSON parse failure strips markdown fences and
    retries once with an explicit correction instruction so transient formatting
    errors from the LLM do not abort the entire episode generation.

    Args:
        system_prompt: Role and formatting instructions for the model.
            Use ``HOSTIFY_PROMPT_GUIDELINES`` as the base.
        user_prompt: Article briefs and the target JSON schema for this call.
        json_schema: JSON Schema dict for output validation and constrained
            decoding hints.  ``None`` skips schema-guided generation.
        max_new_tokens: Token budget for the model response.

    Returns:
        Parsed dict matching the episode schema.

    Raises:
        ValueError: If JSON parsing fails after one retry.
        RuntimeError: If the provider call itself fails (propagated from
            ``_generate_with_provider()``).
    """
    # Attempt 1: standard generation
    raw = _generate_with_provider(system_prompt, user_prompt, max_new_tokens, json_schema)
    try:
        # Strip markdown code fences if the model wraps its output
        clean = re.sub(r"```json\s*|```", "", raw).strip()
        return json.loads(clean)
    except json.JSONDecodeError as e:
        _logger.warning(
            "generate_hostify_struct: JSON parse failed on first attempt (%s). "
            "Retrying with correction instruction.",
            e,
        )

    # Attempt 2: retry with an explicit correction instruction appended
    correction = (
        f"\n\nYour previous response was not valid JSON: {e}\n"
        "Return ONLY the raw JSON object. "
        "No markdown, no explanation, no extra text."
    )
    raw2 = _generate_with_provider(
        system_prompt,
        user_prompt + correction,
        max_new_tokens,
        json_schema,
    )
    try:
        clean2 = re.sub(r"```json\s*|```", "", raw2).strip()
        return json.loads(clean2)
    except json.JSONDecodeError as e2:
        raise ValueError(
            f"generate_hostify_struct: JSON parse failed after retry. "
            f"Last error: {e2}\nRaw output: {raw2[:500]}"
        ) from e2
