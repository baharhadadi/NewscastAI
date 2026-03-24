# services/worker/agent_search.py
"""
services/worker/agent_search.py
--------------------------------
SearxNG-backed agentic news search.  Complements the RSS-based NewsAgent by
fetching live web-search results, then delegating topic selection and article
summarisation to an LLM in a single round-trip.  Use this path when RSS feeds
have not yet propagated a breaking story or when broader web coverage is needed.
"""

import aiohttp
import asyncio
import json
import logging
import re
import tldextract
from datetime import datetime, timezone
from dateutil import parser as dtp
from typing import Dict, List, Any, Optional, Tuple

from .settings import settings
from .constants import TOPIC_EXPANSIONS

# Per-request timeout for MCP LLM chat calls.  Structured generation can take
# several seconds on a cold vLLM instance; 60 s is generous but bounded.
_CHAT_TIMEOUT_S: int = 60

_logger = logging.getLogger(__name__)


async def _mcp_chat_json(system_msg: str, user_msg: str) -> str:
    """Send a structured chat prompt to the MCP LLM endpoint and return raw JSON.

    Implemented inline here (rather than in ``summarizer_client``) because this
    module is fully async and requires a true non-blocking HTTP call.  Using
    ``aiohttp`` — already a dependency of this module — keeps the event loop
    unblocked during the (potentially multi-second) LLM round-trip.

    Args:
        system_msg: System-role prompt string.
        user_msg: User-role prompt string containing candidate articles and schema.

    Returns:
        Raw JSON string from the MCP response (not yet parsed).

    Raises:
        aiohttp.ClientResponseError: On a non-2xx MCP response.
        asyncio.TimeoutError: If the request exceeds ``_CHAT_TIMEOUT_S``.
        KeyError: If the MCP response JSON is missing the ``content`` field.
    """
    url = f"{settings.mcp_url.rstrip('/')}/chat_json"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"system": system_msg, "user": user_msg},
            timeout=aiohttp.ClientTimeout(total=_CHAT_TIMEOUT_S),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json(content_type=None)
            return payload["content"]


def _facet_query(topic: str) -> str:
    """Build an OR-joined SearxNG query string for the given topic.

    Uses TOPIC_EXPANSIONS from constants.py so the RSS ingestion path and the
    SearxNG search path share the same keyword vocabulary.
    """
    terms = TOPIC_EXPANSIONS.get(topic.lower().strip(), [topic])
    # simple OR-joined query helps recall without hand-coded ranking
    return " OR ".join(terms)

async def searx_search(
    query: str,
    *,
    max_results: int = 20,
    time_range: str = "day",
    language: str = "en",
    timeout_s: int = 20,
) -> List[Dict[str, Any]]:
    """Fetch news search results from SearxNG for the given query.

    Args:
        query: Search string (typically an OR-joined keyword list).
        max_results: Maximum number of results to return.
        time_range: SearxNG time_range filter: ``"day"``, ``"week"``,
            ``"month"``, or ``"year"``.
        language: BCP-47 language code forwarded to SearxNG.
        timeout_s: Per-request timeout in seconds.

    Returns:
        List of result dicts with ``title``, ``url``, ``snippet``,
        ``published`` (ISO8601 or ``None``), and ``source`` keys.
        Returns an empty list on any network or parse error so the caller
        can degrade gracefully.
    """
    params = {
        "q": query,
        "format": "json",
        "time_range": time_range,   # 'day'|'week'|'month'|'year'
        "language": language,
    }
    url = f"{settings.searxng_url.rstrip('/')}/search"
    out: List[Dict[str, Any]] = []
    data: Dict[str, Any] = {}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as r:
                r.raise_for_status()
                try:
                    data = await r.json(content_type=None)
                except (json.JSONDecodeError, aiohttp.ContentTypeError) as exc:
                    _logger.warning("SearxNG returned non-JSON response for %r: %s", query, exc)
                    return out
    except asyncio.TimeoutError:
        _logger.warning("SearxNG request timed out after %ds for query %r", timeout_s, query)
        return out
    except aiohttp.ClientError as exc:
        _logger.warning("SearxNG request failed for query %r: %s", query, exc)
        return out

    for it in (data.get("results") or [])[:max_results]:
        u: Optional[str] = it.get("url")
        title: str = (it.get("title") or "").strip()
        snippet: str = re.sub(r"\s+", " ", (it.get("content") or "").strip())
        dt: Optional[str] = it.get("publishedDate") or it.get("published") or it.get("date")
        try:
            published: Optional[str] = dtp.parse(dt).astimezone(timezone.utc).isoformat() if dt else None
        except Exception:
            published = None
        dom = tldextract.extract(u or "")
        source: str = f"{dom.domain}.{dom.suffix}" if dom.suffix else dom.domain
        if u and title:
            out.append({"title": title, "url": u, "snippet": snippet, "published": published, "source": source})
    return out

def _build_chooser_prompt(
    topics: List[str],
    results: List[Dict[str, Any]],
    now_iso: str,
) -> Tuple[str, str]:
    """Build the system and user prompts for the single-LLM chooser call.

    Extracted from ``choose_and_summarize`` to isolate prompt-engineering text
    from the function's control flow.  This makes it straightforward to iterate
    on prompt wording, update the output JSON schema, or unit-test the prompt
    content without spinning up an aiohttp session.

    Invariant: the output schema embedded in the user prompt must stay in sync
    with the dict shape that callers of ``choose_and_summarize`` expect.  If
    the schema changes here, update the docstring of ``choose_and_summarize``
    accordingly.

    Args:
        topics: Ordered list of user topic strings (used in the user prompt).
        results: SearxNG result dicts to include as JSON context for the LLM.
        now_iso: Current UTC timestamp in ISO 8601 format, used so the LLM
            can reason about recency relative to a fixed reference point.

    Returns:
        ``(system_msg, user_msg)`` tuple ready to pass to ``_mcp_chat_json()``.
    """
    results_json = json.dumps(results, ensure_ascii=False)

    system_msg = (
        "You are a concise newsroom researcher for a daily audio briefing. "
        "Pick ONE topic from the user's list that has the most timely, relevant coverage TODAY, "
        "then pick ONE article that is recent, credible, and substantive. "
        "Return STRICT JSON matching the schema, with no extra text. "
        "Prefer wire services / major outlets and very recent timestamps. "
        "Penalize clickbait/opinion-only."
    )

    user_msg = f"""
Now: {now_iso} (America/Toronto). Consider recency relative to this.

User topics (priority order): {", ".join(topics)}

Candidate search results (from SearxNG):
{results_json}

Output JSON schema:
{{
  "chosen_topic": "string (one of the user topics, lowercased)",
  "chosen_url": "string (https URL)",
  "chosen_title": "string",
  "published": "ISO8601 or null",
  "source": "string (registered domain)",
  "why": ["bullet 1","bullet 2","bullet 3"],
  "summary": {{
    "lede": "1-2 sentences, plain English",
    "key_points": ["3-6 compact bullets focused on what happened, why it matters, what's next"],
    "context": "1 short paragraph of background",
    "implications": ["0-3 bullets of concrete impacts"]
  }},
  "script_30s": "A natural 25-35 second anchor-style read suitable for text-to-speech"
}}

Rules:
- If multiple results are similar, prefer the most recent from Reuters/AP/BBC/FT/WSJ/CBC/etc.
- If a result lacks a date, judge by snippet & outlet; still pick if strongest.
- ABSOLUTELY NO text outside the JSON.
""".strip()

    return system_msg, user_msg


async def choose_and_summarize(topics: List[str]) -> Dict[str, Any]:
    """Gather web search results for all topics and delegate topic selection
    and summarisation to a single LLM call.

    Design rationale — single LLM round-trip
    -----------------------------------------
    A naive implementation would use two steps: (1) a ranking call to pick the
    best topic and article, then (2) a summarisation call to write the script.
    This function intentionally collapses both into one structured prompt
    because:

    - The LLM's ranking decision is inherently intertwined with its summary
      quality judgement.  Separating them forces the model to re-read the same
      context twice.
    - One round-trip halves the latency and API cost relative to two sequential
      calls.
    - The strict JSON output schema guarantees the result is directly usable
      by the worker tier without an intermediate parsing step.

    Output JSON schema
    ------------------
    The returned dict mirrors the schema enforced by the LLM prompt::

        {
          "chosen_topic":  str,          # one of the user topics, lowercased
          "chosen_url":    str,          # https URL of the selected article
          "chosen_title":  str,
          "published":     str | None,   # ISO8601 or null
          "source":        str,          # registered domain (e.g. "reuters.com")
          "why":           List[str],    # 3 bullets justifying the pick
          "summary": {
            "lede":         str,         # 1-2 sentence summary
            "key_points":   List[str],   # 3-6 compact bullets
            "context":      str,         # 1 paragraph of background
            "implications": List[str],   # 0-3 impact bullets
          },
          "script_30s":    str,          # 25-35 second anchor-style read for TTS
        }

    When to use this vs ``NewsAgent``
    -----------------------------------
    ``choose_and_summarize`` is the **SearxNG / web-search path**.  Use it
    when:

    - Breaking news has not yet propagated to RSS feeds.
    - Broader web coverage is needed beyond the curated ``DEFAULT_FEEDS`` list.
    - A single high-quality article with a 30-second script is sufficient
      (e.g. short-form briefing mode).

    ``NewsAgent.run()`` is the **RSS path**.  Use it for standard daily
    briefings where multi-article depth, source diversity, and trend
    corroboration signals are important.

    Args:
        topics: Ordered list of topic strings from user preferences.

    Returns:
        Parsed dict matching the output JSON schema described above.

    Raises:
        json.JSONDecodeError: If the LLM returns invalid JSON.
        aiohttp.ClientResponseError: If the MCP ``_mcp_chat_json``
            HTTP call returns a non-2xx status.
    """
    # 1) Gather candidates across topics
    all_results: List[Dict[str, Any]] = []
    per_topic = max(8, min(15, settings.news_max_results // max(1, len(topics))))
    for t in topics:
        q = _facet_query(t)
        rs = await searx_search(q, max_results=per_topic, time_range=settings.news_time_range, language=settings.news_lang)
        # Tag each with the originating topic for the model to consider
        for r in rs:
            r["_topic"] = t.lower()
        all_results.extend(rs)

    # Trim to keep prompts small
    all_results = all_results[: settings.news_max_results]

    # 2) Build the single agentic prompt and let the LLM decide & summarize
    now_iso = datetime.now(timezone.utc).isoformat()
    system_msg, user_msg = _build_chooser_prompt(topics, all_results, now_iso)
    raw = await _mcp_chat_json(system_msg=system_msg, user_msg=user_msg)
    data = json.loads(raw)  # raise if invalid; caught in caller
    return data
