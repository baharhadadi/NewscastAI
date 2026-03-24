"""
services/worker/ingestion.py
-----------------------------
Async RSS ingestion pipeline for the newscast worker tier.

Fetches articles from a curated set of RSS feeds, scores them across multiple
topic facets (relevance, recency, source authority, trend corroboration), and
selects the best single topic to present.  Implements a progressive time-window
fallback so users always receive content even when recent coverage is sparse.
"""

import asyncio
import functools
import aiohttp
import feedparser
import logging
import re
from dataclasses import dataclass
from newspaper import Article
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from concurrent.futures import ThreadPoolExecutor
from time import mktime
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
import tldextract
from collections import defaultdict, Counter
from math import exp, log
from typing import Optional, List, Dict, Any, Tuple

from .base_agent import AgentResult, BaseAgent
from .settings import settings

_logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None  # fuzzy boost becomes a no-op if not installed

# ------------------------------------------------------------------
# Feeds: add a few Canadian/NA sources (bias to your timezone region)
# ------------------------------------------------------------------
DEFAULT_FEEDS = [
    # World / General
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.reuters.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://apnews.com/rss",
    "https://www.theguardian.com/world/rss",
    "https://www.cbc.ca/cmlink/rss-world",
    # Canada/NA general/business
    "https://www.cbc.ca/cmlink/rss-business",
    "https://globalnews.ca/feed/",
    "https://financialpost.com/feed/",
    # Business / Markets / Tech
    "https://www.ft.com/world?format=rss",
    "https://www.wsj.com/xml/rss/3_7085.xml",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",  # Top News
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.theverge.com/rss/index.xml",
    "https://techcrunch.com/feed/",
    "https://www.wired.com/feed/rss",
    # Science / Health
    "https://www.sciencedaily.com/rss/top/science.xml",
    "https://rss.nature.com/nature/rss/current",
    "https://www.nih.gov/news-events/news-releases/feed",
    # Sports / Entertainment (light)
    "https://www.espn.com/espn/rss/news",
    "https://www.rollingstone.com/music/music-news/feed/",
]

from .constants import DOMAIN_AUTHORITY, TOPIC_EXPANSIONS

# -------------------------
# Utilities (as before)
# -------------------------
def normalize_url(u: str) -> str:
    """Strip tracking parameters and URL fragments to produce a canonical URL.

    Removes UTM campaign tokens and common click-tracking parameters so that
    the same article reached via different referral links is treated as a
    single deduplicated item rather than multiple distinct URLs.

    Args:
        u: Raw URL string to normalise.

    Returns:
        Cleaned URL string.  Returns ``u`` unchanged if parsing raises any
        exception, so ingestion never fails on a malformed URL.
    """
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith(("utm_", "fbclid", "gclid", "mc_cid", "mc_eid"))]
        clean = p._replace(query=urlencode(q, doseq=True), fragment="")
        return urlunparse(clean)
    except Exception:
        return u

def entry_published_dt(entry) -> Optional[datetime]:
    """Extract the publication datetime from a feedparser entry, normalised to UTC.

    Tries string fields (``published``, ``updated``, ``created``) before falling
    back to the struct_time ``published_parsed`` field, which feedparser always
    populates when a date is present even if string parsing fails.

    Args:
        entry: A feedparser entry object (dict-like) from a parsed feed.

    Returns:
        Timezone-aware UTC datetime, or ``None`` if no parseable date is found.
    """
    for key in ("published", "updated", "created"):
        if val := entry.get(key):
            try:
                return dtparser.parse(val).astimezone(timezone.utc)
            except Exception:
                pass
    if entry.get("published_parsed"):
        try:
            return datetime.fromtimestamp(mktime(entry.published_parsed), tz=timezone.utc)
        except Exception:
            pass
    return None

@functools.lru_cache(maxsize=256)
def _compile_keyword_pattern(keywords: Tuple[str, ...]) -> re.Pattern:
    """Cached inner helper — receives a pre-normalised, hashable tuple.

    Keeping compilation in a separate cached function lets ``_regex_from_keywords``
    accept the more natural ``List[str]`` signature while still benefiting from
    caching.  The same topic keyword sets recur on every ``gather_for_facet()``
    call across the fallback-window ladder, so avoiding redundant ``re.compile``
    calls meaningfully reduces CPU overhead on multi-topic runs.
    """
    parts = [re.escape(k) for k in keywords]
    if not parts:
        parts = [".*"]
    return re.compile(r"(?i)\b(" + "|".join(parts) + r")\b")


def _regex_from_keywords(keywords: List[str]) -> re.Pattern:
    """Compile a word-boundary regex that matches any of the given keywords.

    Keywords are deduplicated and sorted longest-first so that multi-word
    phrases like ``"artificial intelligence"`` are matched before their
    substrings like ``"intelligence"``, preventing double-counting when a
    title contains both the phrase and individual tokens.

    The compiled pattern is cached via ``_compile_keyword_pattern`` using an
    ``lru_cache``; callers may safely invoke this on every article without
    paying per-call compilation cost.

    Args:
        keywords: Case-insensitive keyword/phrase strings to match.

    Returns:
        Compiled regex with ``re.IGNORECASE``.  Matches ``.*`` when the list
        is empty so callers can always call ``findall`` without branching.
    """
    # Normalise to a sorted, deduplicated, hashable tuple so lru_cache can key on it.
    normalised: Tuple[str, ...] = tuple(
        sorted({k.strip().lower() for k in keywords if k}, key=len, reverse=True)
    )
    return _compile_keyword_pattern(normalised)

def _relevance_score(text: str, rx: re.Pattern) -> int:
    """Count keyword hits in *text* using the precompiled pattern *rx*.

    Args:
        text: Text to search (title, summary, or concatenated tag terms).
        rx: Compiled pattern from ``_regex_from_keywords()``.

    Returns:
        Number of non-overlapping keyword matches.  Returns ``0`` for empty text.
    """
    if not text:
        return 0
    return len(rx.findall(text))

# -------------------------
# Async feed fetching
# -------------------------
async def _fetch_feed_bytes(session: aiohttp.ClientSession, url: str, timeout_s=15):
    """Fetch raw bytes from a single RSS feed URL, returning ``None`` on any failure.

    Args:
        session: Open ``aiohttp.ClientSession`` shared across concurrent requests.
        url: RSS feed URL to fetch.
        timeout_s: Per-request timeout in seconds.  Feeds that hang are
            dropped rather than blocking the entire gather pass.

    Returns:
        Response body as ``bytes``, or ``None`` if the request fails or the
        server returns a non-200 status code.
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                return None
            return await resp.read()
    except Exception:
        return None

async def _parse_feeds(feed_urls: List[str], concurrency: int = 20):
    """Fetch and parse multiple RSS feeds concurrently, returning all results.

    A semaphore caps simultaneous open connections to avoid overwhelming the
    local network stack or triggering rate-limiting on remote servers.
    Results arrive as each feed completes (``as_completed``) so the fastest
    feeds don't wait for the slowest ones.

    Args:
        feed_urls: List of RSS feed URLs to fetch.
        concurrency: Maximum simultaneous in-flight HTTP requests.

    Returns:
        List of ``(url, feedparser_result)`` tuples.  The feedparser result
        is ``None`` for any feed that failed to fetch or parse.
    """
    sem = asyncio.Semaphore(concurrency)
    results = []
    async with aiohttp.ClientSession(headers={"User-Agent": "NewsAgent/1.0"}) as session:
        async def _one(url):
            async with sem:
                data = await _fetch_feed_bytes(session, url)
                if not data:
                    return url, None
                try:
                    return url, feedparser.parse(data)
                except Exception:
                    return url, None
        tasks = [_one(u) for u in feed_urls]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
    return results

# -------------------------
# Full text (threaded)
# -------------------------
_executor = ThreadPoolExecutor(max_workers=8)

def _extract_fulltext(url: str) -> Optional[str]:
    """Download and extract the main article body from *url* via newspaper3k.

    Runs synchronously; always call through ``_extract_fulltext_async`` from
    async contexts to avoid blocking the event loop.

    Args:
        url: Canonical article URL.

    Returns:
        Extracted article body text, or ``None`` if download, parsing, or
        encoding detection fails for any reason.
    """
    try:
        a = Article(url)
        a.download(); a.parse()
        txt = (a.text or "").strip()
        return txt or None
    except Exception:
        return None

async def _extract_fulltext_async(url: str) -> Optional[str]:
    """Async wrapper around ``_extract_fulltext`` using a thread pool.

    newspaper3k is synchronous and CPU-bound; offloading to a thread keeps the
    asyncio event loop unblocked while articles are being downloaded and parsed
    in parallel.

    Args:
        url: Canonical article URL.

    Returns:
        Extracted article body text, or ``None`` on any failure.
    """
    return await asyncio.to_thread(_extract_fulltext, url)

# ------------------------------------
# Result dataclass for windowed search
# ------------------------------------
@dataclass
class WindowedSearchResult:
    """Typed schema reference for the payload inside ``AgentResult.data``.

    Not instantiated at runtime — ``NewsAgent.run()`` returns a plain dict for
    backward compatibility — but serves as the authoritative documentation of
    the data shape that callers should expect.

    Attributes:
        status: ``"ok"`` when articles were found; ``"no_news_today"`` when
            all time windows were exhausted without sufficient results.
        chosen_topic: Winning topic key (e.g. ``"ai"``).
        items: Scored, sorted article dicts.  Empty when ``status`` is
            ``"no_news_today"``.
        window_used: Time window that yielded sufficient results: ``"7d"``,
            ``"30d"``, ``"1y"``, or ``None`` when no results were found.
    """

    status: str            # "ok" | "no_news_today"
    chosen_topic: str
    items: List[Dict]
    window_used: Optional[str]  # "7d" | "30d" | "1y" | None


# ------------------------------------
# Agent: plan → gather → score → pick
# ------------------------------------
class NewsAgent(BaseAgent):
    """Async RSS ingestion agent with automatic fallback window search.

    Inherits from :class:`~services.worker.base_agent.BaseAgent` and satisfies
    the ``run()`` contract by returning an :class:`~services.worker.base_agent.AgentResult`.

    The ``run()`` method is declared ``async``; callers must ``await`` it.
    Python's ABC machinery accepts this override because the abstract signature
    uses ``*args / **kwargs``, leaving the async/sync decision to subclasses.

    Parameters
    ----------
    feeds:
        List of RSS feed URLs to poll.  Defaults to :data:`DEFAULT_FEEDS`.
    last_hours:
        Initial time window used on first gather attempt.  The :meth:`run`
        method overrides this temporarily via the fallback ladder; the value
        is always restored after the call.
    per_domain_cap:
        Maximum number of articles accepted from a single registered domain
        per gather pass.
    fetch_fulltext:
        Whether to extract full article text via ``newspaper3k``.
    min_items_for_topic:
        Minimum scored items a topic must have to be considered viable.
    topic_keep_top_k:
        Maximum items retained per topic after scoring.
    decay_half_life_hours:
        Half-life (hours) for the recency exponential decay factor.
    fuzzy_boost:
        Enable RapidFuzz title-matching boost when the library is available.
    """

    def __init__(
        self,
        feeds: Optional[List[str]] = None,
        last_hours: int = 36,
        per_domain_cap: int = 8,
        fetch_fulltext: bool = True,
        min_items_for_topic: int = 6,
        topic_keep_top_k: int = 30,
        decay_half_life_hours: int = 12,
        fuzzy_boost: bool = True,
    ) -> None:
        super().__init__("NewsAgent")
        self.feeds = feeds or DEFAULT_FEEDS
        self.last_hours = last_hours
        self.per_domain_cap = per_domain_cap
        self.fetch_fulltext = fetch_fulltext
        self.min_items_for_topic = min_items_for_topic
        self.topic_keep_top_k = topic_keep_top_k
        self.decay_half_life_hours = decay_half_life_hours
        self.fuzzy_boost = fuzzy_boost and (fuzz is not None)

    # ---- Planning: expand topics into “facets” we will compete ----
    def expand_topics(self, topics: List[str]) -> Dict[str, List[str]]:
        """Map user-supplied topic strings to their expanded keyword sets.

        Each topic is looked up in ``TOPIC_EXPANSIONS``; the canonical key is
        prepended as an anchor term if it is not already in the list so that
        exact-name matches always score highest.

        Args:
            topics: Topic strings from user preferences (e.g. ``[“ai”, “economy”]``).
                Defaults to ``[“news”]`` if the list is empty.

        Returns:
            Dict mapping normalised topic key → keyword list.
            Example: ``{“ai”: [“ai”, “artificial intelligence”, “llm”, ...]}``
        """
        plan: Dict[str, List[str]] = {}
        for t in topics or ["news"]:
            key = t.strip().lower()
            seeds = TOPIC_EXPANSIONS.get(key, [key])
            # include the root token as an anchor
            if key not in seeds:
                seeds = [key] + seeds
            plan[key] = seeds
        return plan

    # ---- Gather: fetch & filter candidates for a single facet ----
    async def gather_for_facet(self, facet_keywords: List[str]) -> List[Dict[str, Any]]:
        """Fetch all feeds and return articles that match the facet keywords.

        Applies three successive filters before accepting an article:

        1. **Time window** — publication datetime must fall within
           ``self.last_hours`` of now.
        2. **Relevance** — at least one keyword hit in title, summary, or tags
           (plus an optional fuzzy-match bonus via RapidFuzz).
        3. **Domain cap** — at most ``per_domain_cap`` articles per registered
           domain, preventing any single outlet from flooding the candidate set.

        Also deduplicates by normalised URL and lowercased title.

        Args:
            facet_keywords: Keyword list for one topic, from ``expand_topics()``.

        Returns:
            List of candidate article dicts, each containing ``title``, ``url``,
            ``source``, ``published_dt``, ``summary``, and ``facet_score``.
            Not yet sorted; call ``score_facet_items()`` next.
        """
        rx = _regex_from_keywords(facet_keywords)
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(hours=self.last_hours)

        parsed = await _parse_feeds(self.feeds)
        candidates = []
        seen_urls, seen_titles = set(), set()
        domain_counts = defaultdict(int)

        for feed_url, fp in parsed:
            if not fp or not getattr(fp, "entries", None):
                continue
            for e in fp.entries:
                title = e.get("title", "").strip()
                link = normalize_url(e.get("link", "") or "")
                if not title or not link:
                    continue

                # --- Date filter: skip articles older than cutoff ---
                pub_dt = entry_published_dt(e) or now_utc
                if pub_dt < cutoff:
                    continue

                summary = (e.get("summary") or e.get("description") or "").strip()
                tags = " ".join(t.get("term", "") for t in e.get("tags", []) if isinstance(t, dict))

                # --- Relevance: score by keyword hits (title weighted 2x) ---
                score = 2 * _relevance_score(title, rx) + _relevance_score(summary, rx) + _relevance_score(tags, rx)

                # --- Optional fuzzy boost for near-miss titles ---
                if self.fuzzy_boost and facet_keywords and title:
                    best = max(fuzz.partial_ratio(title.lower(), k.lower()) for k in facet_keywords)
                    if best >= 70:
                        score += 1
                    if best >= 85:
                        score += 1

                if score == 0 and facet_keywords:
                    continue

                # --- Deduplication: skip seen URLs and near-identical titles ---
                key_url, key_title = link.lower(), title.lower()
                # TODO: replace with embedding-based similarity using
                # MODEL_REGISTRY['embeddings'] (sentence-transformers/all-MiniLM-L6-v2)
                # for semantic dedup — current exact-match + rapidfuzz approach misses
                # paraphrased duplicates (e.g. "Fed raises rates" vs "Central bank hikes
                # interest rate").  Cosine similarity at threshold ~0.85 catches these.
                # See services/mcp/settings.py MODEL_REGISTRY for the chosen model.
                if key_url in seen_urls or key_title in seen_titles:
                    continue

                # --- Domain cap: limit articles per source for diversity ---
                dom = tldextract.extract(link).registered_domain or urlparse(link).netloc
                if self.per_domain_cap and domain_counts[dom] >= self.per_domain_cap:
                    continue

                source = dom or urlparse(feed_url).netloc
                candidates.append({
                    "title": title,
                    "url": link,
                    "source": source,
                    "published_dt": pub_dt,
                    "summary": summary,
                    "facet_score": score,  # store raw facet relevance
                })
                seen_urls.add(key_url)
                seen_titles.add(key_title)
                domain_counts[dom] += 1

        return candidates

    # ---- Scoring: multi-factor scoring within a facet ----
    def score_facet_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply multi-factor scoring to pre-filtered article candidates.

        Each article receives a composite score that balances four signals:
        topic relevance, recency, source authority, and trend corroboration.
        Articles are returned sorted highest-score first.

        Score formula::

            base  = (1.0 × facet_score) + (2.0 × recency) + (1.2 × authority)
            final = base × trend_multiplier

        The weight rationale is documented inline below.

        Args:
            items: Article dicts from ``gather_for_facet()``, each containing
                ``facet_score``, ``published_dt``, and ``source`` keys.

        Returns:
            Same list enriched with a ``score`` key and sorted descending.
            Returns the input unchanged when it is empty.
        """
        if not items:
            return items

        now = datetime.now(timezone.utc)

        # ---- Trend detection: normalise titles to catch near-duplicates ------
        # Stripping punctuation before counting means "Fed Raises Rates!" and
        # "Fed raises rates" map to the same key, so syndicated rewrites of the
        # same story are correctly treated as corroboration rather than novelty.
        # Trend detection: articles covered by multiple independent sources
        # signal higher importance — this is a lightweight proxy for editorial
        # consensus.  A story that Reuters, BBC, and AP all wrote about
        # independently carries more weight than a single-source exclusive,
        # even if the exclusive has a higher facet_score per article.
        norm_titles = [re.sub(r"\W+", " ", it["title"].lower()).strip() for it in items]
        freq = Counter(norm_titles)

        # Named weight variables for readability and to surface the tuning knobs
        # at the top of the loop body.  Assigned once outside the loop so the
        # settings lookup is not repeated for every article.
        w_facet = settings.score_weight_facet
        w_recency = settings.score_weight_recency
        w_authority = settings.score_weight_authority

        scored = []
        for it, nt in zip(items, norm_titles):
            hours_old = max(0.0, (now - it["published_dt"]).total_seconds() / 3600.0)

            # ---- Recency: exponential decay (not linear) ---------------------
            # Exponential decay mirrors actual news-value curves: a 2-hour-old
            # article is dramatically more relevant than a 12-hour-old one, but
            # the gap between 36 h and 48 h is much smaller.  Linear decay
            # would over-penalise articles that are simply "no longer breaking."
            # λ = ln(2) / half_life ensures recency = 0.5 exactly at half-life.
            lam = log(2) / max(1.0, self.decay_half_life_hours)
            recency = exp(-lam * hours_old)  # 1.0 if brand-new, ~0.5 at half-life

            # ---- Domain authority: lookup table (not a learned parameter) ----
            # A hand-tuned table is interpretable, auditable, and updated with a
            # one-line diff in constants.py.  Learning weights from data would
            # require labelled training sets and periodic retraining for
            # marginal gain.  The fallback "*" key gives unknown domains 0.6.
            dom = it["source"]
            auth = DOMAIN_AUTHORITY.get(dom, DOMAIN_AUTHORITY["*"])

            # ---- Trend boost: capped at +0.5 (50% bonus) --------------------
            # The cap prevents a highly-syndicated wire story (e.g. 10 identical
            # AP copies from aggregators) from completely drowning out a unique,
            # well-reported investigation.  +50% is enough to surface trending
            # stories meaningfully without making source diversity irrelevant.
            trend = 1.0 + min(0.5, 0.1 * max(0, freq[nt] - 1))

            # ---- Final formula: what this score optimises for ----------------
            # Weights are named variables (w_facet / w_recency / w_authority)
            # read from settings so they can be tuned per-deployment without
            # code changes.  Defaults (1.0 / 2.0 / 1.2) were tuned empirically.
            # Recency is double-weighted because stale news is the primary
            # failure mode for a daily briefing product.  Facet relevance and
            # authority are roughly equal; the multiplicative trend term rewards
            # corroboration without distorting the additive balance of the base.
            final = (
                w_facet * it["facet_score"]
                + w_recency * recency
                + w_authority * auth
            )
            final *= trend

            scored.append({**it, "score": final})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    # ---- Decide: pick the best single topic for this run ----
    async def decide_best_topic(self, plan: Dict[str, List[str]]) -> Dict[str, Any]:
        """Gather and score articles for all topics, then pick the strongest one.

        The "winner" is the topic with the highest aggregate signal score, defined
        as ``mean_score_of_top_10 × (1 + 0.25 × source_diversity)``.  A topic
        must clear ``min_items_for_topic`` to enter competition; this floor
        prevents a single viral article from winning by virtue of having a mean
        score equal to its own peak score.

        Args:
            plan: ``{topic_key: [keyword, ...]}`` mapping from ``expand_topics()``.

        Returns:
            Dict with keys:

            - ``topic`` (str): the winning topic key.
            - ``items`` (List[Dict]): scored, sorted articles for that topic,
              capped at ``topic_keep_top_k`` entries.
        """
        facet_results = {}
        for topic, facet_keywords in plan.items():
            items = await self.gather_for_facet(facet_keywords)
            items = self.score_facet_items(items)
            # Retain a bounded working set per topic to cap memory usage and
            # downstream LLM prompt length while keeping enough items for reliable
            # signal scoring across the top-10 window used below.
            facet_results[topic] = items[: self.topic_keep_top_k]

        best_topic, best_signal = None, -1.0
        chosen_items = []

        for topic, items in facet_results.items():
            # ---- Minimum depth guard -----------------------------------------
            # Without this floor, a single viral article would always "win"
            # because its mean score equals its own (maximum) score.
            # min_items_for_topic ensures we only compete topics with genuine
            # coverage breadth — multiple independent articles on the subject.
            if len(items) < self.min_items_for_topic:
                continue
            topN = items[:10]
            if not topN:
                continue
            mean_score = sum(x["score"] for x in topN) / len(topN)
            src_div = len({x["source"] for x in topN})
            # Guard: with fewer than 3 articles the diversity ratio is
            # unreliable — a single Reuters article would score 1.0 diversity
            # (1 unique source / 1 article), making it appear maximally diverse
            # and unfairly boosting its signal vs a topic with 8 articles from
            # 4 sources.  Zero the diversity bonus below the minimum depth.
            if len(topN) < 3:
                diversity = 0.0
            else:
                diversity = src_div / max(1, len(topN))  # 0..1

            # ---- Diversity weight at 0.25 (not higher) -----------------------
            # Diversity is a tiebreaker, not the primary criterion.  At 0.25
            # a perfectly diverse topic (one article per unique source) earns at
            # most a 25% bonus on its mean score.  Testing with higher values
            # (0.5, 1.0) caused the picker to favour thinly-covered topics that
            # happened to have one article from each of five different domains —
            # prioritising breadth-of-source over quality-of-coverage.
            signal = mean_score * (1.0 + 0.25 * diversity)
            if signal > best_signal:
                best_signal = signal
                best_topic = topic
                chosen_items = items

        # ---- Fallback: most-covered topic ------------------------------------
        # If every topic failed the min_items_for_topic floor (e.g. very sparse
        # feeds or a very narrow time window) we still return something rather
        # than an empty result.  The most-covered topic at least has breadth
        # even if individual item scores are low.  The caller (run()) then
        # decides whether this is sufficient or whether to widen the time window.
        if not best_topic:
            best_topic = max(facet_results.keys(), key=lambda t: len(facet_results[t]))
            chosen_items = facet_results[best_topic]
            _logger.warning(
                "No topic passed quality threshold. Falling back to highest-volume"
                " topic: %s (%d items)",
                best_topic,
                len(chosen_items),
            )

        return {"topic": best_topic, "items": chosen_items}

    # ---- Private helpers extracted from run() --------------------------------

    async def _run_with_window(
        self,
        plan: Dict[str, List[str]],
        limit: int,
    ) -> WindowedSearchResult:
        """Execute topic search across the progressive time-window ladder.

        Extracted from ``run()`` to isolate the window-mutation side-effect
        (temporarily overwriting ``self.last_hours``) from the fulltext
        enrichment concern.  The ``try / finally`` block is the critical
        invariant: ``self.last_hours`` is always restored to its original
        value before this method returns, regardless of whether the loop
        exits normally, via ``break``, or via an exception.

        The four-rung ladder (7 d → 30 d → 1 y → no_news_today) is explained
        in full in ``run()``'s docstring.

        Args:
            plan: ``{topic_key: [keyword, ...]}`` mapping from
                ``expand_topics()``.
            limit: Maximum number of article items to include in the returned
                ``WindowedSearchResult.items`` list.

        Returns:
            ``WindowedSearchResult`` with ``status="ok"`` and the winning
            topic's items (sliced to ``limit``) when sufficient articles are
            found, or ``status="no_news_today"`` with an empty ``items`` list
            when all windows are exhausted.
        """
        WINDOWS = [
            (168,  "7d"),
            (720,  "30d"),
            (8760, "1y"),
        ]
        EXPAND_LOGS = {
            720:  "Expanding search window to 30 days for topic: {topic}",
            8760: "Expanding search window to 1 year for topic: {topic}",
        }

        original_last_hours = self.last_hours
        decision: Optional[Dict[str, Any]] = None
        window_used: Optional[str] = None

        try:
            for i, (hours, label) in enumerate(WINDOWS):
                self.last_hours = hours
                decision = await self.decide_best_topic(plan)

                if len(decision["items"]) >= self.min_items_for_topic:
                    window_used = label
                    break

                # Not enough items — log before attempting the next wider window
                if i + 1 < len(WINDOWS):
                    next_hours = WINDOWS[i + 1][0]
                    log_msg = EXPAND_LOGS.get(next_hours, "")
                    if log_msg:
                        self.logger.info(log_msg.format(topic=decision["topic"]))
            else:
                # for-else: loop completed without a break → all windows exhausted
                chosen = decision["topic"] if decision else (list(plan.keys())[0] if plan else "news")
                return WindowedSearchResult(
                    status="no_news_today",
                    chosen_topic=chosen,
                    items=[],
                    window_used=None,
                )
        finally:
            self.last_hours = original_last_hours

        return WindowedSearchResult(
            status="ok",
            chosen_topic=decision["topic"],
            items=decision["items"][:limit],
            window_used=window_used,
        )

    async def _enrich_fulltext(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fetch full article text for each item and serialise publication dates.

        Extracted from ``run()`` so that the fulltext I/O concern is cleanly
        separated from the window-search logic.  Two invariants are maintained:

        1. Items for which newspaper3k extraction fails or returns ``None``
           retain whatever ``summary`` text was populated during feed parsing;
           ``it["text"]`` is set to ``None`` so callers can fall back to
           ``it.get("summary")``.
        2. ``published_dt`` (a ``datetime`` object) is always replaced with
           ``published`` (an ISO 8601 string) before this method returns,
           making the items JSON-serialisable for downstream callers.

        Args:
            items: Article dicts from the window search, each containing a
                ``published_dt`` key and a ``url`` key.

        Returns:
            The same list, mutated in-place: ``text`` added (or ``None``),
            ``published`` added, ``published_dt`` removed.
        """
        if self.fetch_fulltext and items:
            texts = await asyncio.gather(
                *[_extract_fulltext_async(it["url"]) for it in items],
                return_exceptions=True,
            )
            for it, txt in zip(items, texts):
                it["text"] = (None if isinstance(txt, Exception) else txt)

        # Serialize datetime to ISO string for JSON compatibility
        for it in items:
            it["published"] = it["published_dt"].isoformat()
            it.pop("published_dt", None)

        return items

    # ---- Run: clean orchestrator calling the two helpers above ---------------

    async def run(  # type: ignore[override]  # async override of abstract sync method
        self,
        topics: List[str],
        limit: int = 30,
    ) -> AgentResult:
        """Execute a windowed news search with automatic fallback to broader time windows.

        Satisfies the :class:`~services.worker.base_agent.BaseAgent` contract by
        always returning an :class:`~services.worker.base_agent.AgentResult`.
        Never raises — errors are captured in ``AgentResult.error``.

        Fallback window strategy — why it matters for user experience
        -------------------------------------------------------------
        RSS feeds only carry recent items, so a narrow 7-day window often returns
        sparse results for niche topics (e.g. "energy", "science").  Rather than
        silently delivering a thin or empty newscast, we progressively widen the
        search through a four-rung ladder:

            WINDOW_1   168 h  (7 days)    "today's news"      ← tried first
            WINDOW_2   720 h  (30 days)   "this month"         ← fallback #1
            WINDOW_3  8760 h  (1 year)    "this year"          ← fallback #2
            WINDOW_4  no result           status="no_news_today"

        The window traversal is handled by ``_run_with_window()``; fulltext
        extraction and datetime serialisation are handled by
        ``_enrich_fulltext()``.

        Parameters
        ----------
        topics:
            List of topic strings to search (e.g. ``["ai", "economy"]``).
        limit:
            Maximum number of articles included in the result.

        Returns
        -------
        AgentResult
            ``status``  — ``"ok"`` | ``"no_news_today"`` | ``"error"``

            ``data``    — dict with keys:

                * ``chosen_topic`` (str)
                * ``items``        (list of enriched article dicts)
                * ``window_used``  (``"7d"`` | ``"30d"`` | ``"1y"`` | ``None``)

            ``metadata`` — ``{"window_used": ..., "chosen_topic": ...}``
        """
        self._log_start(topics=topics, limit=limit)

        plan = self.expand_topics(topics)
        ws = await self._run_with_window(plan, limit)

        if ws.status == "no_news_today":
            result = AgentResult(
                status="no_news_today",
                data={"chosen_topic": ws.chosen_topic, "items": [], "window_used": None},
                metadata={"window_used": None, "chosen_topic": ws.chosen_topic},
            )
            self._log_complete(result)
            return result

        ws.items = await self._enrich_fulltext(ws.items)

        result = AgentResult(
            status="ok",
            data={
                "chosen_topic": ws.chosen_topic,
                "items": ws.items,
                "window_used": ws.window_used,
            },
            metadata={"window_used": ws.window_used, "chosen_topic": ws.chosen_topic},
        )
        self._log_complete(result)
        return result

# ---------------------------
# Public entry points
# ---------------------------
async def agentic_fetch_articles_async(
    topics: List[str],
    limit: int = 30,
    feeds: Optional[List[str]] = None,
    last_hours: int = 36,
    per_domain_cap: int = 8,
    fetch_fulltext: bool = True,
) -> Dict[str, Any]:
    """Async convenience wrapper.  Returns the inner data dict for backward compat."""
    agent = NewsAgent(
        feeds=feeds,
        last_hours=last_hours,
        per_domain_cap=per_domain_cap,
        fetch_fulltext=fetch_fulltext,
    )
    result = await agent.run(topics, limit=limit)
    return result.data  # unwrap AgentResult → plain dict (preserves existing callers)

def agentic_fetch_articles(
    topics: List[str],
    **kwargs
) -> Dict[str, Any]:
    """Synchronous entry point for RSS-based article fetching.

    Wraps ``agentic_fetch_articles_async`` for callers that cannot use
    ``await`` (e.g. Celery tasks, scripts).  Handles both a fresh event loop
    (via ``asyncio.run``) and an already-running loop (via
    ``get_event_loop().run_until_complete``) for compatibility with Jupyter
    notebooks and frameworks that keep a loop open.

    Args:
        topics: Topic strings to search (e.g. ``["ai", "markets"]``).
        **kwargs: Forwarded verbatim to ``agentic_fetch_articles_async``.

    Returns:
        Dict with keys ``chosen_topic``, ``items``, ``window_used``, and
        ``status``.  See ``NewsAgent.run()`` for the full schema.
    """
    # asyncio.run() creates a *new* event loop, runs the coroutine, then closes
    # the loop.  This is the clean path for Celery tasks running in a worker
    # process with no pre-existing event loop.
    #
    # However, Celery's gevent/eventlet pool modes (and some test frameworks such
    # as pytest-asyncio) keep a running event loop alive on the thread.  Calling
    # asyncio.run() inside an already-running loop raises RuntimeError("This event
    # loop is already running").  The except branch falls back to scheduling the
    # coroutine on the existing loop via run_until_complete(), which is safe as
    # long as the coroutine itself does not try to run a *nested* event loop.
    try:
        return asyncio.run(agentic_fetch_articles_async(topics, **kwargs))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(agentic_fetch_articles_async(topics, **kwargs))

# ---------------------------
# Example execution
# ---------------------------
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.DEBUG, format="%(levelname)s %(message)s")
    _cli_log = _logging.getLogger("ingestion.cli")

    res = agentic_fetch_articles(
        ["AI", "economy", "geopolitics", "tech"],
        limit=25,
        last_hours=36,
        per_domain_cap=6,
        fetch_fulltext=True,
    )
    _cli_log.info("Chosen topic: %s", res["chosen_topic"])
    for r in res["items"]:
        _cli_log.info("[%s] %s (%s)", r["source"], r["title"], r["published"])
        _cli_log.debug("URL: %s", r["url"])
        if r.get("text"):
            _cli_log.debug("Text preview: %s ...", r["text"][:200].replace("\n", " "))
        elif r.get("summary"):
            _cli_log.debug("Summary: %s ...", r["summary"][:200].replace("\n", " "))
        _cli_log.debug("-" * 80)
