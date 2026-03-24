"""
CrewAI-based news retrieval crew for NewscastAI.

Replaces the monolithic NewsAgent with four specialized agents
that collaborate to turn user topics into a ranked, validated
slate of articles ready for script generation.

Why CrewAI here:
  The four retrieval responsibilities — query expansion,
  feed retrieval, credibility ranking, and editorial
  selection — have genuinely different goals and failure
  modes. CrewAI's role-based prompting and sequential task
  delegation lets each agent focus on one concern, making
  the pipeline easier to debug and each stage independently
  improvable without touching the others.

Why not LangGraph here:
  LangGraph excels at stateful pipelines with conditional
  routing and loops (see hostify_graph.py for that pattern).
  Retrieval is a directed sequence with no feedback loops —
  CrewAI's sequential process is the simpler, cleaner fit.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel

from .ingestion import (
    DEFAULT_FEEDS,
    DOMAIN_AUTHORITY,
    TOPIC_EXPANSIONS,
    WindowedSearchResult,
    _extract_fulltext_async,
    _parse_feeds,
    _regex_from_keywords,
    _relevance_score,
    entry_published_dt,
    normalize_url,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured inter-agent communication
# ---------------------------------------------------------------------------

class QueryPlan(BaseModel):
    """Output of QueryGeneratorAgent — one entry per user topic."""
    topic_queries: dict[str, list[str]]
    # e.g. {"ai": ["artificial intelligence", "llm", "openai"], ...}


class RawArticle(BaseModel):
    """A single candidate article before scoring."""
    title: str
    url: str
    source: str
    published: str        # ISO8601
    snippet: str
    topic: str            # which topic facet matched
    facet_score: int      # raw keyword hit count


class ScoredArticle(RawArticle):
    """RawArticle with multi-factor ranking score attached."""
    recency_score: float
    authority_score: float
    trend_boost: float
    final_score: float


class RetrievalResult(BaseModel):
    """Final output of the crew — what tasks.py receives."""
    chosen_topic: str
    items: list[dict[str, Any]]
    window_used: str | None   # "7d" | "30d" | "1y" | None
    status: str               # "ok" | "no_news_today"


# ---------------------------------------------------------------------------
# CrewAI Tools — these are the capabilities agents can invoke
# ---------------------------------------------------------------------------

class FeedFetcherTool(BaseTool):
    """
    Fetches and parses RSS feeds for a list of keyword facets.
    Returns raw candidate articles matching at least one keyword.
    Called by RetrieverAgent.
    """
    name: str = "feed_fetcher"
    description: str = (
        "Fetch RSS feeds and return articles matching the given "
        "keywords. Input: JSON with 'keywords' (list of str) and "
        "'last_hours' (int). Output: list of raw article dicts."
    )

    def _run(self, keywords: list[str], last_hours: int = 168) -> list[dict]:
        """Synchronous wrapper — runs the async feed fetch."""
        return asyncio.run(self._fetch(keywords, last_hours))

    async def _fetch(
        self, keywords: list[str], last_hours: int
    ) -> list[dict]:
        rx = _regex_from_keywords(keywords)
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(hours=last_hours)

        parsed = await _parse_feeds(DEFAULT_FEEDS)
        candidates = []
        seen_urls: set[str] = set()

        for _, fp in parsed:
            if not fp or not getattr(fp, "entries", None):
                continue
            for e in fp.entries:
                title = e.get("title", "").strip()
                link = normalize_url(e.get("link", "") or "")
                if not title or not link or link.lower() in seen_urls:
                    continue

                pub_dt = entry_published_dt(e) or now_utc
                if pub_dt < cutoff:
                    continue

                snippet = (
                    e.get("summary") or e.get("description") or ""
                ).strip()[:500]

                score = (
                    2 * _relevance_score(title, rx)
                    + _relevance_score(snippet, rx)
                )
                if score == 0:
                    continue

                import tldextract
                from urllib.parse import urlparse
                dom = (
                    tldextract.extract(link).registered_domain
                    or urlparse(link).netloc
                )

                seen_urls.add(link.lower())
                candidates.append({
                    "title": title,
                    "url": link,
                    "source": dom,
                    "published": pub_dt.isoformat(),
                    "snippet": snippet,
                    "facet_score": score,
                })

        return candidates


class CredibilityCheckerTool(BaseTool):
    """
    Scores each article's source against the DOMAIN_AUTHORITY
    registry and flags low-credibility sources.
    Called by RankerAgent.
    """
    name: str = "credibility_checker"
    description: str = (
        "Score articles by source credibility using a domain "
        "authority registry. Input: list of article dicts. "
        "Output: same list with 'authority_score' field added."
    )

    def _run(self, articles: list[dict]) -> list[dict]:
        for a in articles:
            dom = a.get("source", "")
            a["authority_score"] = DOMAIN_AUTHORITY.get(
                dom, DOMAIN_AUTHORITY["*"]
            )
        return articles


class RecencyScorerTool(BaseTool):
    """
    Applies exponential decay to score article recency.
    Half-life of 12 hours: an article published 12h ago
    scores 0.5, 24h ago scores ~0.25.
    Called by RankerAgent.
    """
    name: str = "recency_scorer"
    description: str = (
        "Score articles by recency using exponential decay "
        "(half-life=12h). Input: list of article dicts with "
        "'published' ISO field. Output: articles with "
        "'recency_score' added."
    )

    def _run(
        self,
        articles: list[dict],
        half_life_hours: float = 12.0,
    ) -> list[dict]:
        from math import exp, log
        now = datetime.now(timezone.utc)
        lam = log(2) / max(1.0, half_life_hours)
        for a in articles:
            try:
                from dateutil import parser as dtp
                pub = dtp.parse(a["published"]).astimezone(timezone.utc)
                hours_old = max(
                    0.0,
                    (now - pub).total_seconds() / 3600.0
                )
                a["recency_score"] = exp(-lam * hours_old)
            except Exception:
                a["recency_score"] = 0.5  # neutral if unparseable
        return articles


# ---------------------------------------------------------------------------
# The four agents
# ---------------------------------------------------------------------------

def make_query_generator_agent() -> Agent:
    """
    Expands user topics into rich keyword facets for feed search.

    Uses TOPIC_EXPANSIONS as its knowledge base. For unknown topics,
    generates semantically related terms rather than using the bare
    topic string — improving recall from RSS keyword matching.
    """
    return Agent(
        role="Query Generator",
        goal=(
            "Expand user-provided topics into comprehensive keyword "
            "facets that maximize relevant article recall from RSS feeds."
        ),
        backstory=(
            "You are a news desk researcher who knows how topics are "
            "labelled across different publications. Given a topic like "
            "'AI', you know to search for 'artificial intelligence', "
            "'LLM', 'foundation models', and 'machine learning' — not "
            "just the bare acronym. You use the TOPIC_EXPANSIONS "
            "registry and extend it with your own judgment for topics "
            "not in the registry."
        ),
        verbose=False,
        allow_delegation=False,
    )


def make_retriever_agent() -> Agent:
    """
    Fetches candidate articles from RSS feeds using the FeedFetcherTool.

    Runs one fetch per topic facet concurrently, caps results per
    domain to prevent any single source dominating the slate, and
    deduplicates by URL before passing candidates downstream.
    """
    return Agent(
        role="News Retriever",
        goal=(
            "Retrieve a diverse, recent set of candidate articles "
            "from RSS feeds for each topic facet."
        ),
        backstory=(
            "You are a wire room editor who monitors dozens of RSS "
            "feeds simultaneously. You know that Reuters and AP break "
            "stories first, but The Verge covers tech angles that wires "
            "miss. You retrieve broadly and let the ranker filter."
        ),
        tools=[FeedFetcherTool()],
        verbose=False,
        allow_delegation=False,
    )


def make_ranker_agent() -> Agent:
    """
    Scores candidates on recency, authority, and relevance.

    Applies exponential decay for recency (12h half-life),
    domain authority lookup, and trend detection (cross-source
    coverage of the same story signals importance). Returns a
    ranked list with all score components preserved for auditability.
    """
    return Agent(
        role="News Ranker",
        goal=(
            "Score and rank candidate articles by recency, "
            "source credibility, and cross-source trend signal. "
            "Return a ranked list with score components preserved."
        ),
        backstory=(
            "You are a quantitative news analyst. You know that a "
            "Reuters article from 2 hours ago beats a blog post from "
            "this morning, and that a story covered by AP, BBC, and "
            "CBC simultaneously is almost certainly significant. "
            "You score transparently so your rankings are auditable."
        ),
        tools=[CredibilityCheckerTool(), RecencyScorerTool()],
        verbose=False,
        allow_delegation=False,
    )


def make_editorial_agent() -> Agent:
    """
    Selects the best topic and final article slate from ranked candidates.

    Applies the min_items threshold, source diversity weighting,
    and the fallback window ladder (7d → 30d → 1y → no_news_today).
    Acts as the final editorial gate before articles enter the
    script generation pipeline.
    """
    return Agent(
        role="Editorial Selector",
        goal=(
            "Select the single best topic and a diverse slate of "
            "top articles for today's episode. Apply the fallback "
            "window ladder if no topic has sufficient fresh coverage."
        ),
        backstory=(
            "You are a senior editor deciding what goes on the front "
            "page. You know that a topic with 15 articles from 8 "
            "different sources is a stronger story than one with 20 "
            "articles all from the same outlet. If nothing strong "
            "exists in the last 7 days, you look back further rather "
            "than running a thin episode."
        ),
        verbose=False,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# The crew
# ---------------------------------------------------------------------------

class NewsRetrievalCrew:
    """
    Orchestrates the four-agent news retrieval pipeline using
    CrewAI's sequential process.

    Pipeline:
      QueryGeneratorAgent  →  expands topics to keyword facets
      RetrieverAgent       →  fetches RSS candidates per facet
      RankerAgent          →  scores by recency + authority + trend
      EditorialAgent       →  selects best topic and article slate

    The crew runs synchronously (Process.sequential) because each
    stage depends entirely on the previous stage's output. There is
    no parallelism benefit here — use LangGraph (hostify_graph.py)
    for the stateful pipeline stages that need conditional routing.

    Args:
        topics: User-specified interest topics.
        limit: Max articles to return in the final slate.
        last_hours: Initial search window (fallback ladder starts here).
    """

    WINDOW_LADDER = [
        (168,  "7d"),
        (720,  "30d"),
        (8760, "1y"),
    ]

    def __init__(
        self,
        topics: list[str],
        limit: int = 30,
        last_hours: int = 168,
    ):
        self.topics = topics
        self.limit = limit
        self.last_hours = last_hours

        # Instantiate agents
        self._query_agent    = make_query_generator_agent()
        self._retriever      = make_retriever_agent()
        self._ranker         = make_ranker_agent()
        self._editorial      = make_editorial_agent()

    def _build_tasks(self, window_hours: int) -> list[Task]:
        """Build the four sequential tasks for one retrieval attempt."""

        topic_list = ", ".join(self.topics)
        expansions_hint = str({
            t: TOPIC_EXPANSIONS.get(t.lower(), [t])
            for t in self.topics
        })

        task_query = Task(
            description=(
                f"Expand these user topics into keyword facets "
                f"for RSS search: [{topic_list}]\n\n"
                f"Reference expansions: {expansions_hint}\n\n"
                f"For any topic not in the reference, generate "
                f"6-10 semantically related search terms. "
                f"Return a JSON dict mapping each topic to its "
                f"keyword list."
            ),
            expected_output=(
                "JSON dict: {topic: [keyword, ...], ...} "
                "with at least 5 keywords per topic."
            ),
            agent=self._query_agent,
        )

        task_retrieve = Task(
            description=(
                f"Using the keyword facets from the previous task, "
                f"fetch candidate articles from RSS feeds published "
                f"within the last {window_hours} hours.\n\n"
                f"Use the feed_fetcher tool once per topic facet. "
                f"Cap results at 15 articles per domain. "
                f"Return a flat list of raw article dicts."
            ),
            expected_output=(
                "Flat list of article dicts, each with: "
                "title, url, source, published, snippet, facet_score."
            ),
            agent=self._retriever,
        )

        task_rank = Task(
            description=(
                "Score the candidate articles retrieved in the "
                "previous task.\n\n"
                "Steps:\n"
                "1. Use credibility_checker to add authority_score\n"
                "2. Use recency_scorer to add recency_score\n"
                "3. Compute final_score = "
                "   (facet_score * 1.0) + (recency_score * 2.0) "
                "   + (authority_score * 1.2)\n"
                "4. Apply trend_boost: articles whose title appears "
                "   across 2+ sources get +0.1 per additional source "
                "   (max +0.5)\n"
                "5. Sort descending by final_score\n\n"
                "Return the ranked list with all score fields preserved."
            ),
            expected_output=(
                "Ranked list of article dicts with fields: "
                "title, url, source, published, snippet, "
                "facet_score, authority_score, recency_score, "
                "trend_boost, final_score."
            ),
            agent=self._ranker,
        )

        task_editorial = Task(
            description=(
                f"Select the best topic and top {self.limit} articles "
                f"from the ranked slate.\n\n"
                f"Selection rules:\n"
                f"1. Group articles by topic facet\n"
                f"2. Score each topic: mean(top_10_scores) * "
                f"   (1 + 0.25 * source_diversity)\n"
                f"   source_diversity = unique_sources / article_count\n"
                f"3. A topic needs >= 6 articles to qualify\n"
                f"4. If no topic qualifies, pick the one with "
                f"   the most articles (fallback)\n"
                f"5. Return top {self.limit} articles from the "
                f"   winning topic\n\n"
                f"Return JSON with: chosen_topic (str), "
                f"items (list of article dicts)."
            ),
            expected_output=(
                "JSON with chosen_topic (str) and items "
                f"(list of up to {self.limit} article dicts)."
            ),
            agent=self._editorial,
        )

        return [task_query, task_retrieve, task_rank, task_editorial]

    def run(self) -> RetrievalResult:
        """
        Execute the crew with fallback window ladder.

        Tries 7d → 30d → 1y windows. Returns no_news_today
        if no qualifying articles exist across all windows.
        """
        for window_hours, window_label in self.WINDOW_LADDER:
            logger.info(
                "NewsRetrievalCrew: trying window=%s (%dh) "
                "for topics=%s",
                window_label, window_hours, self.topics,
            )

            tasks = self._build_tasks(window_hours)
            crew = Crew(
                agents=[
                    self._query_agent,
                    self._retriever,
                    self._ranker,
                    self._editorial,
                ],
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
            )

            try:
                result = crew.kickoff()
                # Parse the editorial agent's final output
                import json, re
                raw = str(result)
                # Strip markdown fences if present
                clean = re.sub(r"```json\s*|```", "", raw).strip()
                data = json.loads(clean)

                items = data.get("items", [])
                chosen_topic = data.get("chosen_topic", self.topics[0])

                if not items:
                    logger.info(
                        "No articles found in %s window. "
                        "Expanding search.", window_label
                    )
                    continue

                logger.info(
                    "NewsRetrievalCrew: selected topic=%s "
                    "articles=%d window=%s",
                    chosen_topic, len(items), window_label,
                )
                return RetrievalResult(
                    chosen_topic=chosen_topic,
                    items=items[: self.limit],
                    window_used=window_label,
                    status="ok",
                )

            except Exception as e:
                logger.warning(
                    "NewsRetrievalCrew failed on window=%s: %s. "
                    "Trying next window.", window_label, e,
                )
                continue

        # All windows exhausted
        logger.warning(
            "NewsRetrievalCrew: no articles found across all "
            "windows for topics=%s", self.topics,
        )
        return RetrievalResult(
            chosen_topic=self.topics[0] if self.topics else "news",
            items=[],
            window_used=None,
            status="no_news_today",
        )


# ---------------------------------------------------------------------------
# Public entry points — same signatures as ingestion.py
# so tasks.py requires zero changes
# ---------------------------------------------------------------------------

def fetch_with_crew(
    topics: list[str],
    limit: int = 30,
    last_hours: int = 168,
) -> dict[str, Any]:
    """
    Drop-in replacement for agentic_fetch_articles().
    Returns the same dict shape: {chosen_topic, items}.
    """
    crew = NewsRetrievalCrew(topics=topics, limit=limit)
    result = crew.run()
    return {
        "chosen_topic": result.chosen_topic,
        "items": result.items,
        "window_used": result.window_used,
        "status": result.status,
    }
