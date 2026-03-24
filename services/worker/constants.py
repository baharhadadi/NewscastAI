"""
services/worker/constants.py
-----------------------------
Shared constants for the worker-tier ingestion and search pipeline.

These are intentionally kept in one place so that both the RSS ingestion path
(ingestion.py) and the SearxNG agentic search path (agent_search.py) stay in
sync — adding a keyword in one place automatically benefits both.
"""

# ---------------------------------------------------------------------------
# TOPIC_EXPANSIONS
# ---------------------------------------------------------------------------
# Maps a canonical topic key to a list of keyword synonyms used for relevance
# scoring against article titles, summaries, and tags.
#
# How to extend:
#   - Add a new key + synonym list for a new topic.
#   - Extend an existing list with more specific terms (e.g. company names,
#     legislation names) to improve recall without changing the pipeline code.
#   - Keep keywords lowercase; the regex matcher is case-insensitive.
#
TOPIC_EXPANSIONS: dict = {
    "ai":       ["ai", "artificial intelligence", "machine learning", "ml", "llm",
                 "openai", "anthropic", "deepmind", "genai", "transformer", "rag"],
    "economy":  ["economy", "inflation", "gdp", "jobs report", "unemployment",
                 "interest rates", "cpi", "ppi", "central bank", "fed", "recession"],
    "markets":  ["markets", "stocks", "equities", "etf", "nasdaq", "s&p", "dow",
                 "earnings", "ipo", "crypto", "bitcoin"],
    "politics": ["politics", "election", "parliament", "congress", "senate",
                 "white house", "prime minister", "policy", "bill", "lawmakers", "minister"],
    "war":      ["conflict", "war", "military", "airstrike", "frontline", "ceasefire"],
    "health":   ["health", "covid", "vaccine", "who", "cdc", "outbreak", "cancer",
                 "mental health"],
    "science":  ["science", "research", "study", "paper", "scientists", "breakthrough",
                 "quantum", "space", "nasa", "esa"],
    "tech":     ["tech", "software", "hardware", "semiconductor", "chip", "nvidia",
                 "intel", "amd", "apple", "google", "microsoft", "cloud", "cybersecurity"],
    "energy":   ["energy", "oil", "gas", "opec", "renewable", "solar", "wind",
                 "battery", "nuclear", "grid"],
}


# ---------------------------------------------------------------------------
# DOMAIN_AUTHORITY
# ---------------------------------------------------------------------------
# Per-domain scoring weight (0.0–1.0) applied during multi-factor article
# ranking.  Higher values bias the scorer toward more authoritative sources.
#
# How to extend:
#   - Add a new "domain.tld": score entry for any outlet you want to weight.
#   - The special key "*" is the fallback weight for domains not listed here.
#   - Scores above 0.9 should be reserved for wire services and tier-1 papers.
#
DOMAIN_AUTHORITY: dict = {
    # Wire services / tier-1 papers
    "reuters.com":      1.0,
    "apnews.com":       0.95,
    "bbc.co.uk":        0.9,
    "bbc.com":          0.9,
    "nytimes.com":      0.9,
    "ft.com":           0.9,
    "wsj.com":          0.9,
    "nature.com":       0.9,
    # National / regional outlets
    "theguardian.com":  0.85,
    "cbc.ca":           0.85,
    "cnbc.com":         0.8,
    "globalnews.ca":    0.75,
    "arstechnica.com":  0.75,
    # Tech / specialty outlets
    "theverge.com":     0.7,
    "techcrunch.com":   0.7,
    "wired.com":        0.7,
    "sciencedaily.com": 0.6,
    # Fallback weight for any domain not listed above
    "*":                0.6,
}
