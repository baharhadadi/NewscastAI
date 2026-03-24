# NewscastAI — Architecture

NewscastAI generates a personalized daily audio briefing from
live news sources. Given a set of user topics, it retrieves and
ranks articles across 22 RSS feeds, writes a podcast-style script
through a multi-stage LangGraph pipeline with a critique feedback
loop, adds voice markers for natural TTS delivery, and assembles
the final MP3 — delivered via RSS feed or direct API link.

---

## Pipeline Overview
```
User Topics (e.g. "AI", "Finance", "Canada")
│
▼
┌─────────────────────────────────────────────┐
│         CrewAI Retrieval Crew               │
│  ┌──────────────────────────────────────┐   │
│  │ 1. QueryGeneratorAgent               │   │
│  │    topics → keyword facets           │   │
│  │    (uses TOPIC_EXPANSIONS registry)  │   │
│  ├──────────────────────────────────────┤   │
│  │ 2. RetrieverAgent                    │   │
│  │    facets → RSS candidates           │   │
│  │    tool: FeedFetcherTool (22 feeds)  │   │
│  ├──────────────────────────────────────┤   │
│  │ 3. RankerAgent                       │   │
│  │    candidates → scored articles      │   │
│  │    tools: CredibilityCheckerTool,    │   │
│  │           RecencyScorerTool          │   │
│  │    score = facet(1.0) +              │   │
│  │            recency(2.0) +            │   │
│  │            authority(1.2)            │   │
│  ├──────────────────────────────────────┤   │
│  │ 4. EditorialAgent                    │   │
│  │    ranked → chosen topic + slate     │   │
│  │    fallback: 7d → 30d → 1y →        │   │
│  │             no_news_today            │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
│
│  {chosen_topic, items[]}
▼
┌─────────────────────────────────────────────┐
│         LangGraph Script Pipeline           │
│                                             │
│   plan ──► draft ──► validate ──► critique  │
│                          ▲           │      │
│                          │  (reject, │      │
│                          │  max 2x)  │      │
│                          └───────────┘      │
│                               │ (approve)   │
│                               ▼             │
│                           compress          │
│                               │             │
│              critique powered by            │
│              Anthropic tool use API         │
│              (4 scoring tools +             │
│               submit_critique)              │
└─────────────────────────────────────────────┘
│
│  episode {intro, sections[], outro}
▼
┌─────────────────────────────────────────────┐
│         HumanificationAgent                 │
│  Adds <pause> <breath> <emm> <emphasis>     │
│  markers for natural TTS delivery           │
└─────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────┐
│  TTS (gTTS)  →  AudioAssembler (pydub)      │
│  intro clip + section clips + outro clip    │
│  → /mnt/audio/user_{id}_{date}.mp3          │
└─────────────────────────────────────────────┘
│
▼
RSS Feed (/feed/{user_id}.rss)
or API   (/episodes/{user_id}/latest)
```

---

## Service Architecture

| Service  | Stack                    | Port | Responsibility                          |
|----------|--------------------------|------|-----------------------------------------|
| api      | FastAPI + SQLAlchemy     | 8000 | User prefs, episode records, RSS feed   |
| worker   | Celery + APScheduler     | 8001 | Retrieval → script → TTS → assemble     |
| mcp      | FastAPI                  | 7000 | LangGraph script pipeline + TTS         |
| nginx    | nginx:alpine             | 8080 | Audio file serving, API proxy           |
| postgres | postgres:16              | 5432 | User and episode persistence            |
| redis    | redis:7                  | 6379 | Celery broker and result backend        |
| searxng  | searxng/searxng          | 8081 | Metasearch (used by agent_search)       |
| vllm     | vllm/vllm-openai         | 8003 | Local LLM inference (OpenAI-compatible) |
| minio    | minio/minio              | 9000 | Audio file object storage               |

---

## Framework Responsibilities

Three agentic frameworks are used, each chosen for a specific
structural reason — not interchangeable:

**CrewAI** — news retrieval stage
Role-based agents with distinct responsibilities and tools.
Sequential process with no feedback loops. CrewAI's agent
backstory and goal prompting improves per-role focus compared
to a single monolithic class. The four retrieval concerns
(query expansion, fetching, ranking, editorial selection)
are genuinely independent and benefit from separation.

**LangGraph** — script generation stage
Stateful graph with conditional routing. The critique loop
requires routing back to draft() on rejection — a feedback
edge that LangGraph handles natively via conditional_edges.
A sequential function chain cannot express this without
manual state management.

**Anthropic tool use API** — critique evaluation
The critique node uses claude-haiku-4-5 with five structured
tools (four scoring tools + submit_critique). Tool use forces
the model to commit to specific claims before producing a
verdict, making scores individually auditable. Plain prompting
for a structured score produces less reliable and less
inspectable results for this evaluation task.

---

## Agent Details

### CrewAI Retrieval Crew

| Agent               | Tool(s)                                  | Input            | Output                    |
|---------------------|------------------------------------------|------------------|---------------------------|
| QueryGeneratorAgent | None (LLM reasoning)                     | topics[]         | {topic: [keywords]}       |
| RetrieverAgent      | FeedFetcherTool                          | keyword facets   | raw article list          |
| RankerAgent         | CredibilityCheckerTool, RecencyScorerTool| raw articles     | scored + ranked articles  |
| EditorialAgent      | None (LLM reasoning)                     | ranked articles  | chosen_topic + slate      |

### Critique Agent (Anthropic tool use)

| Tool                         | Evaluates                              | Output field           |
|------------------------------|----------------------------------------|------------------------|
| score_factual_consistency    | Claims vs source briefs                | score + unsupported[]  |
| score_narrative_flow         | Audio readability                      | score + issues[]       |
| score_tone_consistency       | Register consistency                   | score + inconsistencies[] |
| score_humanification_readiness | Sentence structure for voice markers | score + suggestions[]  |
| submit_critique              | Final approve/reject                   | approved + instructions |

Approval threshold: all dimensions >= 0.7.
Max iterations before force-approve: 2.

---

## Fallback Window Ladder

If the EditorialAgent finds insufficient articles (< 6) in
the initial 7-day window, the crew retries with progressively
wider windows:
```
7 days  → 30 days  → 1 year  → status: no_news_today
```

On no_news_today, generate_episode() produces a brief
"no fresh articles" episode rather than failing silently.
This is a deliberate UX decision: a subscriber should always
receive something, even if it is just an acknowledgement
that their topics had no coverage today.

---

## Scoring Formula

Articles are ranked by:
```
final_score = (facet_score × 1.0)
            + (recency_score × 2.0)
            + (authority_score × 1.2)
            × trend_boost
```

Where:
- `facet_score` = keyword hit count (title weighted 2×)
- `recency_score` = exp(−λ × hours_old), λ = ln(2)/12
  (half-life of 12 hours — article from 12h ago scores 0.5)
- `authority_score` = DOMAIN_AUTHORITY lookup (reuters.com=1.0,
  unknown=0.6)
- `trend_boost` = 1.0 + min(0.5, 0.1 × (source_count − 1))
  (same story across 3 sources → ×1.2)

Recency is weighted 2× because freshness is the core value
proposition of a daily briefing. Authority is weighted 1.2×
rather than equal to recency because a day-old Reuters article
should still beat a fresh blog post.
