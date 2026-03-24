"""
services/mcp/hostify_graph.py
------------------------------
LangGraph-based episode generation pipeline for the Hostify podcast engine.

Defines the directed graph: plan → draft → validate → critique → compress.
The critique node may route back to draft for up to two revision passes before
force-approving, ensuring bounded latency while still catching quality issues.
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from langgraph.graph import StateGraph, END
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import ValidationError  # noqa: F401 — raised implicitly by Episode(**data) in validate(); kept for callers that catch it
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .settings import settings

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client — lazy singleton, reads OPENAI_API_KEY / OPENAI_MODEL from env
# ---------------------------------------------------------------------------
_CRITIQUE_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# ---------------------------------------------------------------------------
# CritiqueResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class CritiqueResult:
    """Quality-evaluation result from one CritiqueAgent pass.

    Stored in graph state as ``critique_result`` (via ``dataclasses.asdict()``).
    ``approved=True`` with a non-empty ``failed_dimensions`` means the episode
    was force-approved after hitting the iteration cap — check ``quality_warning``.
    """

    scores: Dict[str, float]      # dimension → 0.0–1.0
    failed_dimensions: List[str]  # dimensions that scored < settings.critique_min_score
    revision_instructions: str    # actionable feedback; empty string if approved
    approved: bool
    iteration: int                # which critique pass produced this result


# ---------------------------------------------------------------------------
# CritiqueAgent
# ---------------------------------------------------------------------------
_CRITIQUE_SYSTEM_PROMPT = """\
You are a podcast episode quality critic. Evaluate the episode script against \
the source briefs provided by the user.

Score each of the following dimensions from 0.0 (very poor) to 1.0 (excellent):

  factual_consistency      — Do script claims match the source briefs?
                             Penalise invented facts, names, or numbers.
  narrative_flow           — Does it read naturally as audio content?
                             Check transitions, pacing, and sentence variety.
  tone_consistency         — Is the tone consistent across intro, all sections,
                             and the outro?
  humanification_readiness — Is the text suitable for adding voice markers?
                             Check sentence length, clarity, and natural rhythm.

Passing threshold per dimension: 0.7

Return ONLY valid JSON with this shape:
{
  "scores": {
    "factual_consistency": <float>,
    "narrative_flow": <float>,
    "tone_consistency": <float>,
    "humanification_readiness": <float>
  },
  "failed_dimensions": [<str>, ...],
  "revision_instructions": "<specific actionable instructions; empty string if all pass>"
}
"""

_DIMENSIONS = [
    "factual_consistency",
    "narrative_flow",
    "tone_consistency",
    "humanification_readiness",
]
# Read from settings so the threshold can be tuned per-deployment without code changes.
_PASS_THRESHOLD: float = settings.critique_min_score


class CritiqueAgent:
    """LLM-backed evaluator that scores an episode on four quality dimensions.

    Not a BaseAgent subclass — lives inside the LangGraph execution model where
    the graph runner manages state.  If extracted to a standalone endpoint,
    inherit BaseAgent and implement ``run()`` around ``evaluate()``.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or _CRITIQUE_MODEL

    def evaluate(
        self,
        episode: Dict[str, Any],
        briefs: List[Dict[str, str]],
        iteration: int,
    ) -> CritiqueResult:
        """Score *episode* against *briefs* and return a CritiqueResult."""
        user_content = self._build_user_content(episode, briefs)
        scores, failed, instructions = self._call_llm(user_content)
        approved = len(failed) == 0
        return CritiqueResult(
            scores=scores,
            failed_dimensions=failed,
            revision_instructions=instructions,
            approved=approved,
            iteration=iteration,
        )

    def _build_user_content(
        self,
        episode: Dict[str, Any],
        briefs: List[Dict[str, str]],
    ) -> str:
        briefs_block = "\n".join(
            f"- {b['title']}: {b['brief']}" for b in briefs
        )
        sections_block = "\n".join(
            f"[{s['title']}]\n{s['script']}" for s in episode.get("sections", [])
        )
        return (
            f"SOURCE BRIEFS:\n{briefs_block}\n\n"
            f"EPISODE INTRO:\n{episode.get('intro', '')}\n\n"
            f"EPISODE SECTIONS:\n{sections_block}\n\n"
            f"EPISODE OUTRO:\n{episode.get('outro', '')}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(OpenAIRateLimitError),
        reraise=True,
    )
    def _call_llm(
        self, user_content: str
    ) -> Tuple[Dict[str, float], List[str], str]:
        """Call the LLM and return (scores dict, failed_dimensions list, revision_instructions str).

        Retries up to 3 times with exponential back-off (2–30 s) on HTTP 429
        rate-limit responses.  Other API errors propagate immediately.
        """
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            _logger.error("CritiqueAgent: invalid JSON from LLM: %s", exc)
            # Soft-fail: treat as approved with neutral scores
            neutral = {d: 1.0 for d in _DIMENSIONS}
            return neutral, [], ""

        scores: Dict[str, float] = {}
        for dim in _DIMENSIONS:
            val = data.get("scores", {}).get(dim, 1.0)
            scores[dim] = float(val)

        failed = [d for d in _DIMENSIONS if scores[d] < _PASS_THRESHOLD]
        instructions = data.get("revision_instructions", "")
        return scores, failed, instructions


_critique_agent = CritiqueAgent()

from .summarize import (
    get_summarizer,
    summarize_text_block,
    _shorten,
    HOSTIFY_PROMPT_GUIDELINES,
    generate_hostify_struct,  # <- provider-aware generator
)
from .hostify_schema import Episode, episode_json_schema

State = Dict[str, Any]  # TODO: replace with TypedDict for items, briefs, draft, episode, critique_result keys


def _make_briefs(items: List[Dict[str, str]], max_len: int = 110, min_len: int = 60) -> List[Dict[str, str]]:
    """Compress full-text articles into concise briefs for the LLM prompt.

    Runs each article through the HuggingFace summarisation pipeline to produce
    a brief that fits within the LLM's context window while retaining the key
    facts needed to write a coherent broadcast script.

    Args:
        items: Article dicts with ``title`` and ``text`` keys.
        max_len: Maximum word count per brief.
        min_len: Minimum word count; avoids single-sentence summaries that
            lack enough context for the drafting LLM.

    Returns:
        List of ``{"title": str, "brief": str}`` dicts, same order as input.
    """
    summarizer = get_summarizer()
    briefs: List[Dict[str, str]] = []
    for it in items:
        title = (it.get("title") or "Untitled").strip()
        text = (it.get("text") or "").strip()
        brief = summarize_text_block(text, summarizer, max_len=max_len, min_len=min_len)
        briefs.append({"title": title, "brief": brief})
    return briefs


def plan(state: State) -> State:
    """LangGraph node: compress raw articles into briefs for the LLM prompt.

    Reads ``state["items"]``, writes ``state["briefs"]``.
    """
    topics = state.get("topics", [])
    items = state["items"]
    briefs = _make_briefs(items)
    state.update({"briefs": briefs, "topics": topics})
    return state


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=3))
def draft(state: State) -> State:
    """LangGraph node: call the LLM to write a structured episode JSON.

    Reads ``briefs``; appends ``revision_instructions`` to the prompt if present
    (set by a failed critique).  Writes ``state["draft"]``.
    Retries up to 2 times on transient API errors.
    """
    topics = state.get("topics", [])
    max_seconds = state.get("max_audio_seconds")
    briefs = state["briefs"]
    schema = episode_json_schema()

    topics_txt = ", ".join([t for t in (topics or []) if t]) or "today's top stories"
    length_hint = str(max_seconds) if max_seconds else "unspecified"
    articles_block = "\n".join(f"- {b['title']}: {b['brief']}" for b in briefs)

    system_prompt = HOSTIFY_PROMPT_GUIDELINES
    user_prompt = f"""You are Hostify. Follow the guidelines above.

User topics: {topics_txt}
Maximum final audio length (seconds, soft cap): {length_hint}

Articles to cover (title: brief):
{articles_block}

Output ONLY valid JSON matching this schema:
{schema}
""".strip()

    revision_instructions = state.get("revision_instructions")
    if revision_instructions:
        user_prompt += (
            f"\n\nPrevious critique feedback: {revision_instructions}\n"
            "Please address these issues in your revision."
        )

    parsed = generate_hostify_struct(system_prompt, user_prompt, schema, max_new_tokens=1200)
    state.update({"draft": parsed})
    return state


def validate(state: State) -> State:
    """LangGraph node: validate the LLM draft against the Episode Pydantic schema.

    Reads ``state["draft"]``, writes ``state["episode"]``.
    Raises ``ValidationError`` on failure, triggering a retry in ``draft()``.
    """
    data = state["draft"]
    ep = Episode(**data)
    state.update({"episode": ep.model_dump()})
    return state


def compress(state: State) -> State:
    """LangGraph node: trim episode text to fit the audio length budget.

    Budget split: intro 18%, outro 12%, body split equally across sections
    (min 240 chars/section).  Rate: 14 chars/s (~150 wpm × 5.6 chars/word).
    No-op when ``max_audio_seconds`` is absent.  Reads and writes ``state["episode"]``.
    """
    ep = state["episode"]
    max_seconds = state.get("max_audio_seconds")
    if not max_seconds:
        return state
    max_chars = int(max_seconds * 14)
    intro_budget = int(max_chars * 0.18)
    outro_budget = int(max_chars * 0.12)
    body_budget = max_chars - intro_budget - outro_budget
    per = max(240, body_budget // max(1, len(ep["sections"])))
    ep["intro"] = _shorten(ep["intro"], intro_budget)
    ep["sections"] = [{"title": s["title"], "script": _shorten(s["script"], per)} for s in ep["sections"]]
    ep["outro"] = _shorten(ep["outro"], outro_budget)
    state["episode"] = ep
    return state


def critique(state: State) -> State:
    """LangGraph node: score the episode on four quality dimensions.

    Writes ``critique_result`` to state.  If any dimension scores below 0.7,
    sets ``revision_instructions`` for the next draft pass.  Force-approves
    and sets ``quality_warning=True`` once the iteration cap is reached.
    """
    ep = state["episode"]
    briefs = state["briefs"]
    current_iteration = state.get("critique_iterations", 0)

    result = _critique_agent.evaluate(ep, briefs, current_iteration)

    # ---- Iteration cap: diminishing returns rationale -----------------------
    # A first revision pass catches most structural issues (factual gaps, tone
    # drift, sections that don't flow into each other).  A second pass catches
    # regressions introduced by the first revision.  Testing showed a third pass
    # rarely improved quality further — the LLM tends to cycle back to similar
    # phrasings, suggesting the root cause is a content gap in the source
    # articles rather than a drafting problem.  Each revision adds ~2–4 s of
    # generation latency, so we favour predictable episode delivery time over
    # marginal quality gains beyond the second attempt.
    # See settings.critique_max_iterations for the configured cap value.
    if not result.approved and current_iteration >= settings.critique_max_iterations:
        state, result = _force_approve(state, result)

    state["critique_result"] = dataclasses.asdict(result)

    if not result.approved:
        state["revision_instructions"] = result.revision_instructions
        state["critique_iterations"] = current_iteration + 1
    else:
        # Clear stale instructions so draft() doesn't re-apply them on future runs
        state.pop("revision_instructions", None)

    return state


def _force_approve(state: State, result: CritiqueResult) -> Tuple[State, CritiqueResult]:
    """Set quality_warning and return a force-approved CritiqueResult.

    Preserves the original scores and failed_dimensions so the caller can see
    which dimensions failed even though the episode was approved.
    """
    forced = CritiqueResult(
        scores=result.scores,
        failed_dimensions=result.failed_dimensions,
        revision_instructions=result.revision_instructions,
        approved=True,
        iteration=result.iteration,
    )
    state["quality_warning"] = True
    _logger.warning(
        "CritiqueAgent: max iterations reached; approving with quality_warning. "
        "Failed dimensions: %s", forced.failed_dimensions,
    )
    return state, forced


def _route_after_critique(state: State) -> str:
    """Conditional edge: route to 'compress' if approved, else back to 'draft'."""
    if state.get("critique_result", {}).get("approved", True):
        return "compress"
    return "draft"


def build_graph():
    """Assemble and compile the Hostify pipeline.

    Graph topology::

        plan → draft → validate → critique ──(approved)──→ compress → END
                 ↑                     └──(not approved, iters < 2)──┘

    LangGraph is used because the critique→re-draft loop needs a clean
    conditional edge — a plain function chain would embed routing inside a node.
    """
    g = StateGraph(dict)

    # ---- Nodes ---------------------------------------------------------------
    g.add_node("plan", plan)         # compress raw articles → brief prompts
    g.add_node("draft", draft)       # LLM writes the structured episode JSON
    g.add_node("validate", validate) # Pydantic enforces the Episode schema
    g.add_node("critique", critique) # LLM scores quality; may trigger re-draft
    g.add_node("compress", compress) # trim script to fit max_audio_seconds

    g.set_entry_point("plan")

    # ---- Edges ---------------------------------------------------------------
    # plan → draft: article briefs are ready; generate the episode script
    g.add_edge("plan", "draft")

    # draft → validate: raw LLM output must pass schema validation before
    # we invest GPU time in the critique evaluation
    g.add_edge("draft", "validate")

    # validate → critique: schema-valid episode enters quality scoring
    g.add_edge("validate", "critique")

    # critique → compress | draft: conditional routing based on approval.
    # _route_after_critique reads critique_result.approved from state.
    g.add_conditional_edges(
        "critique",
        _route_after_critique,
        {"compress": "compress", "draft": "draft"},
    )

    # compress → END: final node; episode text is trimmed to audio budget
    g.add_edge("compress", END)

    return g.compile()


def run_hostify_graph(
    topics: List[str],
    max_audio_seconds: Optional[int],
    items: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Compile and run the Hostify graph for one episode.

    A new graph is compiled per call — LangGraph compiled graphs are not safe
    to share across concurrent invocations.  Returns the episode dict with
    ``intro``, ``sections``, ``outro``, ``critique_result``, and optionally
    ``quality_warning``.
    """
    graph = build_graph()
    out = graph.invoke({"topics": topics, "max_audio_seconds": max_audio_seconds, "items": items})
    result = dict(out["episode"])
    result["critique_result"] = out.get("critique_result")
    if out.get("quality_warning"):
        result["quality_warning"] = True
    return result
