"""
services/mcp/humanification_agent.py
--------------------------------------
LLM-powered voice script editor.  Transforms flat podcast scripts into
voice-realistic narration by inserting speech markers (<pause>, <breath>,
<emm>, <emphasis>) calibrated to one of three delivery tones.
"""

import json
import re
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base_agent import AgentResult, BaseAgent

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reads OPENAI_API_KEY / OPENAI_MODEL from the environment (same pattern as
# settings.py in this package)
# ---------------------------------------------------------------------------

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


# ---------------------------------------------------------------------------
# All speech markers the agent may insert
# ---------------------------------------------------------------------------
ALL_MARKERS = ["pause", "breath", "emm", "emphasis"]

_MARKER_RE = re.compile(
    r"<(?:" + "|".join(re.escape(m) for m in ALL_MARKERS) + r")>",
    re.IGNORECASE,
)

# Closing tags for paired markers (only <emphasis> wraps text)
_PAIRED_MARKER_RE = re.compile(r"</?emphasis>", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Tone descriptors injected into the system prompt
# ---------------------------------------------------------------------------
_TONE_GUIDANCE: Dict[str, str] = {
    "warm_professional": (
        "Tone: friendly but authoritative, like a trusted radio host. "
        "Use <breath> and <pause> at natural sentence breaks. "
        "Use <emphasis> on key figures or names. Minimise <emm>."
    ),
    "casual": (
        "Tone: conversational and relaxed, like a friend catching you up on the news. "
        "Allow more <breath> and <emm> markers to feel spontaneous. "
        "Use <emphasis> for surprising facts. Keep it light."
    ),
    "formal": (
        "Tone: clean, authoritative delivery. "
        "Use markers sparingly — prefer <pause> only after critical statements. "
        "Avoid <emm> entirely. A single <emphasis> per paragraph at most."
    ),
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class HumanifiedScript:
    """Output of one ``HumanificationAgent.humanify()`` call.

    Carries both the original and transformed script so callers can diff them,
    fall back to the original if needed, or strip markers for plain-text use
    via ``strip_markers()``.

    Attributes:
        original: The raw input script before any markers were added.
        humanified: The transformed script with speech markers inserted.
            Ready to be fed to a TTS engine that supports SSML-like tags.
        markers_added: Deduplicated, sorted list of marker *types* that appear
            in ``humanified``.  Populated by scanning the actual output text
            (not just the LLM's self-report) so the list is authoritative.

            Marker semantics for TTS engines:

            - ``"pause"`` — insert a ~0.5 s silence; placed after important
              statements to let the listener absorb the information.
            - ``"breath"`` — simulate an audible inhale; placed at natural
              sentence breaks to make delivery feel human and unrushed.
            - ``"emm"`` — add a thinking pause (``"mm..."``); placed where a
              live host would momentarily hesitate before a key fact.
            - ``"emphasis"`` — stress the wrapped word or phrase; placed around
              key names, numbers, or surprising facts.

        tone: The tone key used to generate this script (``"warm_professional"``,
            ``"casual"``, or ``"formal"``).  Recorded so callers can log or
            display which style was applied.
    """

    original: str
    humanified: str
    markers_added: List[str]   # e.g. ["pause", "breath", "emm"]
    tone: str


# ---------------------------------------------------------------------------
# Utility: strip all markers for plain-text fallback
# ---------------------------------------------------------------------------
def strip_markers(script: str) -> str:
    """Remove every speech marker tag from *script*, returning plain text.

    Handles both standalone markers (<pause>, <breath>, <emm>) and the paired
    <emphasis>…</emphasis> wrapper, leaving the wrapped text intact.

    Example
    -------
    >>> strip_markers("Hello <pause> world, <emphasis>great</emphasis> news!")
    'Hello  world, great news!'
    """
    # Remove paired <emphasis> / </emphasis> tags but keep inner text
    text = re.sub(r"</?emphasis>", "", script, flags=re.IGNORECASE)
    # Remove standalone markers
    text = _MARKER_RE.sub("", text)
    # Collapse double-spaces that may result from removal
    text = re.sub(r"  +", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
class HumanificationAgent(BaseAgent):
    """Transform a flat podcast script into a voice-realistic script.

    Inherits from :class:`~services.mcp.base_agent.BaseAgent`.  The primary
    user-facing method is :meth:`humanify`; the :meth:`run` override satisfies
    the ``BaseAgent`` contract and delegates to it.

    The agent calls an LLM to insert speech markers (``<pause>``, ``<breath>``,
    ``<emm>``, ``<emphasis>``) at natural points, calibrated to the requested
    tone.  Factual content is never altered — only markers are added.

    Parameters
    ----------
    model:
        OpenAI model identifier.  Defaults to the ``OPENAI_MODEL`` environment
        variable, falling back to ``"gpt-4o-mini"``.

    Usage
    -----
    ::

        agent = HumanificationAgent()

        # Primary API
        result = agent.humanify(script, tone="warm_professional")
        print(result.humanified)

        # BaseAgent contract API (wraps humanify)
        agent_result = agent.run(script, tone="casual")
        print(agent_result.data.humanified)

        # Plain-text fallback
        print(strip_markers(result.humanified))
    """

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__("HumanificationAgent")
        self._model: str = model or _OPENAI_MODEL

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------
    def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        """Wrap humanify() in an AgentResult for the BaseAgent contract.

        Never raises — exceptions are caught and returned as ``status="error"``.
        """
        script: str = args[0] if args else kwargs.get("script", "")
        tone: str = kwargs.get("tone", "warm_professional")

        self._log_start(script_len=len(script), tone=tone)
        try:
            humanified = self.humanify(script, tone)
            result = AgentResult(
                status="ok",
                data=humanified,
                metadata={"tone": humanified.tone, "markers_added": humanified.markers_added},
            )
        except Exception as exc:
            self.logger.error("HumanificationAgent failed: %s", exc)
            result = AgentResult(status="error", data=None, error=str(exc))

        self._log_complete(result)
        return result

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------
    def humanify(
        self,
        script: str,
        tone: str = "warm_professional",
    ) -> HumanifiedScript:
        """Transform *script* by adding natural speech markers.

        Parameters
        ----------
        script:
            The raw podcast script to transform (plain text).
        tone:
            One of ``"warm_professional"`` (default), ``"casual"``, or
            ``"formal"``.  Controls how many and which types of markers are
            inserted.  Unknown values fall back to ``"warm_professional"``
            with a warning log.

        Returns
        -------
        HumanifiedScript
            Dataclass with the original text, humanified text, a deduplicated
            list of marker types that were actually inserted, and the tone used.
        """
        if tone not in _TONE_GUIDANCE:
            self.logger.warning(
                "Unknown tone %r — falling back to 'warm_professional'.", tone
            )
            tone = "warm_professional"

        system_prompt = self._build_system_prompt(tone)
        humanified_text, markers = self._call_llm(script, system_prompt)

        return HumanifiedScript(
            original=script,
            humanified=humanified_text,
            markers_added=markers,
            tone=tone,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_system_prompt(self, tone: str) -> str:
        tone_block = _TONE_GUIDANCE[tone]
        return (
            "You are a voice script editor. Transform the script to sound natural "
            "when read aloud by adding:\n"
            "- <pause> after important statements (0.5 s pause)\n"
            "- <breath> at natural breathing points between sentences\n"
            "- <emm> where a human host would add a thinking pause\n"
            "- <emphasis> around key words that should be stressed\n\n"
            "Rules:\n"
            "- Add markers sparingly — max 1 per 2 sentences\n"
            "- Never add markers mid-word or mid-number\n"
            "- Preserve all factual content exactly\n"
            f"- {tone_block}\n"
            "- Return JSON: {\"humanified\": str, \"markers_added\": [str]}\n"
            "  where markers_added is the deduplicated list of marker types used "
            "(e.g. [\"pause\", \"breath\"])."
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(OpenAIRateLimitError),
        reraise=True,
    )
    def _call_llm(self, script: str, system_prompt: str) -> Tuple[str, List[str]]:
        """Call the LLM and return ``(humanified_text, markers_list)``.

        Retries up to 3 times with exponential back-off (2–30 s) on HTTP 429
        rate-limit responses.  On JSON parse failure, returns the original
        *script* unchanged with an empty markers list so the agent degrades
        gracefully rather than raising.
        """
        client = _get_client()

        response = client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": script},
            ],
            temperature=0.4,   # low variance — we want consistent, not creative
            max_tokens=4096,
        )

        raw = response.choices[0].message.content or ""

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.logger.error("LLM returned invalid JSON: %s", exc)
            # Graceful fallback: return the script unchanged with no markers
            return script, []

        humanified_text: str = data.get("humanified", script)
        markers_from_llm: List[str] = data.get("markers_added", [])

        # Cross-check: scan the actual output for inserted markers as the
        # authoritative source of truth (guards against LLM mis-reporting).
        # _MARKER_RE matches full tags like <pause>; strip brackets to get names.
        found_names = {
            tag.strip("<>").lower()
            for tag in _MARKER_RE.findall(humanified_text)
        }
        # Merge: trust what's actually in the text, supplement with LLM list
        merged: List[str] = sorted(
            found_names | {m.lower() for m in markers_from_llm if m.lower() in ALL_MARKERS}
        )

        return humanified_text, merged
