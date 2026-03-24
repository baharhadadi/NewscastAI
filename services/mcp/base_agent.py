"""
Base agent abstraction shared across MCP and worker services.

IMPORTANT — DUPLICATION IS INTENTIONAL:
This file exists in both services/mcp/base_agent.py and
services/worker/base_agent.py. The two copies are kept in
sync manually. This is a deliberate tradeoff: a shared
services/common/ package would require coordinated Docker
build changes and introduces import path complexity that is
not justified for a two-service system.

Refactor to services/common/base_agent.py when:
  - A third service needs BaseAgent, OR
  - The class grows beyond ~50 lines

If you modify this file, update the other copy too.
Last synced: 2026-03-24
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging


@dataclass
class AgentResult:
    """Uniform return envelope for every agent in the newscast pipeline.

    Attributes
    ----------
    status:
        Short outcome code.  Standard values: ``"ok"``, ``"error"``,
        ``"no_news_today"``.  Agents may define additional status codes as
        needed; document them in the concrete class docstring.
    data:
        The primary payload — type depends on the concrete agent.
        ``None`` when ``status`` is ``"error"``.
    error:
        Human-readable error message, populated only when ``status == "error"``.
    metadata:
        Optional supplementary information (e.g. timing, window used, iteration
        count).  Callers should treat this as advisory; never required for
        correct downstream behaviour.
    """

    status: str                      # "ok" | "error" | "no_news_today"
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class BaseAgent(ABC):
    """Abstract base class for all newscast agents (MCP and worker tiers).

    Subclasses must implement :meth:`run`.  The base class wires up a
    namespaced logger and provides ``_log_start`` / ``_log_complete`` helpers
    so every agent produces consistent structured log lines without boilerplate.

    Parameters
    ----------
    name:
        Short, unique agent identifier (e.g. ``"MyAgent"``).
        Used as the logger suffix: ``newscast.<name>``.

    Example
    -------
    ::

        class MyAgent(BaseAgent):
            def __init__(self):
                super().__init__("MyAgent")

            def run(self, payload: str) -> AgentResult:
                self._log_start(payload_len=len(payload))
                result = AgentResult(status="ok", data=payload.upper())
                self._log_complete(result)
                return result
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.logger: logging.Logger = logging.getLogger(f"newscast.{name}")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        """Execute the agent's primary task.

        Subclasses must implement this method.  Async subclasses may declare it
        as ``async def run(...)`` — Python's ABC machinery accepts the override.

        Returns
        -------
        AgentResult
            Always returned (never raises).  Callers inspect ``result.status``
            to determine success or failure.
        """

    def _log_start(self, **kwargs: Any) -> None:
        """Emit a structured INFO log marking the start of a run.

        Parameters
        ----------
        **kwargs:
            Key-value pairs describing the run inputs (e.g. ``payload_len=400``).
        """
        self.logger.info("[%s] Starting with: %s", self.name, kwargs)

    def _log_complete(self, result: AgentResult) -> None:
        """Emit a structured INFO log marking the end of a run.

        Parameters
        ----------
        result:
            The :class:`AgentResult` returned by :meth:`run`.
        """
        self.logger.info("[%s] Complete. Status: %s", self.name, result.status)

# Mirror: services/worker/base_agent.py
