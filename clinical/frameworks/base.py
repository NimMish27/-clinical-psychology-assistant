from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from app_logging.logger import get_logger
from clinical.frameworks.models import FrameworkContext, FrameworkMetadata, FrameworkResult

_log = get_logger(__name__)


class BaseFramework(ABC):
    """Abstract base class for all therapeutic framework plugins.

    Provides shared infrastructure for prompt building, LLM invocation,
    JSON parsing, and result construction. Subclasses only need to
    define ``name``, ``description``, ``system_prompt``, and
    implement ``analyze()``.

    Usage::

        class CBTFramework(BaseFramework):
            name = "CBT"
            description = "..."
            system_prompt = "..."
            async def analyze(self, context: FrameworkContext) -> FrameworkResult:
                return await self._run_analysis(context)
    """

    # ── Abstract interface ──────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier — used as a dict key in comparison results."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description of the therapeutic framework."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Framework-specific LLM system prompt that defines the analytical lens."""

    @property
    def metadata(self) -> FrameworkMetadata:
        """Framework-level metadata for discovery and evaluation.

        Default implementation derives ``id``, ``display_name``, and
        ``description`` from the framework's own properties, so every
        registered framework exposes metadata out of the box. Subclasses
        may override to provide richer metadata (version, features, use
        cases, benchmark categories).
        """
        return FrameworkMetadata(
            id=self.name,
            display_name=self.name,
            description=self.description,
        )

    @abstractmethod
    async def analyze(
        self,
        context: FrameworkContext,
    ) -> FrameworkResult:
        """Produce a framework-specific analysis of the clinical case.

        Subclasses should delegate to ``self._run_analysis(context)``
        unless they need custom override logic.
        """

    # ── Shared infrastructure ───────────────────────────────

    async def _run_analysis(
        self,
        context: FrameworkContext,
    ) -> FrameworkResult:
        """Execute the standard analysis flow: prompt → LLM → parse → result.

        Args:
            context: Typed context containing all case data and the LLM service.

        Returns:
            FrameworkResult with framework-specific analysis.
        """
        t_start = time.perf_counter()
        llm = context.llm

        prompt = self._build_prompt(context)

        try:
            raw = await llm.generate(prompt=prompt, system_prompt=self.system_prompt)
            data = self._parse_json_response(raw)
            result = self._build_result(data)
        except Exception as exc:
            _log.error(
                "%s_framework.analysis_failed",
                self.name.lower(),
                error=str(exc),
            )
            result = FrameworkResult(
                framework_name=self.name,
                framework_description=self.description,
                conceptualization=f"{self.name} analysis could not be completed.",
                confidence=0.0,
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        object.__setattr__(result, "analysis_ms", round(elapsed_ms, 2))
        return result

    def _build_prompt(self, context: FrameworkContext) -> str:
        """Build the LLM prompt from the framework context.

        Override in subclass if a framework needs a different prompt
        structure.  Default implementation includes understanding,
        evidence summary, key findings, and raw text.
        """
        parts: list[str] = []

        parts.append("## Case Understanding\n")
        parts.append(context.understanding.model_dump_json(indent=2))

        if context.evidence and context.evidence.overall_summary:
            parts.append("\n## Evidence Summary\n")
            parts.append(context.evidence.overall_summary)
            if context.evidence.key_findings:
                parts.append("\n## Key Findings\n")
                for f in context.evidence.key_findings:
                    parts.append(f"- {f.statement} (confidence: {f.confidence})")

        if context.raw_text:
            parts.append("\n## Raw Case Text\n")
            parts.append(context.raw_text)

        return "\n".join(parts)

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """Extract and parse JSON from an LLM response.

        Handles responses wrapped in markdown code fences and
        responses with surrounding text.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        start = cleaned.index("{")
        end = cleaned.rindex("}")
        return json.loads(cleaned[start : end + 1])

    def _build_result(self, data: dict[str, Any]) -> FrameworkResult:
        """Build a FrameworkResult from parsed JSON data.

        Override in subclass if a framework needs custom field extraction.
        """
        return FrameworkResult(
            framework_name=self.name,
            framework_description=self.description,
            conceptualization=str(data.get("conceptualisation", "")),
            key_concepts=[str(c) for c in data.get("key_concepts", []) if c],
            therapeutic_approaches=[str(t) for t in data.get("therapeutic_approaches", []) if t],
            strengths=[str(s) for s in data.get("strengths", []) if s],
            limitations=[str(x) for x in data.get("limitations", []) if x],
            confidence=float(data.get("confidence", 0.0)),
            evidence_alignment=str(data.get("evidence_alignment", "")),
        )
