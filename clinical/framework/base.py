"""
clinical/framework/base.py — Abstract base classes for framework plugins.

``FrameworkPlugin``  — Minimal contract used by the registry for
                       applicability detection and legacy support.

``ClinicalFramework`` — Rich lifecycle ABC with 10 strongly-typed
                       analysis methods.  All methods ship with
                       production-ready default implementations
                       that delegate to the LLM.  Subclasses override
                       individual methods for custom logic.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from clinical.framework.models import (
    CaseAnalysis,
    CaseConceptualization,
    EvidenceResult,
    FollowupQuestionsResult,
    FrameworkAnalysisResult,
    FrameworkCaseInput,
    FrameworkMetadata,
    FrameworkOutput,
    InterventionPlanResult,
    LimitationsResult,
    RetrievedEvidence,
    SessionObjectivesResult,
    TherapeuticFocusResult,
)
from clinical.framework.prompts.manager import PromptManager
from app_logging.logger import get_logger

if TYPE_CHECKING:
    from clinical.llm import LLMService

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Legacy base — registry compatibility
# ═══════════════════════════════════════════════════════════════

class FrameworkPlugin(ABC):
    """
    Contract every framework plugin must satisfy.

    Usage:
        @register_framework
        class ACTPlugin(FrameworkPlugin):
            @classmethod
            def metadata(cls) -> FrameworkMetadata: ...
            @classmethod
            def is_applicable(cls, case_understanding: dict) -> bool: ...
            async def analyze(cls, case_understanding, formulation, evidence, llm) -> FrameworkAnalysisResult: ...
    """

    # ── Identity (class-level, no instance needed) ──────────────────────────

    @classmethod
    @abstractmethod
    def metadata(cls) -> FrameworkMetadata:
        """Return static metadata describing this framework."""

    @classmethod
    @abstractmethod
    def is_applicable(cls, case_understanding: dict) -> bool:
        """
        Decide whether this framework is appropriate for the given case.

        The ``case_understanding`` dict is the flat dict produced by
        ``CaseUnderstandingResult.to_flat_dict()``.

        Returns:
            True if this framework can provide meaningful clinical guidance
            for the described case; False otherwise.
        """

    # ── Analysis (instance-level, uses LLM) ─────────────────────────────────

    async def analyze(
        self,
        case_understanding: dict,
        formulation: dict | None = None,
        evidence: str | None = None,
        llm: LLMService | None = None,
    ) -> FrameworkAnalysisResult:
        """
        Generate framework-specific clinical guidance for the given case.

        The default implementation builds a prompt from the plugin's
        metadata and delegates to the LLM.  Subclasses may override
        for custom logic.

        Args:
            case_understanding: Flat dict from ``CaseUnderstanding.to_flat_dict()``.
            formulation: Optional dict from ``ClinicalFormulationResult.model_dump()``.
            evidence: Optional evidence summary text.
            llm: Shared LLM service instance.

        Returns:
            FrameworkAnalysisResult with the plugin's guidance.
        """
        meta = self.metadata()

        if isinstance(self, ClinicalFramework) and llm is not None:
            fw_input = FrameworkCaseInput(
                case_understanding=case_understanding,
                formulation=formulation,
                evidence_summary=evidence,
            )
            output = await self.run_full_analysis(fw_input, llm)
            return FrameworkAnalysisResult(
                slug=meta.id,
                display_name=meta.display_name,
                guidance=_summarize_output(output),
                applicable=True,
                error=output.error,
            )

        # Legacy fallback when not a ClinicalFramework subclass
        parts = [
            f"You are a {meta.display_name} clinical specialist.",
            "",
            "Based on the following case information, provide clinical guidance",
            f"from a {meta.display_name} perspective. Focus on how this framework",
            "would conceptualise the case and what intervention strategies it would",
            "recommend. Write 2-4 paragraphs in plain clinical language.",
            "",
            "---",
            "",
            "## Case Understanding",
        ]

        for key, val in case_understanding.items():
            if val is not None and val != [] and val != {} and val != "":
                display = str(val)[:500]
                parts.append(f"- {key}: {display}")

        if formulation:
            parts.append("")
            parts.append("## Clinical Formulation")
            for key, val in formulation.items():
                if val is not None and val != [] and val != {} and val != "":
                    display = str(val)[:300]
                    parts.append(f"- {key}: {display}")

        if evidence:
            parts.append("")
            parts.append("## Evidence Summary")
            parts.append(str(evidence)[:1000])

        prompt = "\n".join(parts)

        if llm is not None:
            try:
                text = await llm.generate(prompt=prompt)
            except Exception as exc:
                return FrameworkAnalysisResult(
                    slug=meta.id,
                    display_name=meta.display_name,
                    applicable=True,
                    error=str(exc),
                )
        else:
            text = (
                f"[{meta.display_name} analysis — LLM not available; "
                f"using {len(meta.intervention_style.techniques)} "
                f"identified techniques]"
            )

        return FrameworkAnalysisResult(
            slug=meta.id,
            display_name=meta.display_name,
            applicable=True,
            guidance=text,
        )


def _summarize_output(output: FrameworkOutput) -> str:
    """Collapse a FrameworkOutput into a single guidance string."""
    parts: list[str] = []
    if output.conceptualization:
        parts.append(f"Conceptualization: {output.conceptualization.conceptualization_summary}")
    if output.therapeutic_focus:
        parts.append(f"Focus: {'; '.join(output.therapeutic_focus.primary_focus_areas[:3])}")
    if output.intervention_plan:
        parts.append(f"Plan: {output.intervention_plan.provisional_strategy[:200]}")
    if output.limitations:
        if output.limitations.caveats:
            parts.append(f"Caveats: {'; '.join(output.limitations.caveats[:2])}")
    return "\n\n".join(parts) if parts else "Analysis completed."


# ═══════════════════════════════════════════════════════════════
# Rich lifecycle base
# ═══════════════════════════════════════════════════════════════

class ClinicalFramework(FrameworkPlugin):
    """
    Production-ready abstract base for therapeutic framework plugins.

    Provides ten strongly-typed lifecycle methods that mirror the
    full clinical reasoning pipeline.  Each method ships with a
    default implementation that delegates to the LLM using prompts
    built from the framework's own metadata — override individual
    methods for custom logic.

    Usage::

        @register_framework
        class ACTPlugin(ClinicalFramework):
            @classmethod
            def metadata(cls) -> FrameworkMetadata: ...
            @classmethod
            def is_applicable(cls, case_understanding: dict) -> bool: ...
    """

    # ── Initialization ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Prepare the framework for use.

        Called once before any lifecycle method.  Subclasses may
        override to pre-load models, warm caches, or validate
        configuration.  The default implementation is a no-op.
        """

    # ── Shared helpers ──────────────────────────────────────────────────────

    def _build_input(
        self,
        case_understanding: dict,
        formulation: dict | None = None,
        evidence: str | None = None,
        raw_text: str = "",
    ) -> FrameworkCaseInput:
        return FrameworkCaseInput(
            case_understanding=case_understanding,
            formulation=formulation,
            evidence_summary=evidence,
            raw_text=raw_text,
        )

    def _system_header(self) -> str:
        """Return the system-prompt header used by every LLM call."""
        m = self.metadata()
        return (
            f"You are a {m.display_name} clinical specialist.\n"
            f"{m.description}\n\n"
            f"Intervention style: {m.intervention_style.primary_focus}\n"
            f"Techniques: {', '.join(m.intervention_style.techniques)}\n"
            f"Therapeutic stance: {m.intervention_style.therapeutic_stance}"
        )

    def _format_input_for_prompt(self, inp: FrameworkCaseInput) -> str:
        """Format FrameworkCaseInput as a prompt section."""
        lines: list[str] = ["## Case Understanding"]
        for k, v in inp.case_understanding.items():
            if v is not None and v != [] and v != {} and v != "":
                lines.append(f"- {k}: {str(v)[:500]}")
        if inp.formulation:
            lines.append("")
            lines.append("## Clinical Formulation")
            for k, v in inp.formulation.items():
                if v is not None and v != [] and v != {} and v != "":
                    lines.append(f"- {k}: {str(v)[:300]}")
        if inp.evidence_summary:
            lines.append("")
            lines.append("## Evidence Summary")
            lines.append(inp.evidence_summary[:1500])
        return "\n".join(lines)

    @staticmethod
    def _schema_json(model: type[BaseModel]) -> str:
        """Return a compact JSON schema for the model."""
        schema = model.model_json_schema()
        props = schema.get("properties", {})
        compact: dict[str, str] = {}
        for name, prop in props.items():
            ptype = prop.get("type", "string")
            desc = prop.get("description", "")
            compact[name] = f"{ptype}: {desc}"
        return json.dumps(compact, indent=2)

    def _render_system_from_template(
        self,
        method_name: str,
        inp: FrameworkCaseInput,
        schema_str: str,
    ) -> str | None:
        """Try to render a prompt template for the given lifecycle method.

        Returns the rendered system prompt string, or ``None`` if no
        template exists (the caller should fall back to dynamic generation).
        """
        slug = self.metadata().id
        mgr = PromptManager.for_framework(slug)
        variables = {
            "system_header": self._system_header(),
            "case_text": self._format_input_for_prompt(inp),
            "output_schema": schema_str,
        }
        return mgr.render(slug, method_name, variables)

    async def _llm_generate_typed(
        self,
        model: type[BaseModel],
        inp: FrameworkCaseInput,
        system_extra: str = "",
        llm: LLMService | None = None,
        template_name: str | None = None,
    ) -> BaseModel:
        """Call the LLM and parse the response into a Pydantic model.

        Args:
            model: Target Pydantic model class.
            inp: FrameworkCaseInput with case data.
            system_extra: Additional system-level instructions (used only
                when no template is available).
            llm: LLM service instance.
            template_name: Optional lifecycle method name for loading a
                prompt template (e.g. ``"analyze_case"``).  When provided
                the system prompt is rendered from the template; when
                absent (or the template is missing), the legacy
                ``system_extra``-based fallback is used.

        Returns:
            An instance of ``model``.

        Raises:
            RuntimeError: If the LLM is unavailable or parsing fails.
        """
        if llm is None:
            return model()

        m = self.metadata()
        schema_str = self._schema_json(model)

        # Try template-based system prompt
        system: str | None = None
        if template_name is not None:
            system = self._render_system_from_template(template_name, inp, schema_str)

        # Fallback to legacy dynamic generation
        if system is None:
            system = (
                f"{self._system_header()}\n\n"
                f"{system_extra}\n\n"
                f"Respond with valid JSON matching this schema exactly:\n"
                f"{schema_str}\n\n"
                "Return ONLY the JSON object. No markdown fences. "
                "No explanatory text before or after."
            )

        prompt_body = self._format_input_for_prompt(inp)
        raw = await llm.generate(prompt=prompt_body, system_prompt=system)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        start = cleaned.index("{")
        end = cleaned.rindex("}")
        data = json.loads(cleaned[start: end + 1])
        return model(**data)

    # ── 1. analyze_case ─────────────────────────────────────────────────────

    async def analyze_case(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> CaseAnalysis:
        """Analyse the case through the framework's clinical lens."""
        m = self.metadata()
        extra = (
            f"As a {m.display_name} practitioner, analyse this case.\n"
            "Identify key clinical features, presenting patterns, "
            "framework-specific observations, severity indicators, "
            "and client strengths/resources."
        )
        try:
            result = await self._llm_generate_typed(
                CaseAnalysis, inp, extra, llm, template_name="analyze_case",
            )
            return result  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("framework.analyze_case_failed", slug=m.id, error=str(exc))
            return CaseAnalysis(
                key_features=[],
                presenting_patterns=[],
                framework_observations=[],
                severity_indicators=[],
                strengths_and_resources=[],
                analysis_summary=f"Analysis unavailable: {exc}",
            )

    # ── 2. retrieve_evidence ────────────────────────────────────────────────

    async def retrieve_evidence(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> EvidenceResult:
        """Retrieve and synthesise evidence relevant to the case."""
        m = self.metadata()
        extra = (
            f"As a {m.display_name} practitioner, retrieve relevant evidence.\n"
            "Identify key findings, synthesise the evidence base, note gaps, "
            "and rate your confidence. If no literature collection is available, "
            "suggest the type of evidence that would be needed."
        )
        try:
            result = await self._llm_generate_typed(
                EvidenceResult, inp, extra, llm, template_name="retrieve_evidence",
            )
            return result  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("framework.retrieve_evidence_failed", slug=m.id, error=str(exc))
            return EvidenceResult(
                retrieved=[],
                synthesis=f"Evidence retrieval unavailable: {exc}",
                key_findings=[],
                evidence_gaps=[],
                confidence=0.0,
            )

    # ── 3. generate_case_conceptualization ──────────────────────────────────

    async def generate_case_conceptualization(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> CaseConceptualization:
        """Generate a framework-specific case conceptualisation."""
        m = self.metadata()
        extra = (
            f"As a {m.display_name} practitioner, generate a case conceptualization.\n"
            "Describe the framework formulation, maintaining factors, core mechanisms, "
            "developmental factors, and provide a summary."
        )
        try:
            result = await self._llm_generate_typed(
                CaseConceptualization, inp, extra, llm,
                template_name="case_conceptualization",
            )
            return result  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("framework.conceptualization_failed", slug=m.id, error=str(exc))
            return CaseConceptualization(
                framework_formulation=f"Conceptualization unavailable: {exc}",
                maintaining_factors=[],
                core_mechanisms=[],
                developmental_factors=[],
                conceptualization_summary=f"Unable to generate: {exc}",
            )

    # ── 4. generate_therapeutic_focus ───────────────────────────────────────

    async def generate_therapeutic_focus(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> TherapeuticFocusResult:
        """Identify the primary therapeutic focus areas."""
        m = self.metadata()
        extra = (
            f"As a {m.display_name} practitioner, identify therapeutic focus areas.\n"
            "Specify primary and secondary focus areas, staging/sequencing, "
            "and the clinical rationale."
        )
        try:
            result = await self._llm_generate_typed(
                TherapeuticFocusResult, inp, extra, llm,
                template_name="therapeutic_focus",
            )
            return result  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("framework.therapeutic_focus_failed", slug=m.id, error=str(exc))
            return TherapeuticFocusResult(
                primary_focus_areas=["Unable to determine focus"],
                secondary_focus_areas=[],
                staging_and_sequencing=f"Staging unavailable: {exc}",
                focus_rationale=f"Rationale unavailable: {exc}",
            )

    # ── 5. generate_intervention_plan ───────────────────────────────────────

    async def generate_intervention_plan(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> InterventionPlanResult:
        """Generate a framework-specific intervention plan."""
        m = self.metadata()
        extra = (
            f"As a {m.display_name} practitioner, generate an intervention plan.\n"
            "List specific interventions with technique and rationale. "
            "Describe the provisional strategy, estimated duration, and expected outcomes."
        )
        try:
            result = await self._llm_generate_typed(
                InterventionPlanResult, inp, extra, llm,
                template_name="intervention_plan",
            )
            return result  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("framework.intervention_plan_failed", slug=m.id, error=str(exc))
            return InterventionPlanResult(
                interventions=[],
                provisional_strategy=f"Plan unavailable: {exc}",
                expected_outcomes=[],
            )

    # ── 6. generate_session_objectives ──────────────────────────────────────

    async def generate_session_objectives(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> SessionObjectivesResult:
        """Define session-level objectives and goals."""
        m = self.metadata()
        extra = (
            f"As a {m.display_name} practitioner, define session objectives.\n"
            "Specify immediate objectives, short-term goals, long-term goals, "
            "and observable progress indicators."
        )
        try:
            result = await self._llm_generate_typed(
                SessionObjectivesResult, inp, extra, llm,
                template_name="session_objectives",
            )
            return result  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("framework.session_objectives_failed", slug=m.id, error=str(exc))
            return SessionObjectivesResult(
                immediate_objectives=[],
                short_term_goals=["Unable to generate goals"],
                long_term_goals=[],
                progress_indicators=[],
            )

    # ── 7. generate_followup_questions ──────────────────────────────────────

    async def generate_followup_questions(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> FollowupQuestionsResult:
        """Suggest follow-up questions for assessment and therapy."""
        m = self.metadata()
        extra = (
            f"As a {m.display_name} practitioner, suggest follow-up questions.\n"
            "Include assessment questions, therapeutic probes, outcome monitoring "
            "questions, and engagement questions."
        )
        try:
            result = await self._llm_generate_typed(
                FollowupQuestionsResult, inp, extra, llm,
                template_name="followup_questions",
            )
            return result  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("framework.followup_questions_failed", slug=m.id, error=str(exc))
            return FollowupQuestionsResult(
                assessment_questions=[],
                therapeutic_probes=[],
                outcome_monitoring=[],
                engagement_questions=[],
            )

    # ── 8. generate_limitations ─────────────────────────────────────────────

    async def generate_limitations(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> LimitationsResult:
        """Describe the limitations and contraindications of this framework."""
        m = self.metadata()
        extra = (
            f"As a {m.display_name} expert, describe limitations.\n"
            "List contraindications, precautions, alternative frameworks that "
            "may be more appropriate, and general caveats."
        )
        try:
            result = await self._llm_generate_typed(
                LimitationsResult, inp, extra, llm, template_name="limitations",
            )
            return result  # type: ignore[return-value]
        except Exception as exc:
            _log.warning("framework.limitations_failed", slug=m.id, error=str(exc))
            return LimitationsResult(
                contraindications=[],
                precautions=[],
                alternative_frameworks=[],
                caveats=[f"Limitations unavailable: {exc}"],
            )

    # ── Orchestrator ────────────────────────────────────────────────────────

    async def run_full_analysis(
        self,
        inp: FrameworkCaseInput,
        llm: LLMService | None = None,
    ) -> FrameworkOutput:
        """Run the full 8-step analysis pipeline and return aggregated output.

        Each lifecycle method is called in sequence.  If a method fails
        the pipeline continues with a fallback result.  Timing is
        recorded per-step and as a total.
        """
        m = self.metadata()
        t_start = time.perf_counter()

        _log.info("framework.run_full_analysis.start", slug=m.id)

        await self.initialize()

        steps: list[tuple[str, Any]] = [
            ("case_analysis", self.analyze_case(inp, llm)),
            ("evidence", self.retrieve_evidence(inp, llm)),
            ("conceptualization", self.generate_case_conceptualization(inp, llm)),
            ("therapeutic_focus", self.generate_therapeutic_focus(inp, llm)),
            ("intervention_plan", self.generate_intervention_plan(inp, llm)),
            ("session_objectives", self.generate_session_objectives(inp, llm)),
            ("followup_questions", self.generate_followup_questions(inp, llm)),
            ("limitations", self.generate_limitations(inp, llm)),
        ]

        results: dict[str, Any] = {}
        errors: list[str] = []

        for name, coro in steps:
            try:
                results[name] = await coro
            except Exception as exc:
                _log.error("framework.step_failed", slug=m.id, step=name, error=str(exc))
                errors.append(f"{name}: {exc}")
                results[name] = None

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        output = FrameworkOutput(
            metadata=m,
            case_analysis=results.get("case_analysis"),
            evidence=results.get("evidence"),
            conceptualization=results.get("conceptualization"),
            therapeutic_focus=results.get("therapeutic_focus"),
            intervention_plan=results.get("intervention_plan"),
            session_objectives=results.get("session_objectives"),
            followup_questions=results.get("followup_questions"),
            limitations=results.get("limitations"),
            processing_ms=round(elapsed_ms, 2),
        )

        if errors:
            output.error = "; ".join(errors)

        _log.info(
            "framework.run_full_analysis.complete",
            slug=m.id,
            elapsed_ms=round(elapsed_ms, 2),
            error_count=len(errors),
        )

        return output
