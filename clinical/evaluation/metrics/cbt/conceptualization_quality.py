"""
clinical/evaluation/metrics/cbt/conceptualization_quality.py

Evaluates the quality of the CBT framework's case conceptualization.

Scores on seven axes:
  1. Core beliefs — identifies beliefs about self, others, world.
  2. Negative automatic thoughts (NATs) — identifies specific NATs.
  3. Cognitive distortions — names specific distortion types.
  4. Maintaining cycle — describes how patterns interlock.
  5. Behavioural patterns — identifies avoidance, safety behaviours.
  6. Specificity — addresses the specific case details.
  7. No diagnostic labels — avoids diagnostic terminology.

Each axis is scored 0-1. The overall score is the mean.
"""

from __future__ import annotations

import re

from clinical.evaluation.benchmarks.cases import BenchmarkCase
from clinical.evaluation.metrics.base import BaseMetric, MetricResult


class CBTConceptualizationQualityMetric(BaseMetric):
    """Evaluate CBT conceptualization quality against a multi-axis rubric.

    Extracts the CBT framework's conceptualization from the pipeline's
    ``framework_results["cbt"].guidance`` text and scores it for fidelity
    to the cognitive-behavioural model.
    """

    name = "cbt_conceptualization_quality"

    # ── CBT model component keywords ─────────────────────────────────────────

    _CORE_BELIEF_KEYWORDS: set[str] = {
        "core belief", "core belief about", "belief about self",
        "belief about others", "belief about the world",
        "schema", "self-schema", "fundamental belief",
    }

    _NAT_KEYWORDS: set[str] = {
        "negative automatic thought", "automatic thought",
        "nat", "thought model", "cognitive triad",
    }

    _COGNITIVE_DISTORTION_KEYWORDS: set[str] = {
        "cognitive distortion", "thinking distortion",
        "unhelpful thinking", "thinking pattern",
        "catastrophising", "catastrophizing",
        "black-and-white thinking", "all-or-nothing thinking",
        "mind reading", "personalisation", "personalization",
        "overgeneralisation", "overgeneralization",
        "emotional reasoning", "should statement",
        "labelling", "labeling", "magnification",
        "disqualifying the positive", "mental filter",
        "fortune-telling", "jumping to conclusions",
    }

    _MAINTAINING_CYCLE_KEYWORDS: set[str] = {
        "maintaining cycle", "maintenance cycle",
        "vicious cycle", "vicious circle",
        "maintaining factor", "feedback loop",
        "cycle of", "mutually reinforcing",
        "perpetuating", "maintains",
    }

    _BEHAVIOURAL_KEYWORDS: set[str] = {
        "avoidance", "safety behaviour", "safety behavior",
        "withdrawal", "inactivity", "escape",
        "neutralising", "neutralizing", "ritual",
        "reassurance seeking", "checking",
        "behavioural pattern", "behavioral pattern",
    }

    _DEVELOPMENTAL_KEYWORDS: set[str] = {
        "developmental", "early experience", "childhood",
        "origins", "formative", "learning history",
    }

    # ── Diagnostic label patterns (same as CBTPlugin) ────────────────────────

    _DIAGNOSTIC_PATTERNS: list[re.Pattern[str]] = [
        re.compile(
            r"(?i)\b(major depressive disorder|clinical depression|mdd|"
            r"generalized anxiety disorder|generalised anxiety disorder|gad|"
            r"social anxiety disorder|sad|social phobia|"
            r"panic disorder|agoraphobia|"
            r"obsessive-compulsive disorder|obsessive compulsive disorder|ocd|"
            r"post-traumatic stress disorder|posttraumatic stress disorder|ptsd|"
            r"bipolar (i|ii|disorder)|"
            r"borderline personality disorder|bpd|"
            r"eating disorder|anorexia nervosa|bulimia nervosa|binge-eating disorder|"
            r"attention deficit hyperactivity disorder|adhd|"
            r"autism spectrum disorder|asd)\b"
        ),
    ]

    async def score_case(
        self,
        case: BenchmarkCase,
        pipeline_state: dict,
    ) -> MetricResult:
        guidance, guidance_available = self._extract_cbt_guidance(pipeline_state)
        if not guidance:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                score=0.0,
                details={"error": "no CBT framework guidance available"},
            )

        text_lower = guidance.lower()
        expected_components = (
            case.ground_truth.expected_cbt_conceptualization_components
        )

        axis_scores = {
            "core_beliefs": self._score_core_beliefs(text_lower),
            "automatic_thoughts": self._score_automatic_thoughts(text_lower),
            "cognitive_distortions": self._score_cognitive_distortions(
                text_lower, case
            ),
            "maintaining_cycle": self._score_maintaining_cycle(text_lower),
            "behavioural_patterns": self._score_behavioural(text_lower),
            "developmental_origins": self._score_developmental(text_lower),
            "specificity": self._score_specificity(
                text_lower, case, expected_components
            ),
            "no_diagnostic_labels": self._score_no_diagnoses(text_lower),
        }

        overall = sum(axis_scores.values()) / len(axis_scores)
        overall = round(min(1.0, overall), 4)

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            score=overall,
            details={
                "axes": {k: round(v, 4) for k, v in axis_scores.items()},
                "guidance_excerpt": guidance[:600],
                "guidance_length": len(guidance),
                "expected_components": expected_components,
            },
        )

    # ── Extraction ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_cbt_guidance(
        pipeline_state: dict,
    ) -> tuple[str, bool]:
        """Extract CBT framework guidance text from pipeline state."""
        framework_results = pipeline_state.get("framework_results", {})
        cbt_result = framework_results.get("cbt")
        if cbt_result and hasattr(cbt_result, "guidance") and cbt_result.guidance:
            return cbt_result.guidance, True
        return "", False

    # ── Axis scorers ─────────────────────────────────────────────────────────

    @staticmethod
    def _score_core_beliefs(text: str) -> float:
        score = sum(1 for kw in CBTConceptualizationQualityMetric._CORE_BELIEF_KEYWORDS if kw in text)
        return min(1.0, score * 0.15)

    @staticmethod
    def _score_automatic_thoughts(text: str) -> float:
        score = sum(1 for kw in CBTConceptualizationQualityMetric._NAT_KEYWORDS if kw in text)
        return min(1.0, score * 0.15)

    @staticmethod
    def _score_cognitive_distortions(
        text: str,
        case: BenchmarkCase,
    ) -> float:
        expected_d = [
            d.lower()
            for d in case.ground_truth.expected_cognitive_distortions
        ]
        if not expected_d:
            return 0.5
        matched = sum(1 for d in expected_d if d in text)
        # Also check generic distortion keywords
        generic_matches = sum(
            1
            for kw in CBTConceptualizationQualityMetric._COGNITIVE_DISTORTION_KEYWORDS
            if kw in text
        )
        raw = (matched / len(expected_d)) + (generic_matches * 0.05)
        return min(1.0, raw)

    @staticmethod
    def _score_maintaining_cycle(text: str) -> float:
        score = sum(
            1
            for kw in CBTConceptualizationQualityMetric._MAINTAINING_CYCLE_KEYWORDS
            if kw in text
        )
        return min(1.0, score * 0.15)

    @staticmethod
    def _score_behavioural(text: str) -> float:
        score = sum(
            1
            for kw in CBTConceptualizationQualityMetric._BEHAVIOURAL_KEYWORDS
            if kw in text
        )
        return min(1.0, score * 0.08)

    @staticmethod
    def _score_developmental(text: str) -> float:
        score = sum(
            1
            for kw in CBTConceptualizationQualityMetric._DEVELOPMENTAL_KEYWORDS
            if kw in text
        )
        return min(1.0, score * 0.15)

    @staticmethod
    def _score_specificity(
        text: str,
        case: BenchmarkCase,
        expected_components: list[str],
    ) -> float:
        # Score based on keyword overlap with expected components
        if not expected_components:
            return 0.5
        component_text = " ".join(expected_components).lower()
        # Extract meaningful words from expected components
        content_words = {
            w
            for phrase in expected_components
            for w in phrase.lower().split()
            if len(w) > 3 and w not in ("about", "with", "from", "that", "this")
        }
        matches = sum(1 for w in content_words if w in text)
        ratio = matches / max(len(content_words), 1)
        return min(1.0, ratio * 1.25)

    @staticmethod
    def _score_no_diagnoses(text: str) -> float:
        for pat in CBTConceptualizationQualityMetric._DIAGNOSTIC_PATTERNS:
            matches = pat.findall(text)
            if matches:
                # Penalise per diagnostic label found (up to 1.0 deduction)
                penalty = min(1.0, len(matches) * 0.25)
                return round(1.0 - penalty, 4)
        return 1.0
