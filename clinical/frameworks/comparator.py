from __future__ import annotations

from clinical.frameworks.models import (
    FrameworkComparisonResult,
    FrameworkComparisonRow,
    FrameworkResult,
)


class FrameworkComparisonEngine:
    """Compares outputs from multiple frameworks and highlights
    convergences and divergences.

    This is a rule-based engine (no LLM call) that:
      1. Aligns each framework's output along shared dimensions.
      2. Identifies points of convergence and divergence.
      3. Produces a structured comparison table.
      4. Renders the comparison as markdown for inclusion in reports.
    """

    DIMENSIONS = [
        "conceptualisation",
        "key_concepts",
        "therapeutic_approaches",
        "strengths",
        "limitations",
    ]

    DIMENSION_LABELS: dict[str, str] = {
        "conceptualisation": "Conceptualisation",
        "key_concepts": "Key Concepts Applied",
        "therapeutic_approaches": "Therapeutic Approaches",
        "strengths": "Strengths for This Case",
        "limitations": "Limitations / Cautions",
    }

    # ── Public API ──────────────────────────────────────────

    def compare(
        self,
        results: dict[str, FrameworkResult],
    ) -> FrameworkComparisonResult:
        if not results:
            return FrameworkComparisonResult()

        rows: list[FrameworkComparisonRow] = []
        all_dim_texts: dict[str, list[str]] = {}

        for dim in self.DIMENSIONS:
            entries: dict[str, str] = {}
            for fw_name, fw_result in results.items():
                entries[fw_name] = self._get_dimension(fw_result, dim)
            if any(v for v in entries.values()):
                rows.append(FrameworkComparisonRow(
                    dimension=self.DIMENSION_LABELS.get(dim, dim),
                    entries=entries,
                ))

            texts = [v for v in entries.values() if v]
            if texts:
                all_dim_texts[dim] = texts

        convergence = self._find_convergence(results, all_dim_texts)
        divergence = self._find_divergence(results, all_dim_texts)

        return FrameworkComparisonResult(
            rows=rows,
            convergence_areas=convergence,
            divergence_areas=divergence,
            recommendation=self._build_recommendation(results, convergence, divergence),
        )

    def to_comparison_markdown(
        self,
        comparison: FrameworkComparisonResult,
        results: dict[str, FrameworkResult],
    ) -> str:
        """Render a framework comparison as a markdown section.

        Produces a complete ``## 12. Framework Comparison`` section
        suitable for appending to the clinical report.
        """
        lines = ["## 12. Framework Comparison\n"]
        lines.append(
            "The following section presents analyses from multiple therapeutic "
            "frameworks for clinician consideration. "
            "Each framework offers a different lens through which to understand "
            "the client's presentation. All are hypotheses — not diagnoses.\n"
        )

        for fw_name in results:
            fw_result = results[fw_name]
            lines.append(f"### {fw_name}")
            lines.append(f"- **Confidence**: {fw_result.confidence:.2f}")
            if fw_result.conceptualization:
                lines.append(f"- **Conceptualisation**: {fw_result.conceptualization}")
            for k in fw_result.key_concepts[:3]:
                lines.append(f"  - {k}")
            if fw_result.strengths:
                lines.append(f"- **Strengths**: {'; '.join(fw_result.strengths[:2])}")
            if fw_result.limitations:
                lines.append(f"- **Limitations**: {'; '.join(fw_result.limitations[:2])}")
            lines.append("")

        if comparison.rows:
            lines.append("### Side-by-Side Comparison\n")
            for row in comparison.rows:
                lines.append(f"**{row.dimension}**")
                for fw_name, text in row.entries.items():
                    if text:
                        lines.append(f"  - **{fw_name}**: {text}")
                lines.append("")

        if comparison.convergence_areas:
            lines.append("### Points of Convergence\n")
            for c in comparison.convergence_areas:
                lines.append(f"- {c}")
            lines.append("")

        if comparison.divergence_areas:
            lines.append("### Points of Divergence\n")
            for d in comparison.divergence_areas:
                lines.append(f"- {d}")
            lines.append("")

        if comparison.recommendation:
            lines.append("### Recommendation\n")
            lines.append(comparison.recommendation)
            lines.append("")

        return "\n".join(lines)

    # ── Private helpers ─────────────────────────────────────

    def _get_dimension(self, result: FrameworkResult, dim: str) -> str:
        if dim == "conceptualisation":
            return result.conceptualization
        if dim == "key_concepts":
            return "; ".join(result.key_concepts) if result.key_concepts else ""
        if dim == "therapeutic_approaches":
            return "; ".join(result.therapeutic_approaches) if result.therapeutic_approaches else ""
        if dim == "strengths":
            return "; ".join(result.strengths) if result.strengths else ""
        if dim == "limitations":
            return "; ".join(result.limitations) if result.limitations else ""
        return ""

    def _find_convergence(
        self,
        results: dict[str, FrameworkResult],
        _all_dim_texts: dict[str, list[str]],
    ) -> list[str]:
        areas: list[str] = []
        high_conf = [n for n, r in results.items() if r.confidence >= 0.7]
        if len(high_conf) >= 2:
            names = " and ".join(high_conf)
            areas.append(
                f"{names} both show strong fit (confidence \u2265 0.7), "
                f"suggesting either framework may offer a useful lens."
            )
        return areas

    def _find_divergence(
        self,
        results: dict[str, FrameworkResult],
        _all_dim_texts: dict[str, list[str]],
    ) -> list[str]:
        areas: list[str] = []
        confidences = {n: r.confidence for n, r in results.items()}
        if len(confidences) >= 2:
            sorted_by_conf = sorted(confidences.items(), key=lambda x: -x[1])
            best = sorted_by_conf[0]
            areas.append(
                f"{best[0]} shows the strongest fit (confidence {best[1]:.1f}), "
                f"suggesting it may be the most evidence-aligned starting point."
            )
        return areas

    @staticmethod
    def _build_recommendation(
        results: dict[str, FrameworkResult],
        _convergence: list[str],
        _divergence: list[str],
    ) -> str:
        if not results:
            return ""
        best = max(results.values(), key=lambda r: r.confidence)
        return (
            f"Based on the available information, {best.framework_name} "
            f"shows the strongest fit (confidence {best.confidence:.1f}). "
            f"The clinician may consider starting with a {best.framework_name} "
            f"formulation while remaining attentive to aspects highlighted "
            f"by other frameworks. All suggestions are for clinician "
            f"consideration only and do not constitute a prescription."
        )
