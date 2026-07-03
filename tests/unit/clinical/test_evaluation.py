from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clinical.evaluation.benchmarks.cases import CASE_DEPRESSION, CASE_ANXIETY
from clinical.evaluation.metrics.citation import CitationAccuracyMetric
from clinical.evaluation.metrics.formulation import FormulationQualityMetric
from clinical.evaluation.metrics.hallucination import HallucinationRateMetric
from clinical.evaluation.metrics.helpfulness import ClinicalHelpfulnessMetric
from clinical.evaluation.metrics.missing_info import MissingInfoDetectionMetric
from clinical.evaluation.metrics.retrieval import RetrievalPrecisionMetric
from clinical.evaluation.metrics.relevance import TherapeuticRelevanceMetric
from clinical.evaluation.report import EvaluationReport, format_report


def _make_response_mock(markdown: str):
    m = MagicMock()
    m.markdown = markdown
    m.sections_generated = 11
    m.confidence = 0.7
    m.model_dump = MagicMock(return_value={"markdown": markdown})
    return m


def _make_chunk(text: str, score: float = 0.9, source: str = "dsm5.pdf", page: int = 1):
    c = MagicMock()
    c.text = text
    c.score = score
    c.source = source
    c.page = page
    c.chunk_id = f"{source}__p{page:04d}__c0000"
    c.rank = 1
    c.metadata = {"source": source, "page": page}
    return c


_CLEAN_MARKDOWN = """## 1. CASE SUMMARY

The client is a 29-year-old first-time mother presenting with persistent low mood and anxiety.

## 4. CLINICAL FORMULATION

The client's difficulties may reflect a pattern of cognitive and behavioural factors maintaining low mood. This formulation is tentative and based on available information. The evidence base supports a formulation-driven approach.

## 6. MISSING INFORMATION

Information about current support systems and alcohol use would help clarify the clinical picture.

## 7. EVIDENCE SUMMARY

Research suggests CBT is effective for postnatal depression (Smith et al., 2020). The evidence supports psychological intervention.

## 8. THERAPEUTIC FOCUS

- Cognitive patterns
- Behavioural activation
- Mother-infant bonding

## 9. SUGGESTED INTERVENTION DIRECTIONS

CBT and behavioural activation are recommended evidence-based approaches.

## 10. REFERENCES

- Smith, J. et al. (2020). CBT for postnatal depression. Journal of Clinical Psychology.
"""


class TestRetrievalPrecisionMetric:
    async def test_perfect_precision(self):
        metric = RetrievalPrecisionMetric()
        state = {
            "retrieved_chunks": [
                _make_chunk("postnatal depression treatment options", 0.95),
                _make_chunk("CBT for postpartum depression", 0.90),
                _make_chunk("anxiety in new mothers interventions", 0.85),
                _make_chunk("mother-infant bonding research", 0.80),
                _make_chunk("perinatal mental health guidelines", 0.75),
            ],
        }
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score >= 0.8

    async def test_no_chunks(self):
        metric = RetrievalPrecisionMetric()
        state = {"retrieved_chunks": []}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 0.0

    async def test_zero_relevance(self):
        metric = RetrievalPrecisionMetric()
        state = {
            "retrieved_chunks": [
                _make_chunk("irrelevant text about unrelated topic", 0.95),
                _make_chunk("completely unrelated content", 0.90),
            ],
        }
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 0.0


class TestCitationAccuracyMetric:
    async def test_all_citations_matched(self):
        metric = CitationAccuracyMetric()
        markdown = """## 7. EVIDENCE SUMMARY

CBT is effective (Smith et al., 2020).

## 10. REFERENCES

- Smith, J. et al. (2020). CBT for depression.
"""
        state = {"response": _make_response_mock(markdown)}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 1.0

    async def test_unmatched_citations(self):
        metric = CitationAccuracyMetric()
        markdown = """## 7. EVIDENCE SUMMARY

CBT is effective (Smith et al., 2020). Also see (Martinez, 2022).

## 10. REFERENCES

- Smith, J. et al. (2020). CBT for depression.
"""
        state = {"response": _make_response_mock(markdown)}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score < 1.0
        assert result.details["unmatched"] >= 1

    async def test_no_citations(self):
        metric = CitationAccuracyMetric()
        state = {"response": _make_response_mock("No citations here.")}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 0.0


class TestFormulationQualityMetric:
    async def test_high_quality_formulation(self):
        metric = FormulationQualityMetric()
        state = {
            "formulation": MagicMock(
                case_summary="The client's difficulties may reflect cognitive patterns consistent with postnatal depression. "
                             "Evidence from Smith et al. (2020) suggests CBT is effective. "
                             "This is a tentative formulation with alternative considerations noted.",
                model_dump=MagicMock(return_value={}),
            ),
        }
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score >= 0.5

    async def test_poor_formulation_with_diagnosis_language(self):
        metric = FormulationQualityMetric()
        state = {
            "formulation": MagicMock(
                case_summary="The patient definitely has major depressive disorder. "
                             "This is clearly a diagnosis.",
                model_dump=MagicMock(return_value={}),
            ),
        }
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score < 0.5

    async def test_no_formulation_data(self):
        metric = FormulationQualityMetric()
        state = {}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 0.0


class TestTherapeuticRelevanceMetric:
    async def test_relevant_interventions_match(self):
        metric = TherapeuticRelevanceMetric()
        state = {
            "response": _make_response_mock(
                "## 9. SUGGESTED INTERVENTION DIRECTIONS\n\nCBT is recommended. "
                "Behavioural activation may help. Psychoeducation for the client."
            ),
        }
        result = await metric.score_case(CASE_DEPRESSION, state)
        expected = len(CASE_DEPRESSION.ground_truth.acceptable_interventions)
        assert result.score > 0 or expected == 0

    async def test_no_response(self):
        metric = TherapeuticRelevanceMetric()
        state = {}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 0.0


class TestMissingInfoDetectionMetric:
    async def test_detects_some_missing_info(self):
        metric = MissingInfoDetectionMetric()

        mock_item = MagicMock()
        mock_item.info_gap = "Current support systems are not described"
        mock_item.clinical_relevance = "Support is important for postnatal depression"

        state = {
            "missing_info": MagicMock(
                missing_information=[mock_item],
            ),
        }
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score >= 0.0

    async def test_no_missing_info(self):
        metric = MissingInfoDetectionMetric()
        state = {}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 0.0


class TestHallucinationRateMetric:
    async def test_no_hallucinations(self):
        metric = HallucinationRateMetric()
        markdown = """## 7. EVIDENCE SUMMARY

CBT is effective (Smith et al., 2020).

## 10. REFERENCES

- Smith, J. et al. (2020). CBT therapy.
"""
        state = {"response": _make_response_mock(markdown)}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 0.0

    async def test_hallucinated_citations(self):
        metric = HallucinationRateMetric()
        markdown = """## 7. EVIDENCE SUMMARY

CBT is effective (Smith et al., 2020). Also see (MadeUp, 2023).

## 10. REFERENCES

- Smith, J. et al. (2020). CBT therapy.
"""
        state = {"response": _make_response_mock(markdown)}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score > 0.0

    async def test_no_response(self):
        metric = HallucinationRateMetric()
        state = {}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 1.0


class TestClinicalHelpfulnessMetric:
    async def test_helpful_response(self):
        metric = ClinicalHelpfulnessMetric()
        state = {
            "response": _make_response_mock(
                "The clinician validates the client's experience. A risk assessment is important. "
                "Specific recommendations include CBT and psychoeducation. "
                "Consider referral to a perinatal mental health service."
            ),
        }
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score >= 0.2

    async def test_empty_response(self):
        metric = ClinicalHelpfulnessMetric()
        state = {}
        result = await metric.score_case(CASE_DEPRESSION, state)
        assert result.score == 0.0


class TestEvaluationReport:
    def test_format_report_creates_output(self):
        report = EvaluationReport()
        output = format_report(report)
        assert isinstance(output, str)
        assert len(output) > 50

    def test_aggregate_empty(self):
        report = EvaluationReport(results=[])
        agg = report.aggregate()
        assert isinstance(agg, dict)

    def test_aggregate_with_results(self):
        from clinical.evaluation.metrics.base import MetricResult

        results = [
            MetricResult(metric="test_metric", case_id="case_1", score=0.8),
            MetricResult(metric="test_metric", case_id="case_2", score=0.6),
        ]
        report = EvaluationReport(results=[MagicMock(
            case_id="case_1", pipeline_elapsed_ms=100, metric_results=[results[0]],
            pipeline_state={}, error=None,
        ), MagicMock(
            case_id="case_2", pipeline_elapsed_ms=200, metric_results=[results[1]],
            pipeline_state={}, error=None,
        )])
        agg = report.aggregate()
        assert "test_metric" in agg
        assert agg["test_metric"]["mean"] == 0.7


class TestBenchmarkCases:
    def test_all_cases_have_required_fields(self):
        from clinical.evaluation.benchmarks.cases import ALL_CASES
        for case in ALL_CASES:
            assert case.case_id
            assert case.title
            assert case.clinical_text
            assert case.input_type in ("case_study", "symptom_list", "single_statement")
            assert case.ground_truth.expected_topics
            assert len(case.clinical_text) >= 50
