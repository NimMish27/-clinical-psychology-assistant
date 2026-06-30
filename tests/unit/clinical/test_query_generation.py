from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical.case_understanding.models import (
    CaseUnderstandingResult,
    ClinicalPresentation,
    ConfidenceRating,
    ContextualFactors,
    DemographicInfo,
    Duration,
    ExtractedField,
    OverallSeverity,
    PreviousTreatment,
    Severity,
    TreatmentHistory,
)
from clinical.query_generation import (
    OptimizedQuery,
    QueryCategory,
    QueryGenerationResult,
    RetrievalQueryGenerator,
)


# ═══════════════════════════════════════════════════════════════
# Model unit tests
# ═══════════════════════════════════════════════════════════════

class TestOptimizedQuery:
    def test_basic_creation(self):
        q = OptimizedQuery(
            query="CBT for burnout",
            category=QueryCategory.TREATMENT,
            weight=2.0,
            rationale="Targets evidence-based treatment for burnout",
        )
        assert q.query == "CBT for burnout"
        assert q.category == QueryCategory.TREATMENT
        assert q.weight == 2.0

    def test_min_weight(self):
        q = OptimizedQuery(
            query="test query",
            category=QueryCategory.DIAGNOSTIC,
            weight=0.1,
            rationale="valid reason here",
        )
        assert q.weight == 0.1

    def test_max_weight(self):
        q = OptimizedQuery(
            query="test query",
            category=QueryCategory.DIAGNOSTIC,
            weight=3.0,
            rationale="valid reason here",
        )
        assert q.weight == 3.0

    def test_frozen(self):
        q = OptimizedQuery(
            query="test query",
            category=QueryCategory.DIAGNOSTIC,
            weight=1.0,
            rationale="valid reason here",
        )
        with pytest.raises(Exception):
            q.query = "changed"

    def test_with_expansion_of(self):
        q = OptimizedQuery(
            query="burnout assessment MBI",
            category=QueryCategory.ASSESSMENT,
            weight=1.2,
            rationale="Assessment for exhaustion",
            expansion_of="exhaustion",
        )
        assert q.expansion_of == "exhaustion"

    def test_min_query_length(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            OptimizedQuery(
                query="ab",
                category=QueryCategory.DIAGNOSTIC,
                weight=1.0,
                rationale="test",
            )

    def test_min_rationale_length(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            OptimizedQuery(
                query="valid query",
                category=QueryCategory.DIAGNOSTIC,
                weight=1.0,
                rationale="ab",
            )

    def test_all_categories(self):
        for cat in QueryCategory:
            q = OptimizedQuery(
                query=f"test {cat.value}",
                category=cat,
                weight=1.0,
                rationale="test rationale",
            )
            assert q.category == cat


class TestQueryGenerationResult:
    def test_basic_creation(self):
        queries = [
            OptimizedQuery(
                query="query one",
                category=QueryCategory.DIAGNOSTIC,
                weight=2.0,
                rationale="valid rationale",
            ),
        ]
        r = QueryGenerationResult(queries=queries)
        assert len(r.queries) == 1
        assert r.generated_at is not None
        assert r.generation_ms >= 0

    def test_min_queries(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryGenerationResult(queries=[])

    def test_to_query_strings_returns_sorted(self):
        queries = [
            OptimizedQuery(query="low", category=QueryCategory.DIAGNOSTIC, weight=1.0, rationale="valid reason one"),
            OptimizedQuery(query="high", category=QueryCategory.DIAGNOSTIC, weight=3.0, rationale="valid reason two"),
            OptimizedQuery(query="mid", category=QueryCategory.DIAGNOSTIC, weight=2.0, rationale="valid reason three"),
        ]
        r = QueryGenerationResult(queries=queries)
        strings = r.to_query_strings()
        assert strings == ["high", "mid", "low"]

    def test_to_weighted_queries(self):
        queries = [
            OptimizedQuery(query="query one", category=QueryCategory.DIAGNOSTIC, weight=1.5, rationale="valid rationale"),
        ]
        r = QueryGenerationResult(queries=queries)
        weighted = r.to_weighted_queries()
        assert weighted == [{"query": "query one", "weight": 1.5}]

    def test_frozen(self):
        q = OptimizedQuery(query="test query", category=QueryCategory.DIAGNOSTIC, weight=1.0, rationale="valid reason")
        r = QueryGenerationResult(queries=[q])
        with pytest.raises(Exception):
            r.queries = []


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.generate = AsyncMock()
    return m


@pytest.fixture
def generator(mock_llm):
    return RetrievalQueryGenerator(llm=mock_llm)


@pytest.fixture
def burnout_case() -> CaseUnderstandingResult:
    return CaseUnderstandingResult(
        demographic=DemographicInfo(
            age=ExtractedField(value=22, confidence=ConfidenceRating.HIGH, source_text="22-year-old"),
            gender=ExtractedField(value="female", confidence=ConfidenceRating.HIGH, source_text="female"),
            occupation=ExtractedField(value="college student", confidence=ConfidenceRating.HIGH, source_text="college student"),
        ),
        clinical_presentation=ClinicalPresentation(
            presenting_concerns=[
                ExtractedField(value="exhaustion", confidence=ConfidenceRating.HIGH, source_text="exhaustion"),
                ExtractedField(value="academic burnout", confidence=ConfidenceRating.HIGH, source_text="academic burnout"),
            ],
            symptoms=[
                ExtractedField(value="chronic fatigue", confidence=ConfidenceRating.HIGH, source_text="always tired"),
                ExtractedField(value="insomnia", confidence=ConfidenceRating.MEDIUM, source_text="difficulty sleeping"),
            ],
            emotional_indicators=[
                ExtractedField(value="emotional detachment", confidence=ConfidenceRating.HIGH, source_text="feeling numb"),
                ExtractedField(value="irritability", confidence=ConfidenceRating.MEDIUM, source_text="easily annoyed"),
            ],
            behavioural_indicators=[
                ExtractedField(value="social withdrawal", confidence=ConfidenceRating.HIGH, source_text="isolating"),
            ],
            duration=Duration(value=3, unit="months", original_text="for 3 months"),
        ),
        contextual_factors=ContextualFactors(
            stressors=[
                ExtractedField(value="academic pressure", confidence=ConfidenceRating.HIGH, source_text="heavy course load"),
            ],
            protective_factors=[
                ExtractedField(value="supportive family", confidence=ConfidenceRating.MEDIUM, source_text="close to family"),
            ],
            risk_factors=[
                ExtractedField(value="perfectionism", confidence=ConfidenceRating.HIGH, source_text="high self-standards"),
            ],
            functional_impairment=ExtractedField(
                value="dropped two classes",
                confidence=ConfidenceRating.HIGH,
                source_text="had to drop two classes",
            ),
            social_context=ExtractedField(
                value="lives in dorm, limited social circle",
                confidence=ConfidenceRating.MEDIUM,
            ),
        ),
        treatment_history=TreatmentHistory(),
        overall_severity=OverallSeverity(
            severity=Severity.MODERATE,
            confidence=ConfidenceRating.MEDIUM,
            rationale="Burnout affecting academic functioning",
        ),
        raw_text="22-year-old college student with exhaustion, perfectionism and emotional detachment.",
    )


_MOCK_FULL_RESPONSE = """{
  "queries": [
    {
      "query": "academic burnout treatment",
      "category": "treatment",
      "weight": 2.5,
      "rationale": "Targets evidence-based treatment for academic burnout",
      "expansion_of": "exhaustion"
    },
    {
      "query": "student burnout assessment MBI",
      "category": "assessment",
      "weight": 1.5,
      "rationale": "Identify appropriate burnout measurement tools",
      "expansion_of": null
    },
    {
      "query": "perfectionism CBT interventions",
      "category": "treatment",
      "weight": 2.0,
      "rationale": "Evidence-based treatment for perfectionism",
      "expansion_of": "perfectionism"
    },
    {
      "query": "emotional detachment burnout mechanism",
      "category": "phenomenology",
      "weight": 1.8,
      "rationale": "Understand emotional detachment in burnout",
      "expansion_of": "emotional detachment"
    },
    {
      "query": "college student mental health guidelines",
      "category": "contextual",
      "weight": 1.2,
      "rationale": "Age and demographic-specific clinical considerations",
      "expansion_of": null
    },
    {
      "query": "chronic fatigue insomnia differential diagnosis",
      "category": "diagnostic",
      "weight": 2.8,
      "rationale": "Rule out alternative diagnoses for fatigue and sleep disturbance",
      "expansion_of": "chronic fatigue"
    }
  ],
  "raw_text_summary": "Academic burnout in a college student with perfectionism and emotional detachment"
}"""

_MOCK_MINIMAL_RESPONSE = """{
  "queries": [],
  "raw_text_summary": null
}"""


# ═══════════════════════════════════════════════════════════════
# Generator tests
# ═══════════════════════════════════════════════════════════════

class TestRetrievalQueryGenerator:
    @pytest.mark.asyncio
    async def test_generates_multi_category_queries(self, generator, mock_llm, burnout_case):
        mock_llm.generate.return_value = _MOCK_FULL_RESPONSE

        result = await generator.generate(burnout_case)

        assert len(result.queries) == 6
        categories = {q.category for q in result.queries}
        assert QueryCategory.TREATMENT in categories
        assert QueryCategory.DIAGNOSTIC in categories
        assert QueryCategory.PHENOMENOLOGY in categories
        assert result.raw_text_summary is not None
        assert result.generation_ms > 0

    @pytest.mark.asyncio
    async def test_queries_sorted_by_weight_descending(self, generator, mock_llm, burnout_case):
        mock_llm.generate.return_value = _MOCK_FULL_RESPONSE

        result = await generator.generate(burnout_case)
        weights = [q.weight for q in result.queries]
        assert weights == sorted(weights, reverse=True)

    @pytest.mark.asyncio
    async def test_to_query_strings(self, generator, mock_llm, burnout_case):
        mock_llm.generate.return_value = _MOCK_FULL_RESPONSE

        result = await generator.generate(burnout_case)
        strings = result.to_query_strings()
        assert len(strings) == 6
        assert all(isinstance(s, str) for s in strings)
        # First string should be highest weight (diagnostic: 2.8)
        assert strings[0] == "chronic fatigue insomnia differential diagnosis"

    @pytest.mark.asyncio
    async def test_to_weighted_queries(self, generator, mock_llm, burnout_case):
        mock_llm.generate.return_value = _MOCK_FULL_RESPONSE

        result = await generator.generate(burnout_case)
        weighted = result.to_weighted_queries()
        assert len(weighted) == 6
        assert weighted[0]["weight"] >= weighted[-1]["weight"]

    @pytest.mark.asyncio
    async def test_minimal_input_returns_fallback(self, generator, mock_llm):
        minimal_case = CaseUnderstandingResult(raw_text="I feel sad.")
        mock_llm.generate.return_value = _MOCK_MINIMAL_RESPONSE

        result = await generator.generate(minimal_case)
        # Empty list from LLM -> rule-based fallback should produce some queries
        assert len(result.queries) >= 1

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rules(self, generator, mock_llm, burnout_case):
        mock_llm.generate.side_effect = RuntimeError("Ollama unavailable")

        result = await generator.generate(burnout_case)
        # Must return rule-based queries, not crash
        assert len(result.queries) >= 1
        assert all(isinstance(q, OptimizedQuery) for q in result.queries)

    @pytest.mark.asyncio
    async def test_llm_returns_malformed_json(self, generator, mock_llm, burnout_case):
        mock_llm.generate.return_value = "this is not json"

        result = await generator.generate(burnout_case)
        # Must fall back to rule-based
        assert len(result.queries) >= 1

    @pytest.mark.asyncio
    async def test_llm_returns_json_with_markdown_fence(self, generator, mock_llm, burnout_case):
        mock_llm.generate.return_value = f"```json\n{_MOCK_FULL_RESPONSE}\n```"

        result = await generator.generate(burnout_case)
        assert len(result.queries) == 6

    @pytest.mark.asyncio
    async def test_llm_returns_partial_invalid_queries(self, generator, mock_llm, burnout_case):
        response = """{
          "queries": [
            {"query": "valid query", "category": "diagnostic", "weight": 1.0, "rationale": "valid rationale text"},
            {"query": "no category", "weight": 1.0, "rationale": "missing category here"},
            {"query": "", "category": "diagnostic", "weight": 1.0, "rationale": "empty string query"},
            {"query": "ok", "category": "unknown_category", "weight": 1.0, "rationale": "bad category text"},
            {"query": "ok", "category": "treatment", "weight": "not_a_number", "rationale": "bad weight text"}
          ],
          "raw_text_summary": null
        }"""
        mock_llm.generate.return_value = response

        result = await generator.generate(burnout_case)
        # Only valid queries should be included; invalid ones silently dropped
        assert len(result.queries) >= 1
        # "valid query" should be present
        queries_text = [q.query for q in result.queries]
        assert "valid query" in queries_text

    @pytest.mark.asyncio
    async def test_empty_input(self, generator, mock_llm):
        empty_case = CaseUnderstandingResult(raw_text="")
        mock_llm.generate.return_value = _MOCK_FULL_RESPONSE

        result = await generator.generate(empty_case)
        # Should still return queries via LLM (empty case is still a valid prompt)
        assert len(result.queries) >= 1


class TestRuleBasedFallback:
    """Test the rule-based fallback directly by triggering LLM failure."""

    @pytest.mark.asyncio
    async def test_rule_based_with_burnout_case(self, generator, mock_llm, burnout_case):
        mock_llm.generate.side_effect = RuntimeError("fail")

        result = await generator.generate(burnout_case)

        assert len(result.queries) >= 1
        # Should include treatment query for burnout
        queries_text = " ".join(q.query for q in result.queries).lower()
        assert "burnout" in queries_text or "exhaust" in queries_text
        assert "perfection" in queries_text

    @pytest.mark.asyncio
    async def test_rule_based_with_depression_case(self, generator, mock_llm):
        case = CaseUnderstandingResult(
            demographic=DemographicInfo(
                age=ExtractedField(value=35, confidence=ConfidenceRating.HIGH, source_text="35"),
                occupation=ExtractedField(value="teacher", confidence=ConfidenceRating.MEDIUM, source_text="teacher"),
            ),
            clinical_presentation=ClinicalPresentation(
                presenting_concerns=[
                    ExtractedField(value="depressed mood", confidence=ConfidenceRating.HIGH, source_text="sad"),
                ],
                symptoms=[
                    ExtractedField(value="anhedonia", confidence=ConfidenceRating.MEDIUM, source_text="no interest"),
                    ExtractedField(value="low energy", confidence=ConfidenceRating.MEDIUM, source_text="tired"),
                ],
                emotional_indicators=[
                    ExtractedField(value="sadness", confidence=ConfidenceRating.HIGH, source_text="sad"),
                ],
            ),
            contextual_factors=ContextualFactors(
                risk_factors=[
                    ExtractedField(value="social isolation", confidence=ConfidenceRating.MEDIUM, source_text="alone"),
                ],
            ),
            overall_severity=OverallSeverity(severity=Severity.MODERATE),
            raw_text="Depressed teacher.",
        )
        mock_llm.generate.side_effect = RuntimeError("fail")

        result = await generator.generate(case)

        assert len(result.queries) >= 1
        queries_text = " ".join(q.query for q in result.queries).lower()
        assert "depress" in queries_text or "anhedonia" in queries_text.lower()

    @pytest.mark.asyncio
    async def test_rule_based_expands_to_treatment(self, generator, mock_llm, burnout_case):
        mock_llm.generate.side_effect = RuntimeError("fail")

        result = await generator.generate(burnout_case)

        queries_text = " ".join(q.query for q in result.queries).lower()
        # The symptom "insomnia" should map to "insomnia CBT-I treatment"
        assert "insomnia" in queries_text or "fatigue" in queries_text

    @pytest.mark.asyncio
    async def test_rule_based_expands_to_assessment(self, generator, mock_llm, burnout_case):
        mock_llm.generate.side_effect = RuntimeError("fail")

        result = await generator.generate(burnout_case)

        queries_text = " ".join(q.query for q in result.queries).lower()
        # Perfectionism should map to "FMPS Frost perfectionism scale"
        assert "perfection" in queries_text

    @pytest.mark.asyncio
    async def test_rule_based_age_specific(self, generator, mock_llm, burnout_case):
        mock_llm.generate.side_effect = RuntimeError("fail")

        result = await generator.generate(burnout_case)

        queries_text = " ".join(q.query for q in result.queries)
        assert "22" in queries_text or "college" in queries_text.lower()

    @pytest.mark.asyncio
    async def test_rule_based_with_minimal_case(self, generator, mock_llm):
        case = CaseUnderstandingResult(raw_text="need help")
        mock_llm.generate.side_effect = RuntimeError("fail")

        result = await generator.generate(case)
        # Minimal case with no extracted fields should still return default query
        assert len(result.queries) == 1
        assert result.queries[0].query == "clinical assessment treatment guidelines"


class TestRuleBasedMapping:
    """Test the mapping functions directly."""

    def test_map_to_treatment(self, mock_llm):
        gen = RetrievalQueryGenerator(llm=mock_llm)
        assert "CBT" in gen._map_to_treatment("anxiety symptoms")
        assert "depression" in gen._map_to_treatment("feeling depressed")
        assert "DBT" in gen._map_to_treatment("self-harm thoughts")
        assert "burnout" in gen._map_to_treatment("burned out at work")
        assert gen._map_to_treatment("unknown symptom") is None

    def test_map_to_scale(self, mock_llm):
        gen = RetrievalQueryGenerator(llm=mock_llm)
        assert "PHQ-9" in gen._map_to_scale("depressed mood")
        assert "GAD-7" in gen._map_to_scale("generalized anxiety")
        assert "MBI" in gen._map_to_scale("burnout scale")
        assert "FMPS" in gen._map_to_scale("perfectionism tendencies")
        assert gen._map_to_scale("unknown symptom") is None


class TestGeneratorStandalone:
    def test_can_be_instantiated_without_pipeline(self, mock_llm):
        gen = RetrievalQueryGenerator(llm=mock_llm)
        assert gen is not None

    def test_result_is_valid_pydantic(self):
        q = OptimizedQuery(query="test query", category=QueryCategory.DIAGNOSTIC, weight=1.0, rationale="valid reason")
        r = QueryGenerationResult(queries=[q])
        assert r.model_dump() is not None
