from __future__ import annotations

import pytest

from clinical.safety_validation.models import (
    SafetyIssueCategory,
    SafetyValidationResult,
    Severity,
)
from clinical.safety_validation.validator import SafetyValidator


@pytest.fixture
def clean_report() -> str:
    return """## 1. CASE SUMMARY

The client is a 32-year-old presenting with low mood and anxiety.

## 2. PRESENTING CONCERNS

- Low mood for the past several weeks
- Difficulty sleeping

## 3. OBSERVED SYMPTOMS

- Depressed mood
- Reduced energy

## 4. CLINICAL FORMULATION

The client's difficulties may reflect a pattern of avoidant coping and
negative self-schemas that maintain their low mood. This formulation is
tentative and based on the available information.

## 7. EVIDENCE SUMMARY

Research suggests that cognitive-behavioural approaches are effective
for mood-related concerns (Smith et al., 2020). The evidence base
supports a formulation-driven approach to treatment.

## 8. THERAPEUTIC FOCUS

- Cognitive patterns
- Behavioural activation

## 10. REFERENCES

- Smith, J. et al. (2020). Cognitive-behavioural therapy for mood disorders.
  Journal of Clinical Psychology, 76(3), 234-250.
"""


@pytest.fixture
def overconfident_report() -> str:
    return """## 1. CASE SUMMARY

The client is definitely suffering from depression. This is clearly a case of MDD.

## 4. CLINICAL FORMULATION

The patient undoubtedly has major depressive disorder. CBT will certainly help.
"""


@pytest.fixture
def unsafe_report() -> str:
    return """## 9. SUGGESTED INTERVENTION DIRECTIONS

The patient should try sertraline 50mg daily. CBT is recommended.

## 10. REFERENCES

- National Institute for Health and Care Excellence. (2022). Depression in adults.
"""


@pytest.fixture
def hallucinated_citation_report() -> str:
    return """## 7. EVIDENCE SUMMARY

According to Smith et al. (2023), CBT is highly effective for GAD.
Martinez & Chen (2022) found similar results in a randomised trial.

## 10. REFERENCES

- National Institute for Health and Care Excellence. (2022). Depression in adults.
"""


@pytest.fixture
def pii_report() -> str:
    return """## 1. CASE SUMMARY

John Smith, a 45-year-old male from Chicago, reports feeling depressed.
The client's phone number is 555-123-4567.
"""


class TestSafetyValidator:
    async def test_clean_report(self, clean_report):
        validator = SafetyValidator()
        result = await validator.validate(clean_report)
        assert result.overall_verdict == "clean"
        assert len(result.issues) == 0
        assert not result.was_revised

    async def test_empty_markdown(self):
        validator = SafetyValidator()
        result = await validator.validate("")
        assert result.overall_verdict == "clean"
        assert len(result.issues) == 0

    async def test_short_markdown(self):
        validator = SafetyValidator()
        result = await validator.validate("Short")
        assert result.overall_verdict == "clean"

    async def test_overconfident_language_detected(self, overconfident_report):
        validator = SafetyValidator()
        result = await validator.validate(overconfident_report)
        overconfident = [i for i in result.issues if i.category == SafetyIssueCategory.OVERCONFIDENT_LANGUAGE]
        assert len(overconfident) >= 1
        assert "definitely" in overconfident[0].excerpt or "definitely" in overconfident[0].explanation

    async def test_overconfident_revision(self, overconfident_report):
        validator = SafetyValidator()
        result = await validator.validate(overconfident_report)
        assert result.was_revised
        assert "likely" in result.markdown.lower() or "probably" in result.markdown.lower()

    async def test_unsupported_diagnosis_detected(self, overconfident_report):
        validator = SafetyValidator()
        result = await validator.validate(overconfident_report)
        diagnoses = [i for i in result.issues if i.category == SafetyIssueCategory.UNSUPPORTED_DIAGNOSIS]
        assert len(diagnoses) >= 1

    async def test_unsupported_diagnosis_adds_disclaimer(self, overconfident_report):
        validator = SafetyValidator()
        result = await validator.validate(overconfident_report)
        assert result.was_revised
        assert "does not constitute a clinical diagnosis" in result.markdown

    async def test_unsafe_recommendation_detected(self, unsafe_report):
        validator = SafetyValidator()
        result = await validator.validate(unsafe_report)
        unsafe = [i for i in result.issues if i.category == SafetyIssueCategory.UNSAFE_RECOMMENDATION]
        assert len(unsafe) >= 1

    async def test_unsafe_recommendation_adds_disclaimer(self, unsafe_report):
        validator = SafetyValidator()
        result = await validator.validate(unsafe_report)
        assert result.was_revised
        assert "medication" in result.revision_summary.lower() or "caution" in result.markdown.lower()

    async def test_hallucinated_citation_detected(self, hallucinated_citation_report):
        validator = SafetyValidator()
        result = await validator.validate(hallucinated_citation_report)
        hallucinations = [i for i in result.issues if i.category == SafetyIssueCategory.HALLUCINATION]
        assert len(hallucinations) >= 1

    async def test_pii_detected(self, pii_report):
        validator = SafetyValidator()
        result = await validator.validate(pii_report)
        ethical = [i for i in result.issues if i.category == SafetyIssueCategory.ETHICAL_CONCERN]
        assert len(ethical) >= 1
        assert any("555-123-4567" in i.excerpt or "John Smith" in i.excerpt for i in ethical)

    async def test_verdict_needs_review(self, unsafe_report):
        validator = SafetyValidator()
        result = await validator.validate(unsafe_report)
        assert result.overall_verdict == "needs_review"

    async def test_verdict_clean(self, clean_report):
        validator = SafetyValidator()
        result = await validator.validate(clean_report)
        assert result.overall_verdict == "clean"

    async def test_result_contains_original_markdown(self, clean_report):
        validator = SafetyValidator()
        result = await validator.validate(clean_report)
        assert result.original_markdown == clean_report

    async def test_severity_high_for_unsafe_recommendation(self, unsafe_report):
        validator = SafetyValidator()
        result = await validator.validate(unsafe_report)
        high = [i for i in result.issues if i.severity == Severity.HIGH]
        assert len(high) >= 1

    async def test_result_has_timestamp(self, clean_report):
        validator = SafetyValidator()
        result = await validator.validate(clean_report)
        assert result.validated_at is not None

    async def test_result_has_validation_ms(self, clean_report):
        validator = SafetyValidator()
        result = await validator.validate(clean_report)
        assert result.validation_ms >= 0

    async def test_model_validation_result_serializable(self, clean_report):
        validator = SafetyValidator()
        result = await validator.validate(clean_report)
        d = result.model_dump()
        assert d["overall_verdict"] == "clean"
        assert d["was_revised"] is False
        assert isinstance(d["issues"], list)
        assert "original_markdown" in d
        assert "markdown" in d

    async def test_partially_clean_report_with_minor_issues(self):
        markdown = """## 1. CASE SUMMARY

The client is definitely experiencing low mood. This may be related to work stress.

## 4. CLINICAL FORMULATION

The formulation considers cognitive and behavioural factors.

## 7. EVIDENCE SUMMARY

Research supports a formulation-driven approach.
"""
        validator = SafetyValidator()
        result = await validator.validate(markdown)
        assert len(result.issues) >= 1
        assert result.overall_verdict in ("minor_issues", "needs_review")

    async def test_non_diagnosis_phrasing_not_flagged(self):
        markdown = """## 4. CLINICAL FORMULATION

The client presents with features consistent with a mood-related difficulty.
This is a tentative formulation for clinician consideration.
"""
        validator = SafetyValidator()
        result = await validator.validate(markdown)
        diagnoses = [i for i in result.issues if i.category == SafetyIssueCategory.UNSUPPORTED_DIAGNOSIS]
        assert len(diagnoses) == 0

    async def test_cited_claim_not_marked_missing(self):
        markdown = """## 7. EVIDENCE SUMMARY

Cognitive-behavioural therapy is effective for anxiety disorders (Smith et al., 2020).

## 10. REFERENCES

- Smith, J. et al. (2020). CBT for anxiety disorders.
"""
        validator = SafetyValidator()
        result = await validator.validate(markdown)
        missing = [i for i in result.issues if i.category == SafetyIssueCategory.MISSING_CITATION]
        assert len(missing) == 0
