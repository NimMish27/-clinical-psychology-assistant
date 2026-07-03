from __future__ import annotations

import re
import time
from typing import Any

from clinical.safety_validation.models import (
    SafetyIssue,
    SafetyIssueCategory,
    SafetyValidationResult,
    Severity,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

_KNOWN_DSM_DISORDERS: list[str] = [
    "major depressive disorder", "mdd",
    "generalised anxiety disorder", "generalized anxiety disorder", "gad",
    "social anxiety disorder", "social phobia",
    "panic disorder",
    "agoraphobia",
    "specific phobia",
    "obsessive-compulsive disorder", "ocd",
    "post-traumatic stress disorder", "ptsd",
    "acute stress disorder",
    "bipolar i disorder", "bipolar ii disorder", "bipolar disorder",
    "schizophrenia",
    "schizoaffective disorder",
    "borderline personality disorder", "bpd",
    "antisocial personality disorder", "aspd",
    "narcissistic personality disorder",
    "avoidant personality disorder",
    "dependent personality disorder",
    "anorexia nervosa",
    "bulimia nervosa",
    "binge-eating disorder",
    "insomnia disorder",
    "adjustment disorder",
    "persistent depressive disorder", "dysthymia",
    "somatic symptom disorder",
    "illness anxiety disorder",
    "dissociative identity disorder",
    "dissociative amnesia",
    "gender dysphoria",
    "autism spectrum disorder", "asd",
    "attention-deficit/hyperactivity disorder", "adhd",
    "conduct disorder",
    "oppositional defiant disorder", "odd",
    "substance use disorder", "alcohol use disorder",
    "gambling disorder",
]

_SECTION_HEADER_RE = re.compile(r"^##\s+\d+\.\s+(.+)$", re.MULTILINE)

_OVERCONFIDENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdefinitely\b", re.IGNORECASE), "Absolute term 'definitely' implies certainty beyond clinical evidence"),
    (re.compile(r"\bcertainly\b", re.IGNORECASE), "Absolute term 'certainly' overstates confidence"),
    (re.compile(r"\bundoubtedly\b", re.IGNORECASE), "Absolute term 'undoubtedly' overstates confidence"),
    (re.compile(r"\bguaranteed?\b", re.IGNORECASE), "Term 'guarantee' implies certainty not possible in clinical practice"),
    (re.compile(r"\balways\b", re.IGNORECASE), "Absolute term 'always' overgeneralises"),
    (re.compile(r"\bnever\b", re.IGNORECASE), "Absolute term 'never' overgeneralises"),
    (re.compile(r"\binvariably\b", re.IGNORECASE), "Absolute term 'invariably' overgeneralises"),
    (re.compile(r"\bno doubt\b", re.IGNORECASE), "Phrase 'no doubt' overstates clinical certainty"),
    (re.compile(r"\bthis (?:is|was) clearly\b", re.IGNORECASE), "Phrase 'clearly' overstates certainty"),
    (re.compile(r"\bit is obvious\b", re.IGNORECASE), "Phrase 'it is obvious' overstates certainty"),
]

_DIAGNOSIS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bdiagnos(?:ed|is)\s+(?:with|of)\s+", re.IGNORECASE),
    re.compile(r"\b(?:meets|meet)\s+(?:the\s+)?(?:DSM-5|ICD-11|DSM|ICD)\s+(?:criteria|diagnostic criteria)\s+for\b", re.IGNORECASE),
    re.compile(r"\bs(?:uffer|uffers|uffering)\s+from\b", re.IGNORECASE),
    re.compile(r"\b(?:patient|client)\s+has\s+(?:\w+\s+){0,4}(?:disorder|depression|anxiety|syndrome)\b", re.IGNORECASE),
    re.compile(r"\ba\s+case\s+of\s+", re.IGNORECASE),
    re.compile(r"\bis\s+(?:a|an)\s+(?:classic|typical|textbook)\s+case\b", re.IGNORECASE),
]

_MEDICATION_NAMES: str = (
    r"(?:ssri|snri|maoi|benzo|stimulant|antidepressant|antipsychotic"
    r"|mood\s+stabiliser|mood\s+stabilizer"
    r"|sertraline|fluoxetine|escitalopram|paroxetine|citalopram"
    r"|venlafaxine|duloxetine|bupropion|mirtazapine"
    r"|olanzapine|risperidone|quetiapine|aripiprazole"
    r"|lithium|lamotrigine|valproate|valproic\s+acid|carbamazepine"
    r"|alprazolam|clonazepam|lorazepam|diazepam"
    r"|methylphenidate|dexamphetamine|amphetamine"
    r"|propranolol|gabapentin|pregabalin)"
)

_MEDICATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:prescribe|prescribed?|dosing?|dosage|dose\s+of)\s+" + _MEDICATION_NAMES + r"\b", re.IGNORECASE),
    re.compile(r"\b(?:should|ought|must|need)\s+(?:try|consider|start|take|begin)\s+(?:" + _MEDICATION_NAMES + r")\b", re.IGNORECASE),
    re.compile(r"\b" + _MEDICATION_NAMES + r"\s+\d+\s*mg\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*mg\s+(?:of\s+)?" + _MEDICATION_NAMES + r"\b", re.IGNORECASE),
]

_CITATION_HEADERS: set[str] = {
    "references", "10. references", "reference",
}

_DIAGNOSIS_HEADERS: set[str] = {
    "4. clinical formulation", "clinical formulation",
    "5. possible differential considerations", "possible differential considerations",
}

_FABRICATED_CITATION_RE = re.compile(
    r"(?:\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?\s*,?\s*\d{4}\)"
    r"|[A-Z][a-z]+\s+et\s+al\.?\s*\(?\d{4}\)?"
    r"|[A-Z][a-z]+\s*&\s*[A-Z][a-z]+\s*\(?\d{4}\)?)",
)

_CONFIDENTIALITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:patient'?s\s+)?(?:full\s+)?(?:name|address|phone|email|ssn|social\s+security|national\s+insurance)\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+-\s+(?:born|aged|dob|date\s+of\s+birth)\b"),
]

_INTERVENTION_SECTION_RE = re.compile(
    r"##\s+\d+\.\s+(?:suggested\s+)?intervention\s+directions?",
    re.IGNORECASE,
)
_MEDICATION_DISCLAIMER = (
    "\n\n> **Clinical caution**: All treatment suggestions are evidence-informed "
    "recommendations for clinician consideration. Medication-related suggestions "
    "must be evaluated by a licensed prescriber. No content in this report "
    "constitutes a prescription or medical advice."
)

_OVERCONFIDENT_DISCLAIMER = (
    "\n\n> **Note on confidence**: Clinical formulations are working hypotheses, "
    "not diagnoses. Confidence levels reflect the quality and completeness of "
    "available information, not diagnostic certainty."
)

_NON_DIAGNOSIS_OVERRIDE = (
    "\n\n> **Important**: This report does not constitute a clinical diagnosis. "
    "All formulations and differential considerations are hypotheses for "
    "clinician consideration, based on the available information. A full "
    "clinical assessment is required before any diagnostic formulation."
)


def _extract_sections(markdown: str) -> dict[str, str]:
    """Split markdown into named sections by ## headers."""
    sections: dict[str, str] = {}
    current_header: str | None = None
    current_lines: list[str] = []

    for line in markdown.split("\n"):
        m = _SECTION_HEADER_RE.match(line)
        if m:
            if current_header:
                sections[current_header.strip().lower()] = "\n".join(current_lines).strip()
            current_header = m.group(1).strip()
            current_lines = []
        elif current_header:
            current_lines.append(line)

    if current_header:
        sections[current_header.strip().lower()] = "\n".join(current_lines).strip()

    return sections


def _check_overconfident_language(markdown: str) -> list[SafetyIssue]:
    issues: list[SafetyIssue] = []
    for pattern, explanation in _OVERCONFIDENT_PATTERNS:
        for match in pattern.finditer(markdown):
            start = max(0, match.start() - 40)
            end = min(len(markdown), match.end() + 40)
            excerpt = markdown[start:end].replace("\n", " ")
            issues.append(SafetyIssue(
                category=SafetyIssueCategory.OVERCONFIDENT_LANGUAGE,
                severity=Severity.LOW,
                location=_find_section(markdown, match.start()),
                excerpt=excerpt[:200],
                explanation=explanation,
                suggestion=f"Replace '{match.group()}' with more tentative language (e.g. 'may', 'suggests', 'is often associated with').",
            ))
    return issues


def _find_section(markdown: str, pos: int) -> str:
    before = markdown[:pos]
    headers = list(re.finditer(r"^##\s+\d+\.\s+(.+)$", before, re.MULTILINE))
    if headers:
        return headers[-1].group(1).strip()
    return "preamble"


def _check_unsupported_diagnoses(markdown: str, sections: dict[str, str]) -> list[SafetyIssue]:
    issues: list[SafetyIssue] = []
    diagnosis_section = None
    for h in _DIAGNOSIS_HEADERS:
        if h in sections:
            diagnosis_section = sections[h]
            break

    formulation_header = next(
        (h for h in sections if "clinical formulation" in h or "differential" in h),
        None,
    )

    for pattern in _DIAGNOSIS_PATTERNS:
        for match in pattern.finditer(markdown):
            start = max(0, match.start() - 30)
            end = min(len(markdown), match.end() + 80)
            excerpt = markdown[start:end].replace("\n", " ")
            loc = _find_section(markdown, match.start())

            for disorder in _KNOWN_DSM_DISORDERS:
                if disorder in excerpt.lower():
                    issues.append(SafetyIssue(
                        category=SafetyIssueCategory.UNSUPPORTED_DIAGNOSIS,
                        severity=Severity.MEDIUM,
                        location=loc,
                        excerpt=excerpt[:200],
                        explanation=(
                            f"Language suggesting a diagnosis ('{disorder}') was detected. "
                            "Clinical formulations should use tentative, hypothesis-driven language."
                        ),
                        suggestion=(
                            f"Rephrase to describe symptoms and patterns without labelling. "
                            f"Instead of stating '{disorder}', describe the observed features "
                            f"and note diagnostic considerations tentatively."
                        ),
                    ))
                    break
    return issues


def _normalize_citation(text: str) -> str:
    """Remove punctuation, parentheses, leading dashes, and extra whitespace."""
    result = text.strip().lstrip("- ").strip()
    result = result.replace("(", "").replace(")", "").replace(",", "").replace(".", "")
    return re.sub(r"\s+", " ", result).lower().strip()


def _check_hallucinated_citations(markdown: str, sections: dict[str, str]) -> list[SafetyIssue]:
    issues: list[SafetyIssue] = []

    ref_section = None
    for h in _CITATION_HEADERS:
        if h in sections:
            ref_section = sections[h]
            break

    normalized_refs: list[str] = []
    if ref_section:
        for line in ref_section.split("\n"):
            norm = _normalize_citation(line)
            if norm and len(norm) > 20:
                normalized_refs.append(norm)

    for match in _FABRICATED_CITATION_RE.finditer(markdown):
        citation = match.group()
        loc = _find_section(markdown, match.start())
        if loc.lower() in _CITATION_HEADERS:
            continue

        normalized_citation = _normalize_citation(citation)
        if not normalized_citation or len(normalized_citation) < 10:
            continue

        is_supported = False
        for ref in normalized_refs:
            c_words = normalized_citation.split()
            r_words = ref.split()
            overlap = sum(1 for w in c_words if w in r_words)
            if overlap >= 3 and overlap / max(len(c_words), 1) >= 0.4:
                is_supported = True
                break

        if not is_supported and ("smith" in normalized_citation or "et al" in normalized_citation):
            issues.append(SafetyIssue(
                category=SafetyIssueCategory.HALLUCINATION,
                severity=Severity.HIGH,
                location=loc,
                excerpt=citation[:200],
                explanation=(
                    f"Citation '{citation}' does not match any reference in the reference section. "
                    "LLMs may fabricate citations."
                ),
                suggestion=(
                    f"Verify that '{citation}' matches an actual source in the reference list, "
                    "or remove it and replace with a supported citation."
                ),
            ))

    return issues


def _check_missing_citations(markdown: str, sections: dict[str, str]) -> list[SafetyIssue]:
    issues: list[SafetyIssue] = []

    evidence_section = None
    for h in sections:
        if "evidence" in h.lower():
            evidence_section = sections[h]
            break

    if evidence_section:
        sentences = re.split(r'(?<=[.!?])\s+', evidence_section)
        for i, sent in enumerate(sentences):
            if len(sent) < 40:
                continue
            if not re.search(r'\([^)]*(?:19|20)\d{2}[^)]*\)', sent) and not re.search(r'\[source\]|according to', sent, re.IGNORECASE):
                for disorder_name in _KNOWN_DSM_DISORDERS[:20]:
                    if disorder_name in sent.lower():
                        issues.append(SafetyIssue(
                            category=SafetyIssueCategory.MISSING_CITATION,
                            severity=Severity.MEDIUM,
                            location="evidence summary",
                            excerpt=sent[:200],
                            explanation=(
                                f"Clinical claim about '{disorder_name}' without a supporting citation."
                            ),
                            suggestion=(
                                "Add a reference citation (e.g. DSM-5, APA guideline, or specific study) "
                                "to support this claim, or rephrase as a tentative observation."
                            ),
                        ))
                        break
    return issues


def _check_ethical_concerns(markdown: str) -> list[SafetyIssue]:
    issues: list[SafetyIssue] = []
    for pattern in _CONFIDENTIALITY_PATTERNS:
        for match in pattern.finditer(markdown):
            start = max(0, match.start() - 40)
            end = min(len(markdown), match.end() + 40)
            excerpt = markdown[start:end].replace("\n", " ")
            issues.append(SafetyIssue(
                category=SafetyIssueCategory.ETHICAL_CONCERN,
                severity=Severity.HIGH,
                location=_find_section(markdown, match.start()),
                excerpt=excerpt[:200],
                explanation="Potential personally identifiable information (PII) detected in the response.",
                suggestion="Remove or redact any identifiable information. Use generic descriptors instead (e.g. 'the client' rather than names or identifiers).",
            ))
            break
    return issues


def _check_unsafe_recommendations(markdown: str) -> list[SafetyIssue]:
    issues: list[SafetyIssue] = []
    for pattern in _MEDICATION_PATTERNS:
        for match in pattern.finditer(markdown):
            start = max(0, match.start() - 60)
            end = min(len(markdown), match.end() + 60)
            excerpt = markdown[start:end].replace("\n", " ")
            issues.append(SafetyIssue(
                category=SafetyIssueCategory.UNSAFE_RECOMMENDATION,
                severity=Severity.HIGH,
                location=_find_section(markdown, match.start()),
                excerpt=excerpt[:200],
                explanation="This appears to be a medication-specific suggestion, which requires a licensed prescriber.",
                suggestion="Remove dosing/prescribing language. If medication is relevant, add a disclaimer and refer to a qualified prescriber.",
            ))
    return issues


def _revise_response(
    markdown: str,
    sections: dict[str, str],
    issues: list[SafetyIssue],
) -> tuple[str, str]:
    """Apply automatic revisions to address issues. Returns (revised_markdown, summary)."""
    revised = markdown
    changes: list[str] = []

    has_medication = any(
        i.category == SafetyIssueCategory.UNSAFE_RECOMMENDATION
        for i in issues
    )
    has_diagnosis = any(
        i.category == SafetyIssueCategory.UNSUPPORTED_DIAGNOSIS
        for i in issues
    )
    has_overconfident = any(
        i.category == SafetyIssueCategory.OVERCONFIDENT_LANGUAGE
        for i in issues
    )

    # Replace overconfident language
    if has_overconfident:
        replacements = {
            r"\bdefinitely\b": "likely",
            r"\bcertainly\b": "probably",
            r"\bundoubtedly\b": "likely",
            r"\bguaranteed?\b": "may",
            r"\balways\b": "often",
            r"\bnever\b": "rarely",
            r"\binvariably\b": "frequently",
            r"\bno doubt\b": "it appears",
            r"\bthis (?:is|was) clearly\b": "this is",
            r"\bit is obvious\b": "it appears",
        }
        for pattern_str, replacement in replacements.items():
            revised = re.sub(pattern_str, replacement, revised, flags=re.IGNORECASE)
        changes.append("Replaced overconfident/absolute language with tentative phrasing")

    if has_medication:
        intervention_section = _INTERVENTION_SECTION_RE.search(revised)
        if intervention_section:
            revised += _MEDICATION_DISCLAIMER
            changes.append("Added medication caution disclaimer")
        else:
            revised += _MEDICATION_DISCLAIMER
            changes.append("Added medication caution disclaimer")

    if has_diagnosis:
        if _NON_DIAGNOSIS_OVERRIDE not in revised:
            revised += _NON_DIAGNOSIS_OVERRIDE
            changes.append("Added non-diagnosis disclaimer above clinical sections")

    if has_overconfident and not has_diagnosis:
        if _OVERCONFIDENT_DISCLAIMER not in revised:
            revised += _OVERCONFIDENT_DISCLAIMER
            changes.append("Added note on confidence/hypothesis nature of formulations")

    summary = "; ".join(changes) if changes else "No automatic revisions needed"
    return revised, summary


def _determine_verdict(issues: list[SafetyIssue]) -> str:
    severities = [i.severity for i in issues]
    if Severity.HIGH in severities:
        return "needs_review"
    if Severity.MEDIUM in severities:
        return "minor_issues"
    if issues:
        return "minor_issues"
    return "clean"


class SafetyValidator:
    """Validates clinical responses for safety, accuracy, and ethical concerns.

    Performs six checks:
        1. Unsupported diagnoses — statements implying diagnosis
        2. Hallucinations — fabricated or unmatched citations
        3. Missing citations — clinical claims without supporting references
        4. Overconfident language — absolute/definitive phrasing
        5. Ethical concerns — potential PII or confidentiality issues
        6. Unsafe recommendations — medication advice, dosage suggestions

    If issues are found, the validator will revise the response and return
    both the validation report and the updated markdown.
    """

    async def validate(
        self,
        markdown: str,
    ) -> SafetyValidationResult:
        t_start = time.perf_counter()

        if not markdown or len(markdown.strip()) < 20:
            return SafetyValidationResult(
                markdown=markdown,
                original_markdown=markdown,
                overall_verdict="clean",
            )

        sections = _extract_sections(markdown)
        issues: list[SafetyIssue] = []

        issues.extend(_check_overconfident_language(markdown))
        issues.extend(_check_unsupported_diagnoses(markdown, sections))
        issues.extend(_check_hallucinated_citations(markdown, sections))
        issues.extend(_check_missing_citations(markdown, sections))
        issues.extend(_check_ethical_concerns(markdown))
        issues.extend(_check_unsafe_recommendations(markdown))

        revised_md = markdown
        revision_summary = ""
        was_revised = False
        if issues:
            revised_md, revision_summary = _revise_response(markdown, sections, issues)
            was_revised = revised_md != markdown

        overall_verdict = _determine_verdict(issues)

        _log.info(
            "safety_validation.complete",
            issues_found=len(issues),
            verdict=overall_verdict,
            was_revised=was_revised,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return SafetyValidationResult(
            markdown=revised_md,
            original_markdown=markdown,
            issues=issues,
            was_revised=was_revised,
            revision_summary=revision_summary,
            overall_verdict=overall_verdict,
            validation_ms=round(elapsed_ms, 2),
        )
