from __future__ import annotations

import json
import re
import time
from typing import Any

from clinical.case_understanding.models import CaseUnderstandingResult
from clinical.llm import LLMService
from clinical.query_generation.models import (
    OptimizedQuery,
    QueryCategory,
    QueryGenerationResult,
)
from app_logging.logger import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical information retrieval specialist. Given the extracted case \
information below, generate 4-8 optimized search queries to retrieve the most \
relevant clinical evidence from a knowledge base of clinical documents \
(DSM-5, ICD-11, treatment guidelines, research papers, assessment tools).

Cover these categories across the queries:
1. DIAGNOSTIC — criteria, differentials, classification (weight 2.5-3.0)
2. TREATMENT — evidence-based interventions, efficacy, guidelines (weight 2.0-2.5)
3. PHENOMENOLOGY — symptom presentation, mechanisms, course (weight 1.5-2.0)
4. CONTEXTUAL — psychosocial, demographic, cultural factors (weight 1.0-1.5)
5. ASSESSMENT — screening tools, scales, measurement (weight 1.0-1.5)
6. RISK — risk factors, safety, prognosis (weight 1.5-2.5)

Rules:
- Each query must be a concise search statement (3-15 words), not a question
- Use terminology a clinician would search for (e.g. "CBT for generalized anxiety", not "What is the treatment for anxiety?")
- Expand the case into related concepts the knowledge base may use (e.g. if the case mentions "exhaustion", also include "chronic fatigue" or "burnout")
- Assign higher weights to queries that target the most critical clinical concerns
- Each query must include a rationale explaining what evidence it aims to surface

Respond EXACTLY in this JSON format. No markdown fences. No text outside the JSON.

{
  "queries": [
    {
      "query": "search term here",
      "category": "diagnostic",
      "weight": 2.5,
      "rationale": "why this query improves retrieval",
      "expansion_of": null
    }
  ],
  "raw_text_summary": "brief summary of what the queries target"
}

Allowed categories: diagnostic, treatment, phenomenology, contextual, assessment, risk
"""


class RetrievalQueryGenerator:
    """Generate optimized retrieval queries from extracted case understanding.

    Analyzes the extracted case information and produces multiple targeted
    search queries across diagnostic, treatment, phenomenological,
    contextual, assessment, and risk categories.  Designed to be reusable
    by LangGraph agents — no pipeline dependency.

    Usage::

        gen = RetrievalQueryGenerator(llm_service)
        result = await gen.generate(case_understanding)
        queries = result.to_query_strings()          # ["CBT for burnout", ...]
        weighted = result.to_weighted_queries()       # [{"query": "...", "weight": 2.5}, ...]
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def generate(
        self,
        case: CaseUnderstandingResult,
    ) -> QueryGenerationResult:
        """Generate optimized queries from a case understanding result."""
        t_start = time.perf_counter()

        raw_text_summary: str | None = None
        try:
            raw = await self._llm.generate(
                prompt=self._build_prompt(case),
                system_prompt=_SYSTEM_PROMPT,
            )
            data = self._parse_response(raw)
            queries = self._validate_queries(data.get("queries", []))
            raw_text_summary = data.get("raw_text_summary")
        except Exception as exc:
            _log.warning(
                "query_generation.llm_fallback",
                error=str(exc),
            )
            queries = self._rule_based_fallback(case)

        if not queries:
            queries = [self._default_query()]

        queries_sorted = sorted(queries, key=lambda q: q.weight, reverse=True)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return QueryGenerationResult(
            queries=queries_sorted,
            raw_text_summary=raw_text_summary,
            generation_ms=round(elapsed_ms, 2),
        )

    def _build_prompt(self, case: CaseUnderstandingResult) -> str:
        flat = case.to_flat_dict()
        sections: list[str] = []

        sections.append("=== CASE INFORMATION ===")

        if flat.get("age") or flat.get("gender") or flat.get("occupation"):
            parts = []
            for k in ("age", "gender", "occupation"):
                v = flat.get(k)
                if v:
                    parts.append(f"{k}: {v}")
            sections.append("Demographics: " + "; ".join(parts))

        def fmt_list(items: list[Any], label: str) -> None:
            if items:
                sections.append(f"{label}: " + ", ".join(str(i) for i in items))

        fmt_list(flat.get("presenting_concerns", []), "Presenting concerns")
        fmt_list(flat.get("symptoms", []), "Symptoms")
        fmt_list(flat.get("emotional_indicators", []), "Emotional indicators")
        fmt_list(flat.get("behavioural_indicators", []), "Behavioural indicators")
        fmt_list(flat.get("stressors", []), "Stressors")
        fmt_list(flat.get("protective_factors", []), "Protective factors")
        fmt_list(flat.get("risk_factors", []), "Risk factors")

        if flat.get("functional_impairment"):
            sections.append(f"Functional impairment: {flat['functional_impairment']}")
        if flat.get("social_context"):
            sections.append(f"Social context: {flat['social_context']}")
        if flat.get("duration"):
            sections.append(f"Duration: {flat['duration']}")

        prev_tx = flat.get("previous_treatment", [])
        if prev_tx:
            tx_strs = [f"{t['modality']} ({t['response']})" for t in prev_tx if t.get("modality")]
            sections.append("Previous treatment: " + "; ".join(tx_strs))

        if flat.get("severity") and flat["severity"] != "unspecified":
            sections.append(f"Severity: {flat['severity']}")

        return "\n".join(sections) if len(sections) > 1 else "No case information available."

    def _parse_response(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        start = cleaned.index("{")
        end = cleaned.rindex("}")
        return json.loads(cleaned[start : end + 1])

    def _validate_queries(self, raw: list[dict[str, Any]]) -> list[OptimizedQuery]:
        validated: list[OptimizedQuery] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("query"):
                continue
            try:
                category_raw = item.get("category", "phenomenology")
                if category_raw not in {c.value for c in QueryCategory}:
                    category_raw = "phenomenology"
                validated.append(
                    OptimizedQuery(
                        query=str(item["query"]).strip(),
                        category=QueryCategory(category_raw),
                        weight=float(item.get("weight", 1.0)),
                        rationale=str(item.get("rationale", "No rationale provided")).strip(),
                        expansion_of=str(item["expansion_of"]).strip()
                        if item.get("expansion_of")
                        else None,
                    )
                )
            except (ValueError, TypeError) as exc:
                _log.warning("query_generation.invalid_query", error=str(exc), item=item)
        return validated

    def _rule_based_fallback(self, case: CaseUnderstandingResult) -> list[OptimizedQuery]:
        flat = case.to_flat_dict()
        queries: list[OptimizedQuery] = []
        seen: set[str] = set()

        def add(q: str, cat: QueryCategory, w: float, reason: str, src: str | None = None) -> None:
            ql = q.lower().strip()
            if ql not in seen and len(ql) >= 3:
                seen.add(ql)
                queries.append(
                    OptimizedQuery(
                        query=q.strip(),
                        category=cat,
                        weight=w,
                        rationale=reason,
                        expansion_of=src,
                    )
                )

        # Expand presenting concerns and symptoms into search queries
        for item in flat.get("presenting_concerns", []):
            s = str(item).strip()
            if s:
                add(s, QueryCategory.PHENOMENOLOGY, 1.8, f"Primary presenting concern: {s}")

        for item in flat.get("symptoms", []):
            s = str(item).strip()
            if s:
                add(s, QueryCategory.PHENOMENOLOGY, 1.5, f"Reported symptom: {s}")

        for item in flat.get("emotional_indicators", []):
            s = str(item).strip()
            if s:
                add(
                    f"{s} emotional presentation",
                    QueryCategory.PHENOMENOLOGY,
                    1.3,
                    f"Affective presentation: {s}",
                )

        # Diagnostic queries from symptoms and concerns
        all_symptoms: list[str] = [
            str(i) for i in (flat.get("symptoms", [])
                             + flat.get("presenting_concerns", [])
                             + flat.get("emotional_indicators", []))
            if i
        ]
        if all_symptoms:
            primary = all_symptoms[:3]
            add(
                " ".join(primary),
                QueryCategory.DIAGNOSTIC,
                2.5,
                "Diagnostic criteria for primary symptom cluster",
            )

        # Treatment queries
        treatment_terms: list[str] = []
        for item in all_symptoms:
            t = self._map_to_treatment(item)
            if t:
                treatment_terms.append(t)
        if treatment_terms:
            for t in treatment_terms[:3]:
                add(
                    t,
                    QueryCategory.TREATMENT,
                    2.0,
                    f"Evidence-based treatment for: {t}",
                    src=next((s for s in all_symptoms if s.lower() in t.lower()), None),
                )

        # Contextual queries
        for item in flat.get("stressors", []):
            s = str(item).strip()
            if s:
                add(
                    f"{s} psychosocial stressor",
                    QueryCategory.CONTEXTUAL,
                    1.2,
                    f"Psychosocial context: {s}",
                )

        for item in flat.get("risk_factors", []):
            s = str(item).strip()
            if s:
                add(
                    s,
                    QueryCategory.RISK,
                    2.0,
                    f"Risk factor assessment: {s}",
                )

        # Demographic-specific queries
        age = flat.get("age")
        if age:
            add(
                f"mental health {age} year old",
                QueryCategory.CONTEXTUAL,
                1.0,
                "Age-specific clinical considerations",
            )

        occupation = flat.get("occupation")
        if occupation:
            add(
                f"{occupation} mental health",
                QueryCategory.CONTEXTUAL,
                1.0,
                f"Occupation-specific mental health factors",
            )

        # Assessment queries
        for item in all_symptoms:
            scale = self._map_to_scale(item)
            if scale:
                add(scale, QueryCategory.ASSESSMENT, 1.2, f"Assessment scale for: {item}")

        return queries

    def _map_to_treatment(self, symptom: str) -> str | None:
        """Map a symptom/concern to a common treatment query term."""
        sl = symptom.lower()
        mappings: dict[str, str] = {
            "anxiety": "anxiety disorders treatment CBT",
            "depress": "depression treatment evidence-based",
            "trauma": "trauma-informed therapy PTSD treatment",
            "burn": "burnout intervention prevention",
            "exhaust": "chronic fatigue management",
            "insomnia": "insomnia CBT-I treatment",
            "suicidal": "suicide risk assessment intervention",
            "adhd": "ADHD treatment adult",
            "ocd": "OCD ERP treatment",
            "eating": "eating disorder treatment guidelines",
            "substance": "substance use disorder treatment",
            "grief": "grief therapy complicated bereavement",
            "anger": "anger management CBT",
            "self-harm": "self-harm risk assessment intervention DBT",
            "psychosis": "psychosis early intervention antipsychotic",
            "personality": "personality disorder DBT schema therapy",
            "bipolar": "bipolar disorder mood stabilizer psychosocial",
            "panic": "panic disorder CBT interoceptive exposure",
            "social": "social anxiety CBT exposure",
            "perfection": "perfectionism treatment CBT self-compassion",
            "procrastination": "procrastination CBT executive function",
            "loneliness": "loneliness social connection intervention",
            "relationship": "relationship conflict couples therapy",
            "academic": "academic stress student mental health",
        }
        for key, treatment in mappings.items():
            if key in sl:
                return treatment
        return None

    def _map_to_scale(self, symptom: str) -> str | None:
        """Map a symptom to a relevant assessment scale query."""
        sl = symptom.lower()
        mappings: dict[str, str] = {
            "depress": "PHQ-9 depression screening",
            "anxiety": "GAD-7 anxiety screening",
            "trauma": "PCL-5 PTSD checklist",
            "insomnia": "ISI insomnia severity index",
            "adhd": "ASRS ADHD screening adult",
            "ocd": "Y-BOCS OCD severity",
            "eating": "EAT-26 eating attitudes",
            "substance": "AUDIT alcohol use disorders",
            "bipolar": "MDQ mood disorder questionnaire",
            "burnout": "MBI Maslach burnout inventory",
            "suicidal": "C-SSRS suicide risk assessment",
            "self-harm": "SHI self-harm inventory",
            "panic": "PDSS panic disorder severity",
            "social": "SIAS social interaction anxiety",
            "ptsd": "PCL-5 PTSD checklist",
            "stress": "PSS perceived stress scale",
            "perfection": "FMPS Frost perfectionism scale",
            "grief": "ICG inventory complicated grief",
            "anger": "STAXI state-trait anger expression",
            "quality of life": "WHOQOL quality of life assessment",
        }
        for key, scale in mappings.items():
            if key in sl:
                return scale
        return None

    def _default_query(self) -> OptimizedQuery:
        return OptimizedQuery(
            query="clinical assessment treatment guidelines",
            category=QueryCategory.DIAGNOSTIC,
            weight=1.0,
            rationale="Fallback query when no case information is available.",
        )
