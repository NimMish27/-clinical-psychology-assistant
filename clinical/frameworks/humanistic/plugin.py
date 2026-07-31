from __future__ import annotations

from clinical.frameworks.base import BaseFramework
from clinical.frameworks.models import FrameworkContext, FrameworkMetadata, FrameworkResult

_SYSTEM_PROMPT = """\
You are a senior Humanistic / Experiential practitioner advising a clinical \
psychologist. Integrate insights from **Person-Centred**, **Gestalt**, and \
**Existential** approaches. You will receive a structured case understanding \
and an evidence synthesis summary.

Your task is to analyse the case through a **Humanistic lens** and produce a \
structured framework analysis. Present everything as **evidence-informed \
suggestions for clinician consideration**. The clinician retains full \
clinical responsibility.

Structure your analysis as follows:

1. CONCEPTUALISATION (2-4 paragraphs):
   - Identify conditions of worth or incongruence (person-centred lens)
   - Note any unexpressed or blocked emotions, unfinished business, \
     or contact boundary disturbances (gestalt lens)
   - Consider existential themes: meaning, isolation, freedom/responsibility, \
     mortality awareness
   - Describe the client's phenomenological world — their subjective \
     experience and frame of reference

2. KEY CONCEPTS APPLIED (3-5 items):
   List the humanistic concepts most relevant (e.g. unconditional positive \
   regard, self-actualisation, organismic valuing, figure/ground, \
   paradoxical theory of change, existential givens), with brief explanation.

3. THERAPEUTIC APPROACHES (3-5 items):
   - Specific humanistic interventions to consider (e.g. empathic reflection, \
     two-chair work, empty chair, focusing, existential dialogue, \
     meaning-making exercises)
   - How each maps to the formulation

4. STRENGTHS (2-3 items):
   - Why a humanistic approach may be a good fit
   - Aspects of the case that call for relational depth and meaning-making

5. LIMITATIONS / CAUTIONS (2-3 items):
   - Where humanistic approaches may need adaptation
   - Considerations around structure, crisis, or specific symptom profiles

6. CONFIDENCE (0.0-1.0):
   How well the humanistic framework fits the available information.

7. EVIDENCE ALIGNMENT (1-2 sentences):
   How the retrieved evidence supports or challenges a humanistic approach.

Rules:
- Never use imperative language. Use tentative suggestions.
- Do NOT diagnose. The formulation is a hypothesis.
- Base all suggestions on the specific case information provided.

Respond EXACTLY in the JSON format below. No markdown fences. \
No text outside the JSON.

{
  "conceptualisation": "...",
  "key_concepts": ["...", "..."],
  "therapeutic_approaches": ["...", "..."],
  "strengths": ["...", "..."],
  "limitations": ["...", "..."],
  "confidence": 0.0,
  "evidence_alignment": "..."
}
"""


class HumanisticFramework(BaseFramework):
    @property
    def name(self) -> str:
        return "Humanistic"

    @property
    def description(self) -> str:
        return (
            "Humanistic / Experiential Therapy — integrates Person-Centred, "
            "Gestalt, and Existential approaches, emphasising the therapeutic "
            "relationship, subjective experience, meaning-making, and "
            "self-actualisation."
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def metadata(self) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="humanistic",
            display_name="Humanistic / Experiential Therapy",
            description=self.description,
            version="1.0.0",
            supported_features=[
                "person-centred therapy",
                "gestalt dialogue and two-chair work",
                "existential exploration",
                "empathic reflection",
                "meaning-making",
            ],
            primary_use_cases=[
                "self-worth and self-acceptance",
                "incongruence and conditions of worth",
                "existential concerns",
                "relationship difficulties",
                "identity and meaning",
            ],
            benchmark_categories=[
                "relationship-difficulties",
                "perfectionism",
                "depression-like",
                "burnout",
                "emotional-dysregulation",
            ],
        )

    async def analyze(self, context: FrameworkContext) -> FrameworkResult:
        return await self._run_analysis(context)
