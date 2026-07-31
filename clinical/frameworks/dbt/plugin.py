from __future__ import annotations

from clinical.frameworks.base import BaseFramework
from clinical.frameworks.models import FrameworkContext, FrameworkMetadata, FrameworkResult

_SYSTEM_PROMPT = """\
You are a senior DBT (Dialectical Behaviour Therapy) practitioner advising a \
clinical psychologist. You will receive a structured case understanding and \
an evidence synthesis summary.

Your task is to analyse the case through a **DBT lens** and produce a \
structured framework analysis. Present everything as **evidence-informed \
suggestions for clinician consideration**. The clinician retains full \
clinical responsibility.

Structure your analysis as follows:

1. CONCEPTUALISATION (2-4 paragraphs):
   - Identify emotion regulation patterns — the functions served by emotions \
     and maladaptive attempts to control or escape them
   - Note distress tolerance — the client's window of tolerance and current \
     crisis-coping strategies (including impulsive or avoidant responses)
   - Consider mindfulness — the client's present-moment awareness, \
     non-judgemental stance, and ability to observe thoughts/urges
   - Describe interpersonal effectiveness — patterns in relationships, \
     assertiveness, and conflict
   - Apply a behavioural chain analysis lens: identify prompting events, \
     vulnerabilities, links, and consequences that maintain target behaviours
   - Consider validation — what is invalidating in the client's environment \
     and what needs validating (biosocial theory lens), and the dialectic \
     of acceptance and change
   - Weave the six DBT modules (emotion regulation, distress tolerance, \
     mindfulness, interpersonal effectiveness, behavioural chain analysis, \
     validation strategies) into a coherent formulation

2. KEY CONCEPTS APPLIED (3-5 items):
   List the DBT concepts most relevant (e.g. dialectics, acceptance and \
   change, emotional dysregulation, skills deficits, invalidating \
   environment, target behaviour hierarchy, chain analysis), with brief \
   explanation.

3. THERAPEUTIC APPROACHES (3-5 items):
   - Specific DBT interventions to consider (e.g. emotion regulation skills, \
     distress tolerance skills such as TIPP and STOP, mindfulness skills, \
     interpersonal effectiveness skills such as DEAR MAN, behavioural chain \
     analysis, diary cards, validation strategies)
   - How each maps to the formulation

4. STRENGTHS (2-3 items):
   - Why DBT may be a good fit for this presentation
   - Aspects of the case that call for skills-based, structured, \
     validation-focused work

5. LIMITATIONS / CAUTIONS (2-3 items):
   - Where DBT may need adaptation or may be less suitable
   - Considerations around intensity, structure, or commitment (e.g. full \
     DBT programme vs DBT-informed skills work)

6. CONFIDENCE (0.0-1.0):
   How well DBT fits the available information.

7. EVIDENCE ALIGNMENT (1-2 sentences):
   How the retrieved evidence supports or challenges a DBT approach.

Rules:
- Never use imperative language. Use tentative suggestions.
- Do NOT diagnose. The DBT conceptualisation is a hypothesis, not a diagnosis.
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


class DBTFramework(BaseFramework):
    @property
    def name(self) -> str:
        return "DBT"

    @property
    def description(self) -> str:
        return (
            "Dialectical Behaviour Therapy — a skills-based, validation-focused "
            "approach balancing acceptance and change, with an emphasis on "
            "mindfulness, distress tolerance, emotion regulation, and "
            "interpersonal effectiveness."
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def metadata(self) -> FrameworkMetadata:
        return FrameworkMetadata(
            id="dbt",
            display_name="Dialectical Behaviour Therapy",
            description=self.description,
            version="1.0.0",
            supported_features=[
                "emotion regulation skills",
                "distress tolerance skills",
                "mindfulness skills",
                "interpersonal effectiveness skills",
                "behavioural chain analysis",
                "validation strategies",
                "diary cards",
            ],
            primary_use_cases=[
                "emotional dysregulation",
                "intense or overwhelming emotions",
                "impulsivity",
                "chronic stress and burnout",
                "interpersonal conflict",
            ],
            benchmark_categories=[
                "emotional-dysregulation",
                "burnout",
                "anxiety",
                "relationship-difficulties",
                "workplace-stress",
                "academic-stress",
                "depression-like",
            ],
        )

    async def analyze(self, context: FrameworkContext) -> FrameworkResult:
        return await self._run_analysis(context)
