"""
scripts/bootstrap_prompts.py — Generate framework prompt template YAML files.

Run:  python scripts/bootstrap_prompts.py

Creates ``prompts/`` directories under each framework plugin and
writes framework-specific Jinja2 template files for all 8 lifecycle
methods.  Existing files are not overwritten.

Uses ``string.Template`` for bootstrapping so that ``{{ ... }}``
Jinja2 delimiters pass through to the YAML output unmodified.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

BASE = Path(__file__).resolve().parent.parent / "clinical" / "framework"

# ── Framework-specific terminology & concepts ─────────────────────────────

FRAMEWORK_PROFILES: dict[str, dict] = {
    "cbt": {
        "name": "Cognitive Behavioral Therapy",
        "concepts": {
            "analyze": "Identify automatic thoughts, cognitive distortions, core beliefs, and behavioural patterns maintaining the problem",
            "evidence": "CBT outcome research, cognitive models of specific disorders, behavioural activation protocols",
            "conceptualize": "Formulate using the cognitive model: activating events -> beliefs -> consequences; identify maintaining cycles",
            "focus": "Cognitive restructuring of distorted thoughts, behavioural activation, exposure to feared stimuli",
            "interventions": "Thought records, behavioural experiments, exposure hierarchies, activity scheduling, Socratic questioning",
            "objectives": "Symptom reduction, skill acquisition, relapse prevention through cognitive and behavioural change",
            "followup": "Symptom monitoring, homework compliance, skill generalisation, cognitive rehearsal",
            "limitations": "May be less suitable when cognitive capacity is limited, or when problems are primarily systemic or developmental",
        },
    },
    "act": {
        "name": "Acceptance and Commitment Therapy",
        "concepts": {
            "analyze": "Identify experiential avoidance, cognitive fusion, loss of contact with values, and rigid behavioural patterns",
            "evidence": "ACT outcome studies for chronic pain, anxiety, depression; process-of-change research on psychological flexibility",
            "conceptualize": "Formulate using the hexaflex model: identifying fusion-avoidance patterns and disconnection from values",
            "focus": "Creative hopelessness, acceptance, defusion, self-as-context, values clarification, committed action",
            "interventions": "Metaphors, experiential exercises, mindfulness, values card sort, committed action plans",
            "objectives": "Increase psychological flexibility, reduce experiential avoidance, clarify and enact values",
            "followup": "Values consistency check, willingness ratings, defusion practice, committed action review",
            "limitations": "Less structured than CBT; may be challenging for clients seeking concrete symptom relief or with limited insight",
        },
    },
    "dbt": {
        "name": "Dialectical Behavior Therapy",
        "concepts": {
            "analyze": "Assess emotion dysregulation patterns, invalidation history, behavioural dyscontrol, and skills deficits across four modules",
            "evidence": "DBT RCTs for borderline personality disorder, emotion dysregulation, self-harm, and eating disorders",
            "conceptualize": "Formulate using biosocial theory: biological vulnerability + invalidating environment -> emotion dysregulation -> behavioural patterns",
            "focus": "Stage-based treatment: behavioural control (Stage 1) -> emotional experiencing (Stage 2) -> ordinary happiness (Stage 3)",
            "interventions": "Skills training (mindfulness, distress tolerance, emotion regulation, interpersonal effectiveness), diary cards, coaching calls",
            "objectives": "Reduce life-threatening behaviours, increase emotion regulation skills, improve interpersonal functioning",
            "followup": "Diary card review, skills practice tracking, chain analysis of problem behaviours",
            "limitations": "High therapist and client time commitment; requires specialised training; may not suit less severe presentations",
        },
    },
    "mi": {
        "name": "Motivational Interviewing",
        "concepts": {
            "analyze": "Identify stage of change, level of ambivalence, change talk vs sustain talk, and intrinsic motivation",
            "evidence": "MI RCTs for substance use, health behaviour change, treatment engagement; process research on change talk",
            "conceptualize": "Formulate using the transtheoretical model and decisional balance; identify discrepancies between values and behaviour",
            "focus": "Evoking change talk, softening sustain talk, rolling with resistance, supporting self-efficacy",
            "interventions": "OARS (open questions, affirmations, reflections, summaries), decisional balance, importance/confidence rulers, agenda mapping",
            "objectives": "Resolve ambivalence, strengthen commitment language, increase intrinsic motivation for change",
            "followup": "Change talk monitoring, commitment strength assessment, action planning, follow-through support",
            "limitations": "Primarily a brief intervention; less suited for severe mental illness as a standalone treatment; requires fidelity to MI spirit",
        },
    },
    "humanistic": {
        "name": "Person-Centred / Humanistic Therapy",
        "concepts": {
            "analyze": "Assess incongruence between self-concept and experience, conditions of worth, and blocks to self-actualisation",
            "evidence": "Rogers' necessary and sufficient conditions research; common factors and therapeutic alliance literature",
            "conceptualize": "Formulate using the self-theory model: conditions of worth -> incongruence -> defensiveness -> disorganisation",
            "focus": "Therapeutic relationship as the primary vehicle for change; unconditional positive regard, empathy, genuineness",
            "interventions": "Reflection, empathic understanding, congruence, unconditional positive regard, focusing, experiential processing",
            "objectives": "Reduce incongruence, strengthen self-regard, increase organismic trusting, enhance meaning-making",
            "followup": "Experiential exploration, meaning-making, self-regard shifts, congruence tracking",
            "limitations": "Less structured than directive approaches; may be insufficient for severe symptoms without additional interventions",
        },
    },
    "alliance_rupture": {
        "name": "Alliance-Focused / Rupture Repair",
        "concepts": {
            "analyze": "Identify rupture markers (withdrawal vs confrontation), attachment patterns, and metacommunication readiness",
            "evidence": "Safran & Muran alliance rupture research; meta-analyses linking rupture repair to improved outcomes",
            "conceptualize": "Formulate rupture cycle: trigger -> rupture marker -> therapist response -> client response -> resolution or impasse",
            "focus": "Metacommunication about the therapeutic relationship, repairing alliance strains, addressing interpersonal patterns",
            "interventions": "Metacommunication, immediacy, self-disclosure, collaborative empiricism about the relationship",
            "objectives": "Repair alliance ruptures, improve metacommunication skills, generalise relational learning to outside relationships",
            "followup": "Alliance monitoring, rupture marker tracking, metacommunication practice, relational pattern identification",
            "limitations": "Requires therapist comfort with immediacy; may not be the primary intervention when alliance is strong",
        },
    },
    "general": {
        "name": "General Psychotherapy",
        "concepts": {
            "analyze": "Assess presenting problems, risk factors, protective factors, treatment history, and client preferences using a biopsychosocial lens",
            "evidence": "Common factors research, practice guidelines, meta-analyses of psychotherapy outcomes",
            "conceptualize": "Integrative formulation drawing on biological, psychological, social, and cultural factors relevant to the presentation",
            "focus": "Building alliance, establishing goals, psychoeducation, evidence-informed intervention matching client preferences",
            "interventions": "Active listening, psychoeducation, goal setting, progress monitoring, relapse prevention, collaborative treatment planning",
            "objectives": "Establish therapeutic alliance, reduce symptoms, improve functioning, enhance coping, prevent relapse",
            "followup": "Progress monitoring, outcome assessment, therapy review, maintenance planning, termination preparation",
            "limitations": "May lack the specificity of specialised frameworks for particular disorders; integrates flexibly but with less theoretical precision",
        },
    },
}

# ── Template YAML for each lifecycle method ───────────────────────────────
# Uses $framework_name and $concept_<method> as string.Template placeholders.
# "{{ ... }}" passes through to the YAML output for runtime Jinja2 rendering.

TEMPLATES: dict[str, str] = {
    "analyze_case": """\
version: "1.0.0"
description: "$framework_name — analyse a clinical case through the framework lens"
variables:
  - name: system_header
    description: "System prompt header identifying the framework"
    required: true
  - name: case_text
    description: "Formatted case understanding, formulation, and evidence"
    required: true
  - name: output_schema
    description: "JSON schema for the CaseAnalysis model"
    required: true
templates:
  en: |
    {{ system_header }}

    As a $framework_name practitioner, analyse this case.

    $concept_analyze

    {{ case_text }}

    Respond with valid JSON matching this schema exactly:
    {{ output_schema }}

    Return ONLY the JSON object. No markdown fences.
    No explanatory text before or after.
""",
    "retrieve_evidence": """\
version: "1.0.0"
description: "$framework_name — retrieve and synthesise relevant evidence"
variables:
  - name: system_header
    description: "System prompt header identifying the framework"
    required: true
  - name: case_text
    description: "Formatted case understanding, formulation, and evidence"
    required: true
  - name: output_schema
    description: "JSON schema for the EvidenceResult model"
    required: true
templates:
  en: |
    {{ system_header }}

    As a $framework_name practitioner, retrieve relevant evidence.

    $concept_evidence

    {{ case_text }}

    Identify key findings, synthesise the evidence base, note gaps,
    and rate your confidence. If no literature collection is available,
    suggest the type of evidence that would be needed.

    Respond with valid JSON matching this schema exactly:
    {{ output_schema }}

    Return ONLY the JSON object. No markdown fences.
    No explanatory text before or after.
""",
    "case_conceptualization": """\
version: "1.0.0"
description: "$framework_name — generate a case conceptualisation"
variables:
  - name: system_header
    description: "System prompt header identifying the framework"
    required: true
  - name: case_text
    description: "Formatted case understanding, formulation, and evidence"
    required: true
  - name: output_schema
    description: "JSON schema for the CaseConceptualization model"
    required: true
templates:
  en: |
    {{ system_header }}

    As a $framework_name practitioner, generate a case conceptualisation.

    $concept_conceptualize

    {{ case_text }}

    Describe the framework formulation, maintaining factors, core mechanisms,
    developmental factors, and provide a summary.

    Respond with valid JSON matching this schema exactly:
    {{ output_schema }}

    Return ONLY the JSON object. No markdown fences.
    No explanatory text before or after.
""",
    "therapeutic_focus": """\
version: "1.0.0"
description: "$framework_name — identify therapeutic focus areas"
variables:
  - name: system_header
    description: "System prompt header identifying the framework"
    required: true
  - name: case_text
    description: "Formatted case understanding, formulation, and evidence"
    required: true
  - name: output_schema
    description: "JSON schema for the TherapeuticFocusResult model"
    required: true
templates:
  en: |
    {{ system_header }}

    As a $framework_name practitioner, identify therapeutic focus areas.

    $concept_focus

    {{ case_text }}

    Specify primary and secondary focus areas, staging/sequencing,
    and the clinical rationale.

    Respond with valid JSON matching this schema exactly:
    {{ output_schema }}

    Return ONLY the JSON object. No markdown fences.
    No explanatory text before or after.
""",
    "intervention_plan": """\
version: "1.0.0"
description: "$framework_name — generate an intervention plan"
variables:
  - name: system_header
    description: "System prompt header identifying the framework"
    required: true
  - name: case_text
    description: "Formatted case understanding, formulation, and evidence"
    required: true
  - name: output_schema
    description: "JSON schema for the InterventionPlanResult model"
    required: true
templates:
  en: |
    {{ system_header }}

    As a $framework_name practitioner, generate an intervention plan.

    $concept_interventions

    {{ case_text }}

    List specific interventions with technique and rationale.
    Describe the provisional strategy, estimated duration, and expected outcomes.

    Respond with valid JSON matching this schema exactly:
    {{ output_schema }}

    Return ONLY the JSON object. No markdown fences.
    No explanatory text before or after.
""",
    "session_objectives": """\
version: "1.0.0"
description: "$framework_name — define session objectives and goals"
variables:
  - name: system_header
    description: "System prompt header identifying the framework"
    required: true
  - name: case_text
    description: "Formatted case understanding, formulation, and evidence"
    required: true
  - name: output_schema
    description: "JSON schema for the SessionObjectivesResult model"
    required: true
templates:
  en: |
    {{ system_header }}

    As a $framework_name practitioner, define session objectives.

    $concept_objectives

    {{ case_text }}

    Specify immediate objectives, short-term goals, long-term goals,
    and observable progress indicators.

    Respond with valid JSON matching this schema exactly:
    {{ output_schema }}

    Return ONLY the JSON object. No markdown fences.
    No explanatory text before or after.
""",
    "followup_questions": """\
version: "1.0.0"
description: "$framework_name — suggest follow-up questions"
variables:
  - name: system_header
    description: "System prompt header identifying the framework"
    required: true
  - name: case_text
    description: "Formatted case understanding, formulation, and evidence"
    required: true
  - name: output_schema
    description: "JSON schema for the FollowupQuestionsResult model"
    required: true
templates:
  en: |
    {{ system_header }}

    As a $framework_name practitioner, suggest follow-up questions.

    $concept_followup

    {{ case_text }}

    Include assessment questions, therapeutic probes, outcome monitoring
    questions, and engagement questions.

    Respond with valid JSON matching this schema exactly:
    {{ output_schema }}

    Return ONLY the JSON object. No markdown fences.
    No explanatory text before or after.
""",
    "limitations": """\
version: "1.0.0"
description: "$framework_name — describe limitations and contraindications"
variables:
  - name: system_header
    description: "System prompt header identifying the framework"
    required: true
  - name: case_text
    description: "Formatted case understanding, formulation, and evidence"
    required: true
  - name: output_schema
    description: "JSON schema for the LimitationsResult model"
    required: true
templates:
  en: |
    {{ system_header }}

    As a $framework_name expert, describe limitations.

    $concept_limitations

    {{ case_text }}

    List contraindications, precautions, alternative frameworks that
    may be more appropriate, and general caveats.

    Respond with valid JSON matching this schema exactly:
    {{ output_schema }}

    Return ONLY the JSON object. No markdown fences.
    No explanatory text before or after.
""",
}

LIFECYCLE_METHODS = [
    "analyze_case",
    "retrieve_evidence",
    "case_conceptualization",
    "therapeutic_focus",
    "intervention_plan",
    "session_objectives",
    "followup_questions",
    "limitations",
]

CONCEPT_KEY_MAP: dict[str, str] = {
    "analyze_case": "analyze",
    "retrieve_evidence": "evidence",
    "case_conceptualization": "conceptualize",
    "therapeutic_focus": "focus",
    "intervention_plan": "interventions",
    "session_objectives": "objectives",
    "followup_questions": "followup",
    "limitations": "limitations",
}


def _generate_templates() -> None:
    """Generate all template YAML files for all frameworks."""
    for slug, profile in FRAMEWORK_PROFILES.items():
        prompts_dir = BASE / slug / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        skipped = 0

        for method in LIFECYCLE_METHODS:
            dest = prompts_dir / f"{method}.yaml"
            if dest.exists():
                skipped += 1
                continue

            raw = TEMPLATES[method]
            concept_key = CONCEPT_KEY_MAP[method]
            mapping = {
                "framework_name": profile["name"],
                f"concept_{concept_key}": profile["concepts"][concept_key],
            }
            rendered = Template(raw).safe_substitute(mapping)
            dest.write_text(rendered, encoding="utf-8")
            generated += 1

        print(
            f"  [{slug}] {prompts_dir.relative_to(BASE)}: "
            f"{generated} generated, {skipped} skipped"
        )


def main() -> None:
    print("Generating framework prompt templates...")
    _generate_templates()
    print("Done.")


if __name__ == "__main__":
    main()
