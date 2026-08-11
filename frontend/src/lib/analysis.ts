import type { Framework } from "../api/client";
import type { CaseRecord } from "./types";

export interface FrameworkAssessment {
  framework: Framework;
  score: number; // suitability 0-100
  verdict: "recommended" | "consider" | "not-indicated";
  matched: string[];
  reasons: string[];
  interventions: string[];
}

interface AspectRule {
  pattern: string;
  label: string;
  interventions?: string[];
}

interface FrameworkConfig {
  label: string;
  color: string;
  rules: AspectRule[];
  defaults: string[];
  fitReason: string;
  notFitReason: string;
  caution?: string;
}

const CONFIG: Record<Framework, FrameworkConfig> = {
  cbt: {
    label: "CBT",
    color: "#6366f1",
    rules: [
      { pattern: "rumination", label: "rumination", interventions: ["Cognitive restructuring", "Worry-time scheduling"] },
      { pattern: "worry", label: "excessive worry", interventions: ["Cognitive restructuring", "Worry-time scheduling"] },
      { pattern: "catastrophi", label: "catastrophic thinking", interventions: ["Cognitive restructuring", "Decatastrophizing"] },
      { pattern: "insomni", label: "sleep disruption", interventions: ["Stimulus control for sleep"] },
      { pattern: "panic", label: "panic", interventions: ["Interoceptive exposure", "CBT psychoeducation"] },
      { pattern: "anxiet", label: "anxiety", interventions: ["Cognitive restructuring"] },
      { pattern: "avoid", label: "avoidance", interventions: ["Exposure hierarchy"] },
      { pattern: "anhedoni", label: "anhedonia", interventions: ["Behavioral activation"] },
      { pattern: "depress", label: "low mood", interventions: ["Behavioral activation"] },
      { pattern: "motivat", label: "low motivation", interventions: ["Behavioral activation"] },
      { pattern: "muscle tension", label: "muscle tension", interventions: ["Progressive muscle relaxation"] },
    ],
    defaults: ["Cognitive restructuring", "Structured psychoeducation", "Behavioral experiments"],
    fitReason:
      "Symptoms and maintaining factors map onto the cognitive-behavioral model — thoughts, avoidance, and physiology each give a clear treatment target.",
    notFitReason:
      "Few of the documented features map directly onto the cognitive-behavioral model; consider whether a structured symptom model fits the presentation.",
    caution: "Recheck the case for trauma or value-conflict themes that a purely CBT frame could under-serve.",
  },
  act: {
    label: "ACT",
    color: "#0ea5e9",
    rules: [
      { pattern: "pain", label: "chronic pain", interventions: ["Values clarification", "Graded activity / committed action"] },
      { pattern: "catastrophi", label: "pain catastrophizing", interventions: ["Cognitive defusion", "Acceptance-based techniques"] },
      { pattern: "avoid", label: "experiential avoidance", interventions: ["Cognitive defusion", "Graded activity / committed action"] },
      { pattern: "control", label: "control struggle", interventions: ["Acceptance-based techniques", "Values clarification"] },
      { pattern: "withdraw", label: "social withdrawal", interventions: ["Committed action", "Values clarification"] },
      { pattern: "isolat", label: "social isolation", interventions: ["Committed action", "Values clarification"] },
      { pattern: "anxiet", label: "anxiety", interventions: ["Defusion", "Willingness / acceptance"] },
      { pattern: "distress", label: "distress", interventions: ["Distress tolerance", "Willingness / acceptance"] },
      { pattern: "pain-related", label: "pain-related distress", interventions: ["Values clarification", "Cognitive defusion"] },
    ],
    defaults: ["Values clarification", "Cognitive defusion", "Acceptance-based techniques"],
    fitReason:
      "ACT suits presentations where control-and-avoid strategies keep distress going — chronic pain, fusion with thoughts, or values-disconnected activity.",
    notFitReason:
      "The case shows limited experiential-avoidance or values-conflict themes that ACT is best placed to target.",
    caution: "Ensure CBT-style symptom targets are still monitored if ACT becomes the primary frame.",
  },
  humanistic: {
    label: "Humanistic",
    color: "#10b981",
    rules: [
      { pattern: "grief", label: "grief", interventions: ["Grief narrative work", "Empathic reflection"] },
      { pattern: "loss", label: "loss", interventions: ["Meaning-making dialogue", "Empathic reflection"] },
      { pattern: "bereav", label: "bereavement", interventions: ["Grief narrative work"] },
      { pattern: "meaning", label: "meaning-making", interventions: ["Meaning-making dialogue"] },
      { pattern: "withdraw", label: "social withdrawal", interventions: ["Person-centered exploration"] },
      { pattern: "isolat", label: "social isolation", interventions: ["Person-centered exploration"] },
      { pattern: "loneli", label: "loneliness", interventions: ["Person-centered exploration"] },
      { pattern: "adjust", label: "adjustment", interventions: ["Person-centered exploration", "Empathic reflection"] },
      { pattern: "identity", label: "identity", interventions: ["Experiential self-exploration"] },
      { pattern: "self-esteem", label: "self-worth", interventions: ["Unconditional positive regard"] },
    ],
    defaults: ["Person-centered exploration", "Empathic reflection", "Genuineness"],
    fitReason:
      "Humanistic work fits presentations centred on grief, loss, adjustment, or self-experience where a safe relational space is the core mechanism.",
    notFitReason:
      "The case is dominated by symptom-specific targets; humanistic work could still support the alliance but may not be the primary lever.",
    caution: "Pair with structured outcome monitoring so progress remains measurable.",
  },
};

function normalize(record: CaseRecord): string {
  return [
    record.presentingConcern,
    ...record.concerns,
    ...record.symptoms,
    ...record.stressors,
  ]
    .join(" ")
    .toLowerCase();
}

export function analyzeFrameworks(
  record: CaseRecord,
): Record<Framework, FrameworkAssessment> {
  const text = normalize(record);

  const result = {} as Record<Framework, FrameworkAssessment>;
  (Object.keys(CONFIG) as Framework[]).forEach((framework) => {
    const cfg = CONFIG[framework];
    const matched = cfg.rules.filter((r) => text.includes(r.pattern));

    let score = 35 + Math.min(3, matched.length) * 16;
    if (record.framework === framework) score += 7;
    score = Math.min(92, Math.round(score));

    const verdict: FrameworkAssessment["verdict"] =
      score >= 70 ? "recommended" : score >= 45 ? "consider" : "not-indicated";

    const interventions = [
      ...new Set(matched.flatMap((m) => m.interventions ?? [])),
      ...cfg.defaults,
    ].slice(0, 3);

    const reasons: string[] = [];
    if (matched.length > 0) {
      reasons.push(`Case features: ${matched.map((m) => m.label).join(", ")}.`);
    }
    reasons.push(matched.length > 0 ? cfg.fitReason : cfg.notFitReason);
    if (record.framework === framework) {
      reasons.push("Currently the active framework for this case.");
    }
    if (cfg.caution && matched.length > 0) {
      reasons.push(cfg.caution);
    }

    result[framework] = {
      framework,
      score,
      verdict,
      matched: matched.map((m) => m.label),
      reasons,
      interventions,
    };
  });

  return result;
}

export function bestFit(
  assessments: Record<Framework, FrameworkAssessment>,
): FrameworkAssessment {
  return (Object.values(assessments) as FrameworkAssessment[]).reduce((a, b) =>
    b.score > a.score ? b : a,
  );
}
