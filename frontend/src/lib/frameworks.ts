export type FrameworkCardId =
  | "cbt"
  | "act"
  | "dbt"
  | "mi"
  | "humanistic"
  | "alliance-repair"
  | "general";

export interface FrameworkCardData {
  id: FrameworkCardId;
  name: string;
  tagline: string;
  description: string;
  focus: string;
  whenToUse: string;
  tags: string[];
  color: string;
  confidence: number; // 0-1
  evidenceCount: number;
  featured?: boolean;
}

export const FRAMEWORK_CARDS: FrameworkCardData[] = [
  {
    id: "cbt",
    name: "CBT",
    tagline: "Cognitive Behavioral Therapy",
    description:
      "Structured, problem-focused therapy that targets the thoughts, behaviors, and physiology maintaining a presenting problem.",
    focus: "Thought–behavior change",
    whenToUse:
      "Fits presentations with clear symptom targets and maintaining cycles — rumination, avoidance, panic, insomnia, and low mood.",
    tags: ["Cognitive restructuring", "Behavioral activation", "Exposure"],
    color: "#6366f1",
    confidence: 0.86,
    evidenceCount: 48,
  },
  {
    id: "act",
    name: "ACT",
    tagline: "Acceptance & Commitment Therapy",
    description:
      "Values-directed living built on psychological flexibility — defusion, acceptance, and committed action.",
    focus: "Values & flexibility",
    whenToUse:
      "Strong for chronic pain, experiential avoidance, and presentations where control strategies keep distress going.",
    tags: ["Defusion", "Values work", "Committed action"],
    color: "#0ea5e9",
    confidence: 0.79,
    evidenceCount: 32,
  },
  {
    id: "dbt",
    name: "DBT",
    tagline: "Dialectical Behavior Therapy",
    description:
      "Skills-based treatment balancing acceptance and change across emotion regulation, distress tolerance, and interpersonal effectiveness.",
    focus: "Emotion regulation",
    whenToUse:
      "Fits intense, reactive emotional patterns and repeated crises — emotion dysregulation and interpersonal instability.",
    tags: ["Emotion regulation", "Distress tolerance", "Interpersonal effectiveness"],
    color: "#d946ef",
    confidence: 0.74,
    evidenceCount: 21,
  },
  {
    id: "mi",
    name: "Motivational Interviewing",
    tagline: "MI",
    description:
      "Collaborative, person-centred dialogue that strengthens intrinsic motivation and resolves ambivalence about change.",
    focus: "Ambivalence & motivation",
    whenToUse:
      "Useful when readiness is uncertain — ambivalence, low motivation, or resistance to change across any target.",
    tags: ["OARS", "Change talk", "Rolling with resistance"],
    color: "#f59e0b",
    confidence: 0.81,
    evidenceCount: 27,
  },
  {
    id: "humanistic",
    name: "Humanistic",
    tagline: "Person-Centred & Experiential",
    description:
      "Relational, non-directive exploration that fosters self-acceptance, meaning-making, and depth of contact.",
    focus: "Self-experience & meaning",
    whenToUse:
      "Fits grief, loss, identity, and adjustment themes where a safe relational space is the core mechanism of change.",
    tags: ["Empathic reflection", "Genuineness", "Meaning-making"],
    color: "#10b981",
    confidence: 0.69,
    evidenceCount: 15,
  },
  {
    id: "alliance-repair",
    name: "Alliance Repair",
    tagline: "Rupture & Repair",
    description:
      "Evidence-based strategies to detect and repair ruptures in the therapeutic relationship before they derail treatment.",
    focus: "Therapeutic alliance",
    whenToUse:
      "Deploy across modalities whenever withdrawal, confrontation, or missed sessions signal an alliance rupture.",
    tags: ["Rupture markers", "Repair dialogue", "Meta-communication"],
    color: "#f43f5e",
    confidence: 0.72,
    evidenceCount: 18,
  },
  {
    id: "general",
    name: "General Psychotherapy",
    tagline: "Integrative & Transdiagnostic",
    description:
      "Common-factors integration — alliance, expectation, and structure — applicable across presentations and modalities.",
    focus: "Common factors & integration",
    whenToUse:
      "A flexible foundation when no single manualized frame clearly leads, or as the integrative backbone of a case.",
    tags: ["Common factors", "Psychoeducation", "Shared formulation"],
    color: "#64748b",
    confidence: 0.64,
    evidenceCount: 40,
    featured: true,
  },
];

export const FRAMEWORK_ID_TO_KB: Partial<Record<FrameworkCardId, string>> = {
  cbt: "cbt",
  act: "act",
  humanistic: "humanistic",
};
