import type { Framework } from "../api/client";

export type FrameworkCardId =
  | "cbt"
  | "act"
  | "dbt"
  | "mi"
  | "humanistic"
  | "alliance-repair"
  | "general";

export interface InterventionItem {
  name: string;
  desc: string;
}

export interface SessionPhase {
  phase: string;
  weeks: string;
  focus: string;
  items: string[];
}

export interface HomeworkItem {
  name: string;
  cadence: string;
  desc: string;
}

export interface FrameworkWorkspaceData {
  conceptualization: string;
  model: { label: string; points: string[] };
  therapeuticFocus: string;
  strengths: string[];
  limitations: string[];
  assessmentQuestions: string[];
  suggestedInterventions: InterventionItem[];
  sessionPlan: SessionPhase[];
  homework: HomeworkItem[];
}

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
  workspace: FrameworkWorkspaceData;
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
    workspace: {
      conceptualization:
        "CBT conceptualizes the presenting problem as maintained by a cycle of thoughts, emotions, behaviors, and physiological reactions. For this client, the pattern centers on situational triggers, automatic interpretations, and the avoidance and safety behaviors that keep the cycle running. The formulation identifies core beliefs, rules, and conditional assumptions that drive moment-to-moment cognition, and connects each to a concrete treatment target.",
      model: {
        label: "Cognitive-behavioral model",
        points: [
          "Situation → cognition → emotion → behavior cycle",
          "Core beliefs and assumptions shape automatic thoughts",
          "Avoidance and safety behaviors maintain anxiety and low mood",
        ],
      },
      therapeuticFocus:
        "Target the thoughts, behaviors, and physiological responses maintaining the symptom cycle through active, structured, and time-limited techniques.",
      strengths: [
        "Large, well-replicated evidence base across anxiety, depression, and insomnia",
        "Highly structured — clear agenda, session content, and progress tracking",
        "Teaches durable self-management skills the client can keep using",
        "Strong fit for manualized protocols and outcome measurement",
      ],
      limitations: [
        "Relies on client insight and consistent between-session engagement",
        "Can under-serve complex relational, grief, or identity themes",
        "Effectiveness depends on homework completion",
      ],
      assessmentQuestions: [
        "What goes through your mind right before the worry starts?",
        "What do you do to manage the feeling in the moment?",
        "When did this pattern begin, and what was happening then?",
        "How strongly do you believe that thought right now, on a scale of 0–100?",
      ],
      suggestedInterventions: [
        {
          name: "Cognitive restructuring",
          desc: "Socratic questioning and thought records to test and reframe unhelpful beliefs.",
        },
        {
          name: "Behavioral experiments",
          desc: "Real-world tests of predictions to disconfirm catastrophic beliefs.",
        },
        {
          name: "Exposure hierarchy",
          desc: "Graduated, repeated contact with feared stimuli to reduce avoidance.",
        },
      ],
      sessionPlan: [
        {
          phase: "Engagement & assessment",
          weeks: "Sessions 1–2",
          focus: "Build the formulation and set the agenda",
          items: ["Psychoeducation on the CBT cycle", "Co-develop a formulation diagram", "Introduce the thought record"],
        },
        {
          phase: "Skill building",
          weeks: "Sessions 3–6",
          focus: "Core cognitive and behavioral skills",
          items: ["Cognitive restructuring practice", "Behavioral experiments", "Relaxation / stimulus control"],
        },
        {
          phase: "Consolidation & relapse prevention",
          weeks: "Sessions 7–10",
          focus: "Generalize skills and prevent relapse",
          items: ["Review progress against goals", "Relapse prevention blueprint", "Fade session frequency"],
        },
      ],
      homework: [
        {
          name: "Thought record",
          cadence: "Daily",
          desc: "Capture 2–3 situations, automatic thoughts, and an alternative balanced thought.",
        },
        {
          name: "Behavioral experiment",
          cadence: "Weekly",
          desc: "Run one planned experiment and log prediction vs. outcome.",
        },
        {
          name: "Sleep-wake schedule",
          cadence: "Daily",
          desc: "Keep consistent bed and rise times, logging sleep onset.",
        },
      ],
    },
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
    workspace: {
      conceptualization:
        "ACT frames the client's struggle as psychological inflexibility — fusion with difficult thoughts and experiential avoidance that steadily narrow a valued life. Distress is a normal part of the human condition; the effort to control or eliminate it is what creates the loop. The formulation maps how cognitive fusion, avoidance, and loss of contact with values maintain suffering, and where flexibility can be restored.",
      model: {
        label: "The hexaflex",
        points: [
          "Six core processes of psychological flexibility",
          "Fusion + avoidance = rigidity and suffering",
          "Values provide the compass for committed action",
        ],
      },
      therapeuticFocus:
        "Increase psychological flexibility so the client can act in line with values even in the presence of difficult thoughts and feelings.",
      strengths: [
        "Strong evidence for chronic pain, distress, and values-disconnected presentations",
        "Values-driven and highly individualized",
        "Process-based and transdiagnostic",
        "Compatible with mindfulness-based practice",
      ],
      limitations: [
        "Less structured than manualized CBT — progress can be harder to track",
        "Metaphorical and experiential work requires clinician fluency",
        "Weaker fit for acute crisis stabilization",
      ],
      assessmentQuestions: [
        "What has the struggle with this feeling cost you over the past year?",
        "What would you do differently if this feeling were smaller?",
        "What does a good day look like in terms of what matters to you?",
        "Where does avoidance of discomfort show up in your week?",
      ],
      suggestedInterventions: [
        {
          name: "Cognitive defusion",
          desc: "Observe thoughts as passing events rather than literal truths.",
        },
        {
          name: "Values clarification",
          desc: "Identify and rank core life values to guide committed action.",
        },
        {
          name: "Committed action",
          desc: "Set concrete, values-based behavioral goals across life domains.",
        },
      ],
      sessionPlan: [
        {
          phase: "Open & orient",
          weeks: "Sessions 1–2",
          focus: "Assess inflexibility and introduce the model",
          items: ["Creative hopelessness dialogue", "Introduce the hexaflex", "Values card sort"],
        },
        {
          phase: "Core processes",
          weeks: "Sessions 3–6",
          focus: "Defusion, acceptance, and present-moment awareness",
          items: ["Leaves-on-a-stream defusion", "Willingness exercises", "Mindfulness practice"],
        },
        {
          phase: "Values & action",
          weeks: "Sessions 7–10",
          focus: "Committed action and maintenance",
          items: ["Values-based action plans", "Review committed actions", "Flexible responding and relapse prevention"],
        },
      ],
      homework: [
        {
          name: "Values card sort",
          cadence: "One-off",
          desc: "Rank and select the five most important life values.",
        },
        {
          name: "Willingness practice",
          cadence: "Daily",
          desc: "Notice a difficult thought and practice a defusion skill.",
        },
        {
          name: "Committed action step",
          cadence: "Weekly",
          desc: "One concrete action toward a stated value, logged.",
        },
      ],
    },
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
    workspace: {
      conceptualization:
        "DBT frames the client's problems as arising from emotion dysregulation interacting with an invalidating environment. Dialectics — the balance of acceptance and change — underpin every intervention. The formulation traces each crisis through triggers, vulnerability factors, emotional escalation, and reactive behaviors, then targets every link with a specific skill module.",
      model: {
        label: "Biosocial model",
        points: [
          "Emotion vulnerability × invalidating environment",
          "Dialectics: acceptance + change at every turn",
          "Four skill modules: mindfulness, distress tolerance, emotion regulation, interpersonal effectiveness",
        ],
      },
      therapeuticFocus:
        "Build a life worth living by replacing maladaptive coping with a structured skills toolkit across four modules.",
      strengths: [
        "Strong evidence for emotion dysregulation and crisis-prone presentations",
        "Skills-based, structured, and measurable",
        "Includes coaching, diary cards, and consultation team structures",
        "Explicitly balances acceptance and change",
      ],
      limitations: [
        "Resource-intensive — skills group, phone coaching, and team consultation",
        "Typically a longer commitment",
        "May be more than needed for mild, low-acuity presentations",
      ],
      assessmentQuestions: [
        "What happens in your body and mind in the hour before a crisis?",
        "Which skills have you tried, and which helped briefly?",
        "On a scale of 0–10, how intense was the urge to act on that feeling?",
        "Who is part of your safety and support plan right now?",
      ],
      suggestedInterventions: [
        {
          name: "Distress tolerance skills",
          desc: "Crisis survival strategies such as TIPP, self-soothing, and pros and cons.",
        },
        {
          name: "Emotion regulation",
          desc: "Identify emotions, reduce vulnerability, and build positive experiences.",
        },
        {
          name: "Interpersonal effectiveness",
          desc: "DEAR MAN, GIVE, and FAST for balanced requests and boundaries.",
        },
      ],
      sessionPlan: [
        {
          phase: "Orientation & commitment",
          weeks: "Sessions 1–4",
          focus: "Assessment, agreement, and diary cards",
          items: ["Orient to DBT structure", "Agree targets and hierarchy", "Introduce diary cards"],
        },
        {
          phase: "Skills modules",
          weeks: "Sessions 5–16",
          focus: "Cycle through the four skill modules",
          items: ["Mindfulness", "Distress tolerance", "Emotion regulation", "Interpersonal effectiveness"],
        },
        {
          phase: "Consolidation & generalization",
          weeks: "Sessions 17–24",
          focus: "Apply skills to real crises and strengthen support",
          items: ["Coaching generalization", "Relapse prevention", "Graduation planning"],
        },
      ],
      homework: [
        {
          name: "Diary card",
          cadence: "Daily",
          desc: "Log urges, behaviors, and which skills were used each day.",
        },
        {
          name: "Skill practice",
          cadence: "Daily",
          desc: "Practice one skill per day and note its effect.",
        },
        {
          name: "Mindfulness check-in",
          cadence: "Daily",
          desc: "A five-minute mindfulness practice with a brief log.",
        },
      ],
    },
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
    workspace: {
      conceptualization:
        "MI conceptualizes behavior change as emerging from the resolution of ambivalence. Rather than directing the client, the clinician draws out the client's own reasons for change. The formulation tracks readiness, importance, confidence, and the discrepancy between current behavior and the client's values and goals, and locates change talk as the engine of progress.",
      model: {
        label: "Spirit of MI",
        points: [
          "Ambivalence is a target, not an obstacle",
          "Change talk predicts behavior change",
          "Spirit: partnership, acceptance, compassion, evocation",
        ],
      },
      therapeuticFocus:
        "Strengthen intrinsic motivation and commitment by evoking and amplifying the client's own change talk.",
      strengths: [
        "Highly effective for engagement, readiness, and retention",
        "Brief and easily integrated with other approaches",
        "Excellent for low-motivation or coerced contexts",
        "Strong relational and non-judgmental stance",
      ],
      limitations: [
        "Usually insufficient as a standalone for severe symptoms",
        "Requires discipline to avoid the expert trap",
        "Progress can be difficult to quantify",
      ],
      assessmentQuestions: [
        "On a scale of 0–10, how important is change right now? Why not lower?",
        "What would you miss about the way things are now?",
        "What would be good about changing — for you and for others?",
        "If nothing changed in five years, how would that feel?",
      ],
      suggestedInterventions: [
        {
          name: "OARS",
          desc: "Open questions, affirmations, reflections, and summaries to deepen dialogue.",
        },
        {
          name: "Decisional balance",
          desc: "Explore the pros and cons of change and of the status quo.",
        },
        {
          name: "Readiness ruler",
          desc: "Assess importance and confidence, then elicit reasons for the rating.",
        },
      ],
      sessionPlan: [
        {
          phase: "Engagement",
          weeks: "Sessions 1–2",
          focus: "Build the alliance and understand the client's frame",
          items: ["Open-ended exploration", "Reflective listening practice", "Agenda mapping"],
        },
        {
          phase: "Focus & evoking",
          weeks: "Sessions 3–6",
          focus: "Develop discrepancy and elicit change talk",
          items: ["Decisional balance", "Readiness rulers", "Elicit–Provide–Elicit information exchange"],
        },
        {
          phase: "Planning",
          weeks: "Sessions 7–10",
          focus: "Consolidate commitment and plan",
          items: ["Change plan worksheet", "Strengthen commitment language", "Set first action steps"],
        },
      ],
      homework: [
        {
          name: "Change plan worksheet",
          cadence: "Between sessions",
          desc: "Draft steps, supports, and a commitment statement.",
        },
        {
          name: "Readiness reflection",
          cadence: "Weekly",
          desc: "Re-rate importance and confidence, noting what shifted.",
        },
        {
          name: "Pro-and-con log",
          cadence: "Weekly",
          desc: "Track the benefits and costs of change as they arise.",
        },
      ],
    },
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
    workspace: {
      conceptualization:
        "The humanistic formulation understands distress as a response to incongruence between the client's actual self and conditions of worth imposed from within. Symptoms reflect the organism's drive toward growth being blocked by self-critical or introjected expectations. The work creates a relational climate of congruence, empathy, and unconditional positive regard in which self-acceptance can be restored.",
      model: {
        label: "Person-centred theory",
        points: [
          "Self-actualization is an innate drive",
          "Conditions of worth distort self-experience",
          "The therapeutic relationship is the mechanism of change",
        ],
      },
      therapeuticFocus:
        "Restore the client's capacity for self-acceptance and authentic living through a genuine, non-directive relationship.",
      strengths: [
        "Deep relational safety and alliance strength",
        "Person-centred dignity and client autonomy",
        "Excellent fit for grief, identity, and meaning themes",
        "Transdiagnostic flexibility",
      ],
      limitations: [
        "Fewer manualized protocols; sessions are harder to structure",
        "Limited evidence for acute symptom reduction on its own",
        "Relies heavily on the therapist's relational skill",
      ],
      assessmentQuestions: [
        "What would it mean for you to feel truly yourself?",
        "When do you feel most judged — by others or by yourself?",
        "What do you need from this space to feel safe exploring?",
        "Where do you notice a gap between who you are and who you should be?",
      ],
      suggestedInterventions: [
        {
          name: "Empathic reflection",
          desc: "Mirroring meaning and feeling to deepen client self-exploration.",
        },
        {
          name: "Genuineness",
          desc: "Authentic, transparent presence in the therapeutic contact.",
        },
        {
          name: "Meaning-making dialogue",
          desc: "Explore narratives of loss, identity, and purpose together.",
        },
      ],
      sessionPlan: [
        {
          phase: "Rapport & exploration",
          weeks: "Sessions 1–3",
          focus: "Establish relational safety and shared understanding",
          items: ["Unconditional positive regard", "Open exploration of experience", "Notice conditions of worth"],
        },
        {
          phase: "Deepening",
          weeks: "Sessions 4–7",
          focus: "Work with incongruence and self-criticism",
          items: ["Reflect self-critical introjects", "Explore shoulds vs. wants", "Support emergent meaning"],
        },
        {
          phase: "Integration",
          weeks: "Sessions 8–10",
          focus: "Consolidate self-acceptance and plan the ending",
          items: ["Review shift in self-experience", "Plan for autonomy and maintenance", "Foster ongoing self-regard"],
        },
      ],
      homework: [
        {
          name: "Self-reflection journal",
          cadence: "Daily",
          desc: "Note moments of self-judgment and what being yourself felt like.",
        },
        {
          name: "Values grounding",
          cadence: "Weekly",
          desc: "Reconnect with a valued activity and notice internal responses.",
        },
      ],
    },
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
    workspace: {
      conceptualization:
        "Alliance ruptures are understood as markers in the therapeutic relationship — withdrawal or confrontation — that signal a breakdown in the collaborative bond. The formulation tracks the rupture event, the client's contribution such as negative expectations and misattributions, and the therapist's response. Left unrepaired, ruptures predict dropout; repaired, they can themselves become a therapeutic event.",
      model: {
        label: "Rupture and repair",
        points: [
          "Two rupture types: withdrawal and confrontation",
          "Client negative relational expectations fuel ruptures",
          "Repair is itself an intervention",
        ],
      },
      therapeuticFocus:
        "Detect ruptures early, name them non-defensively, and restore collaboration without shaming the client.",
      strengths: [
        "Directly targets the strongest predictor of outcome — the alliance",
        "Transdiagnostic and integrative",
        "Improves retention and engagement",
        "Brief, focused interventions",
      ],
      limitations: [
        "Requires therapist self-awareness and non-defensiveness",
        "Repair alone is not a complete therapy",
        "Subtle ruptures are easy to miss",
      ],
      assessmentQuestions: [
        "How has the work been feeling for you this week?",
        "Has anything I said or did sat differently with you?",
        "Did you hold anything back in our last session?",
        "What would make this feel more like we are on the same team?",
      ],
      suggestedInterventions: [
        {
          name: "Meta-communication",
          desc: "Name the rupture explicitly and invite the client's perspective.",
        },
        {
          name: "Validating the client's experience",
          desc: "Acknowledge the legitimacy of the rupture, apologizing where warranted.",
        },
        {
          name: "Repair dialogue",
          desc: "Explore the event, its meaning, and renegotiate the work.",
        },
      ],
      sessionPlan: [
        {
          phase: "Detect & normalize",
          weeks: "Session 1",
          focus: "Identify markers and address them early",
          items: ["Watch for withdrawal and confrontation", "Normalize ruptures", "Invite direct feedback"],
        },
        {
          phase: "Repair dialogue",
          weeks: "Sessions 2–3",
          focus: "Name the rupture and understand it",
          items: ["Meta-communication", "Explore the client's meaning", "Validate and apologize if indicated"],
        },
        {
          phase: "Renegotiate & rebuild",
          weeks: "Sessions 4–6",
          focus: "Restore collaboration and reinforce trust",
          items: ["Re-align on goals", "Strengthen the working alliance", "Monitor for recurrence"],
        },
      ],
      homework: [
        {
          name: "Alliance check-in",
          cadence: "Before sessions",
          desc: "Rate how the work is feeling (0–10) and leave one note for the therapist.",
        },
        {
          name: "Feedback log",
          cadence: "Weekly",
          desc: "Note any moment that felt off and what was needed.",
        },
      ],
    },
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
    workspace: {
      conceptualization:
        "General psychotherapy integrates common factors — alliance, empathy, expectation, and structure — into a coherent transdiagnostic frame. The formulation identifies shared mechanisms such as hope, regulation, insight, and behavioral change rather than diagnosis-specific targets, and selects techniques from across traditions based on what the client needs in the present moment.",
      model: {
        label: "Common-factors framework",
        points: [
          "Common factors drive a large share of outcome",
          "A shared, written formulation unifies the approach",
          "Technique is matched to client need, not to dogma",
        ],
      },
      therapeuticFocus:
        "Deliver a coherent, integrative therapy organized around the alliance, a shared formulation, and the client's own goals.",
      strengths: [
        "Flexible across presentations and severities",
        "Prioritizes the alliance — the strongest predictor of outcome",
        "Pragmatic and client-centered",
        "Smoothly integrates tools from every tradition",
      ],
      limitations: [
        "Less specific technique guidance",
        "Requires strong clinician judgment and breadth",
        "Risk of fuzziness without a clear written formulation",
      ],
      assessmentQuestions: [
        "What does this client need from therapy most right now?",
        "What shared, written formulation can we both work from?",
        "Which mechanism are we targeting this session — regulation, insight, or action?",
        "Are our goals and outcome measures still aligned?",
      ],
      suggestedInterventions: [
        {
          name: "Shared formulation",
          desc: "Build a transparent, written model of the problem with the client.",
        },
        {
          name: "Psychoeducation",
          desc: "Normalize symptoms and explain the rationale for treatment.",
        },
        {
          name: "Common-factors strengthening",
          desc: "Deepen the alliance, expectations, and structure throughout.",
        },
      ],
      sessionPlan: [
        {
          phase: "Alliance & formulation",
          weeks: "Sessions 1–2",
          focus: "Build the relationship and a written shared formulation",
          items: ["Assess goals and expectations", "Co-construct the formulation", "Agree outcome measures"],
        },
        {
          phase: "Mechanism-matched work",
          weeks: "Sessions 3–8",
          focus: "Target regulation, insight, or action as indicated",
          items: ["Select techniques from any tradition", "Monitor outcomes", "Adjust approach flexibly"],
        },
        {
          phase: "Consolidation",
          weeks: "Sessions 9–12",
          focus: "Embed gains and plan maintenance",
          items: ["Review outcomes", "Relapse prevention", "Graduated ending"],
        },
      ],
      homework: [
        {
          name: "Formulation journal",
          cadence: "Weekly",
          desc: "Note how the shared model explains the week's experiences.",
        },
        {
          name: "Outcome tracking",
          cadence: "Weekly",
          desc: "Complete the agreed measure each week.",
        },
        {
          name: "Skill practice",
          cadence: "Daily",
          desc: "Practice the active mechanism — regulation, insight, or action — chosen for the week.",
        },
      ],
    },
  },
];

/** Framework cards that map to a curated API framework retriever. */
export const FRAMEWORK_ID_TO_API: Record<FrameworkCardId, Framework | null> = {
  cbt: "cbt",
  act: "act",
  dbt: null,
  mi: null,
  humanistic: "humanistic",
  "alliance-repair": null,
  general: null,
};

export const FRAMEWORK_ID_TO_KB: Partial<Record<FrameworkCardId, string>> = {
  cbt: "cbt",
  act: "act",
  humanistic: "humanistic",
};
