import type { CaseRecord } from "./types";

export const CASES: CaseRecord[] = [
  {
    id: "case-1042",
    initials: "JM",
    name: "Jordan Mitchell",
    age: 34,
    gender: "F",
    status: "in-progress",
    stage: "evidence",
    presentingConcern: "Generalized anxiety with rumination",
    framework: "cbt",
    clinician: "Dr. A. Reyes",
    updatedAt: "2026-08-10",
    nextAppointment: "Aug 14, 9:30 AM",
    stageProgress: 48,
    color: "#6366f1",
    concerns: [
      "Excessive worry about work performance",
      "Difficulty sleeping, onset insomnia",
      "Muscle tension and restlessness",
      "Rumination loops at night",
    ],
    history:
      "Symptom onset ~18 months ago following a promotion. No prior mental health treatment. Family history of anxiety on maternal side.",
    goals: [
      "Reduce nightly rumination episodes from nightly to ≤2 per week",
      "Practice cognitive restructuring for work-performance thoughts",
      "Establish consistent sleep-wake schedule",
    ],
    interventions: ["Cognitive restructuring", "Stimulus control for sleep", "Worry-time scheduling"],
    safetyNotes: [
      "No active suicidal ideation. PHQ-9 item 9 = 0.",
      "Occasional passive thoughts of 'not coping', no plan or intent.",
      "Denies self-harm history.",
    ],
    riskFlags: [{ level: "low", label: "Anxiety-related insomnia" }],
    safetyChecklist: [
      { id: "s1", label: "Suicidal ideation screen completed", detail: "PHQ-9 item 9 + direct inquiry", done: true },
      { id: "s2", label: "Risk escalation plan reviewed", detail: "Clinician + 24h contact", done: true },
      { id: "s3", label: "Self-harm history documented", detail: "None reported", done: true },
      { id: "s4", label: "Crisis resources provided", detail: "Hotline card handed out", done: false },
      { id: "s5", label: "Caregiver / support contact logged", detail: "Sister: +1 (555) 010-3344", done: false },
    ],
  },
  {
    id: "case-1038",
    initials: "RL",
    name: "Rafael Luna",
    age: 41,
    gender: "M",
    status: "open",
    stage: "understanding",
    presentingConcern: "Chronic pain-related distress",
    framework: "act",
    clinician: "Dr. S. Okafor",
    updatedAt: "2026-08-09",
    nextAppointment: "Aug 15, 11:00 AM",
    stageProgress: 22,
    color: "#0ea5e9",
    concerns: ["Persistent low-back pain 6 yrs", "Catastrophizing about flare-ups", "Activity avoidance"],
    history:
      "Chronic musculoskeletal pain following workplace injury. Prior physio; limited psychological intervention.",
    goals: ["Increase values-based activity despite discomfort", "Practice defusion with pain-related thoughts"],
    interventions: ["Values clarification", "Cognitive defusion", "Graded activity"],
    safetyNotes: ["No acute safety concerns reported."],
    riskFlags: [],
    safetyChecklist: [
      { id: "s1", label: "Suicidal ideation screen completed", detail: "No active ideation", done: true },
      { id: "s2", label: "Risk escalation plan reviewed", detail: "Standard plan", done: false },
    ],
  },
  {
    id: "case-1031",
    initials: "PD",
    name: "Priya Deshmukh",
    age: 28,
    gender: "F",
    status: "open",
    stage: "understanding",
    presentingConcern: "Grief and adjustment",
    framework: "humanistic",
    clinician: "Dr. M. Chen",
    updatedAt: "2026-08-08",
    nextAppointment: "Aug 16, 2:00 PM",
    stageProgress: 18,
    color: "#10b981",
    concerns: ["Prolonged grief following loss", "Social withdrawal", "Low energy"],
    history: "Loss of parent 8 months ago. Struggling to re-engage with routine.",
    goals: ["Explore grief narrative in safe space", "Reconnect with two valued relationships"],
    interventions: ["Person-centered exploration", "Empathic reflection"],
    safetyNotes: ["No safety concerns."],
    riskFlags: [],
    safetyChecklist: [
      { id: "s1", label: "Suicidal ideation screen completed", detail: "None", done: true },
    ],
  },
  {
    id: "case-1027",
    initials: "TW",
    name: "Tyler Whitfield",
    age: 52,
    gender: "M",
    status: "on-hold",
    stage: "plan",
    presentingConcern: "Depression, mild-moderate",
    framework: "cbt",
    clinician: "Dr. A. Reyes",
    updatedAt: "2026-07-30",
    nextAppointment: "—",
    stageProgress: 71,
    color: "#f59e0b",
    concerns: ["Anhedonia", "Reduced appetite", "Low motivation"],
    history: "Episode ~4 months; responsive to CBT so far. Paused for vacation.",
    goals: ["Behavioral activation with 3 weekly activities", "Identify early-warning signs"],
    interventions: ["Behavioral activation", "Activity scheduling"],
    safetyNotes: ["Stable mood, no acute risk."],
    riskFlags: [],
    safetyChecklist: [
      { id: "s1", label: "Suicidal ideation screen completed", detail: "None", done: true },
    ],
  },
  {
    id: "case-1019",
    initials: "AK",
    name: "Ava Kim",
    age: 19,
    gender: "F",
    status: "in-progress",
    stage: "safety",
    presentingConcern: "Panic disorder with agoraphobia",
    framework: "cbt",
    clinician: "Dr. S. Okafor",
    updatedAt: "2026-08-11",
    nextAppointment: "Aug 13, 10:00 AM",
    stageProgress: 92,
    color: "#ef4444",
    concerns: ["Recurrent panic attacks", "Avoiding public transport", "Fear of losing control"],
    history: "Onset following a panic episode at university. Now avoids commuting alone.",
    goals: ["Complete 4 interoceptive exposure sessions", "Attend class solo via train"],
    interventions: ["Interoceptive exposure", "In-vivo exposure hierarchy", "Psychoeducation"],
    safetyNotes: [
      "No self-harm or suicidal ideation.",
      "Mild agoraphobic avoidance, containment plan in place.",
    ],
    riskFlags: [{ level: "low", label: "Situational avoidance" }],
    safetyChecklist: [
      { id: "s1", label: "Suicidal ideation screen completed", detail: "None", done: true },
      { id: "s2", label: "Risk escalation plan reviewed", detail: "Yes", done: true },
      { id: "s3", label: "Exposure safety protocol signed", detail: "Client + clinician", done: true },
      { id: "s4", label: "Crisis resources provided", detail: "Done", done: true },
    ],
  },
];

export function getCase(id: string): CaseRecord | undefined {
  return CASES.find((c) => c.id === id);
}

export const ATTENTION_QUEUE = [
  {
    id: "case-1019",
    label: "Ava Kim",
    reason: "Safety checklist incomplete — stage at 92%",
    level: "warning" as const,
  },
  {
    id: "case-1038",
    label: "Rafael Luna",
    reason: "No update in 8 days",
    level: "info" as const,
  },
];

export const RECENT_ACTIVITY = [
  { id: "case-1042", text: "Evidence search run for Jordan Mitchell", time: "2h ago" },
  { id: "case-1019", text: "Safety sign-off 4/4 completed by Dr. S. Okafor", time: "5h ago" },
  { id: "case-1031", text: "Intake notes entered for Priya Deshmukh", time: "Yesterday" },
  { id: "case-1038", text: "ACT evidence bundle added to Rafael Luna", time: "2 days ago" },
];

export interface SystemStatusEntry {
  status: "ok" | "degraded" | "down";
  latency?: number;
  detail?: string;
}

export const SYSTEM_STATUS: Record<string, SystemStatusEntry> = {
  api: { status: "ok", latency: 42 },
  ollama: { status: "ok", latency: 188 },
  chroma: { status: "ok", latency: 11 },
  embedding: { status: "ok", latency: 96 },
  knowledgeBase: { status: "ok", detail: "act · 4 chunks" },
};

export interface SessionRecord {
  id: string;
  clientName: string;
  clientInitials: string;
  color: string;
  date: string;
  time: string;
  durationMin: number;
  framework: string;
  stage: string;
  notes: string;
}

export const SESSIONS: SessionRecord[] = [
  {
    id: "s-221",
    clientName: "Jordan Mitchell",
    clientInitials: "JM",
    color: "#6366f1",
    date: "Today",
    time: "9:30 AM",
    durationMin: 50,
    framework: "CBT",
    stage: "Evidence",
    notes: "Cognitive restructuring worksheet introduced; sleep diary reviewed.",
  },
  {
    id: "s-220",
    clientName: "Ava Kim",
    clientInitials: "AK",
    color: "#ef4444",
    date: "Today",
    time: "10:00 AM",
    durationMin: 55,
    framework: "CBT",
    stage: "Safety",
    notes: "Interoceptive exposure #3 completed; panic ratings 6/10 → 3/10.",
  },
  {
    id: "s-219",
    clientName: "Rafael Luna",
    clientInitials: "RL",
    color: "#0ea5e9",
    date: "Yesterday",
    time: "11:00 AM",
    durationMin: 45,
    framework: "ACT",
    stage: "Understanding",
    notes: "Values card sort — 5 core values identified.",
  },
  {
    id: "s-218",
    clientName: "Priya Deshmukh",
    clientInitials: "PD",
    color: "#10b981",
    date: "Yesterday",
    time: "2:00 PM",
    durationMin: 50,
    framework: "Humanistic",
    stage: "Understanding",
    notes: "Grief narrative exploration; empathic reflection deepened alliance.",
  },
  {
    id: "s-217",
    clientName: "Tyler Whitfield",
    clientInitials: "TW",
    color: "#f59e0b",
    date: "Jul 30",
    time: "3:30 PM",
    durationMin: 45,
    framework: "CBT",
    stage: "Plan",
    notes: "Behavioral activation schedule set for the coming week.",
  },
];
