import type { Framework } from "../api/client";

export type Section =
  | "workspace"
  | "cases"
  | "evidence"
  | "frameworks"
  | "knowledge"
  | "safety"
  | "settings";

export type CaseStage = "understanding" | "evidence" | "formulation" | "plan" | "safety";

export type CaseTab =
  | "overview"
  | "analysis"
  | "assessment"
  | "evidence"
  | "formulation"
  | "plan"
  | "safety";

export type CaseStatus = "open" | "in-progress" | "on-hold" | "closed";

export interface ClinicalCase {
  id: string;
  initials: string;
  name: string;
  age: number;
  gender: string;
  status: CaseStatus;
  stage: CaseStage;
  presentingConcern: string;
  framework: Framework | "general";
  clinician: string;
  updatedAt: string;
  nextAppointment: string;
  stageProgress: number; // 0-100
  color: string;
}

export interface CaseRecord extends ClinicalCase {
  concerns: string[];
  symptoms: string[];
  stressors: string[];
  protectiveFactors: string[];
  history: string;
  goals: string[];
  interventions: string[];
  safetyNotes: string[];
  riskFlags: { level: "none" | "low" | "moderate" | "high"; label: string }[];
  safetyChecklist: { id: string; label: string; detail: string; done: boolean }[];
}

export interface Route {
  section: Section;
  caseId?: string;
  caseTab?: CaseTab;
  frameworkId?: string;
}
