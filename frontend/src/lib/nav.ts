import { createContext, useContext } from "react";
import type { CaseTab, Route, Section } from "./types";
import type { FrameworkCardId } from "./frameworks";

export interface NavState {
  route: Route;
  navigate: (route: Route) => void;
  openSection: (section: Section) => void;
  openCase: (caseId: string, tab?: CaseTab) => void;
  openFramework: (frameworkId: FrameworkCardId) => void;
  toCasesIndex: () => void;
}

export const NavContext = createContext<NavState>({
  route: { section: "workspace" },
  navigate: () => {},
  openSection: () => {},
  openCase: () => {},
  openFramework: () => {},
  toCasesIndex: () => {},
});

export function useNav(): NavState {
  return useContext(NavContext);
}

export const STAGE_LABELS: Record<string, string> = {
  understanding: "Understanding",
  evidence: "Evidence",
  formulation: "Formulation",
  plan: "Plan",
  safety: "Safety",
};

export const CASE_TABS: Array<{ id: CaseTab; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "analysis", label: "Analysis" },
  { id: "assessment", label: "Assessment" },
  { id: "evidence", label: "Evidence" },
  { id: "formulation", label: "Formulation" },
  { id: "plan", label: "Plan" },
  { id: "safety", label: "Safety" },
];
