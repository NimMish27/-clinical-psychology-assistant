import { useCallback, useMemo, useState } from "react";
import Shell from "./components/Shell";
import { NavContext } from "./lib/nav";
import type { Route } from "./lib/types";
import { CommandPaletteProvider, ToastProvider, type PaletteAction } from "./ui/primitives";
import { BookIcon, FolderIcon, GridIcon, LibraryIcon, SettingsIcon, ShieldIcon } from "./ui/icons";
import HomeDashboard from "./pages/HomeDashboard";
import CaseIndexPage from "./pages/CaseIndexPage";
import CaseWorkspacePage from "./pages/CaseWorkspacePage";
import EvidencePage from "./pages/EvidencePage";
import KnowledgeBasePage from "./pages/KnowledgeBasePage";
import SafetyPage from "./pages/SafetyPage";
import SettingsPage from "./pages/SettingsPage";

export default function App() {
  const [route, setRoute] = useState<Route>({ section: "workspace" });

  const openSection = useCallback((section: Route["section"]) => {
    setRoute({ section });
  }, []);

  const openCase = useCallback((caseId: string, caseTab?: string) => {
    setRoute({ section: "cases", caseId, caseTab: (caseTab ?? "overview") as never });
  }, []);

  const toCasesIndex = useCallback(() => setRoute({ section: "cases" }), []);

  const nav = useMemo(
    () => ({
      route,
      navigate: setRoute,
      openSection,
      openCase,
      toCasesIndex,
    }),
    [route, openSection, openCase, toCasesIndex],
  );

  const defaultCommands = useCallback(
    (): PaletteAction[] => [
      { id: "g-ws", label: "Go to Workspace", keywords: "home dashboard overview", icon: <GridIcon />, action: () => openSection("workspace") },
      { id: "g-cases", label: "Go to Case Index", keywords: "cases clients index", icon: <FolderIcon />, action: toCasesIndex },
      { id: "g-evidence", label: "Open Evidence Explorer", keywords: "literature search evidence ask", icon: <LibraryIcon />, action: () => openSection("evidence") },
      { id: "g-kb", label: "Open Knowledge Base", keywords: "knowledge documents frameworks", icon: <BookIcon />, action: () => openSection("knowledge") },
      { id: "g-safety", label: "Open Safety Center", keywords: "safety risk checklist signoff", icon: <ShieldIcon />, action: () => openSection("safety") },
      { id: "g-settings", label: "Open Settings", keywords: "settings preferences", icon: <SettingsIcon />, action: () => openSection("settings") },
      ...["case-1042", "case-1038", "case-1031", "case-1027", "case-1019"].map((id) => ({
        id: `c-${id}`,
        label: `Open case ${id}`,
        keywords: "case client",
        icon: <FolderIcon />,
        action: () => openCase(id, "overview"),
      })),
    ],
    [openSection, toCasesIndex, openCase],
  );

  let page: React.ReactNode;
  if (route.section === "cases") {
    page = route.caseId ? <CaseWorkspacePage caseId={route.caseId} /> : <CaseIndexPage />;
  } else if (route.section === "evidence") {
    page = <EvidencePage />;
  } else if (route.section === "knowledge") {
    page = <KnowledgeBasePage />;
  } else if (route.section === "safety") {
    page = <SafetyPage />;
  } else if (route.section === "settings") {
    page = <SettingsPage />;
  } else {
    page = <HomeDashboard />;
  }

  return (
    <ToastProvider>
      <CommandPaletteProvider onDefaultOpen={defaultCommands}>
        <NavContext.Provider value={nav}>
          <Shell>{page}</Shell>
        </NavContext.Provider>
      </CommandPaletteProvider>
    </ToastProvider>
  );
}
