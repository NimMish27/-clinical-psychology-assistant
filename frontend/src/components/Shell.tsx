import type { ReactNode } from "react";
import { useNav } from "../lib/nav";
import { getCase, SYSTEM_STATUS } from "../lib/sampleData";
import { Badge, Meter, ProgressSegments, Ring } from "../ui/primitives";
import {
  ActivityIcon,
  BookIcon,
  BrainIcon,
  ClockIcon,
  DatabaseIcon,
  FolderIcon,
  GridIcon,
  LibraryIcon,
  SearchIcon,
  SettingsIcon,
  ShieldIcon,
  SparklesIcon,
} from "../ui/icons";

const RAIL_ITEMS = [
  { section: "workspace", label: "Workspace", icon: GridIcon },
  { section: "cases", label: "Cases", icon: FolderIcon },
  { section: "evidence", label: "Evidence", icon: LibraryIcon },
  { section: "knowledge", label: "Knowledge", icon: BookIcon },
  { section: "safety", label: "Safety", icon: ShieldIcon },
  { section: "settings", label: "Settings", icon: SettingsIcon },
] as const;

export default function Shell({ children }: { children: ReactNode }) {
  const { route } = useNav();
  const caseId = route.section === "cases" ? route.caseId : undefined;

  return (
    <div className="shell">
      <NavRail />
      <SectionSidebar />
      <div className="main">
        <MainHeader />
        <div className="main-content">{children}</div>
      </div>
      {caseId && <InsightRail caseId={caseId} />}
    </div>
  );
}

/* ── Nav rail ─────────────────────────────────────────────────────── */

function NavRail() {
  const { route, openSection } = useNav();
  return (
    <nav className="rail" aria-label="Primary">
      <div className="rail-brand">
        <BrainIcon />
      </div>
      {RAIL_ITEMS.map(({ section, label, icon: Icon }) => (
        <button
          key={section}
          className={`rail-item ${route.section === section ? "active" : ""}`}
          onClick={() => openSection(section)}
          aria-label={label}
        >
          <Icon />
          {section === "safety" && <span className="rail-badge">1</span>}
          <span className="rail-tip">{label}</span>
        </button>
      ))}
      <div className="rail-spacer" />
    </nav>
  );
}

/* ── Section sidebar ──────────────────────────────────────────────── */

function SectionSidebar() {
  const { route } = useNav();

  if (route.section === "cases") {
    return <CasesSidebar />;
  }

  const config: Record<string, { head: string; sub: string; items: Array<{ id: string; label: string }> }> = {
    workspace: {
      head: "Workspace",
      sub: "Today",
      items: [
        { id: "overview", label: "Overview" },
        { id: "activity", label: "Activity" },
        { id: "status", label: "System status" },
      ],
    },
    evidence: {
      head: "Evidence",
      sub: "Literature",
      items: [
        { id: "explorer", label: "Evidence Explorer" },
        { id: "saved", label: "Saved searches" },
      ],
    },
    knowledge: {
      head: "Knowledge",
      sub: "Sources",
      items: [
        { id: "framework", label: "Framework Knowledge" },
        { id: "documents", label: "My Documents" },
      ],
    },
    safety: {
      head: "Safety",
      sub: "Guardrails",
      items: [
        { id: "center", label: "Safety Center" },
        { id: "checklist", label: "Sign-off checklist" },
        { id: "incidents", label: "Incident log" },
      ],
    },
    settings: {
      head: "Settings",
      sub: "Preferences",
      items: [
        { id: "general", label: "General" },
        { id: "kb", label: "Knowledge base" },
        { id: "llm", label: "LLM & retrieval" },
        { id: "about", label: "About" },
      ],
    },
  };

  const { head, sub, items } = config[route.section];
  return (
    <aside className="section-sidebar">
      <div className="sidebar-section-head">
        <div className="eyebrow">{sub}</div>
        <h2>{head}</h2>
      </div>
      <div className="sidebar-nav">
        {items.map((item, i) => (
          <button key={item.id} className={`sidebar-item ${i === 0 ? "active" : ""}`}>
            {item.label}
          </button>
        ))}
      </div>
    </aside>
  );
}

function CasesSidebar() {
  const { route, openCase, toCasesIndex } = useNav();
  const cases = ["case-1042", "case-1038", "case-1031", "case-1027", "case-1019"];

  if (route.caseId) {
    const record = getCase(route.caseId);
    return (
      <aside className="section-sidebar">
        <div className="sidebar-section-head">
          <div className="eyebrow">Cases</div>
          <h2>{record ? record.name : "Case"}</h2>
        </div>
        <div className="sidebar-nav">
          <button className="sidebar-item" onClick={toCasesIndex}>
            <SearchIcon /> Case Index
          </button>
        </div>
        <div className="sidebar-group-label">Case tabs</div>
        <div className="sidebar-nav">
          {[
            { id: "overview", label: "Overview" },
            { id: "assessment", label: "Assessment" },
            { id: "evidence", label: "Evidence" },
            { id: "formulation", label: "Formulation" },
            { id: "plan", label: "Plan" },
            { id: "safety", label: "Safety" },
          ].map((t) => (
            <button
              key={t.id}
              className={`sidebar-item ${route.caseTab === t.id || (t.id === "overview" && !route.caseTab) ? "active" : ""}`}
              onClick={() => openCase(route.caseId!, t.id as never)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="sidebar-group-label">Other cases</div>
        <div className="sidebar-case">
          {cases
            .filter((id) => id !== route.caseId)
            .map((id) => {
              const c = getCase(id);
              if (!c) return null;
              return (
                <button key={id} className="sidebar-case-item" onClick={() => openCase(id, "overview")}>
                  <span className="sidebar-case-avatar" style={{ background: c.color }}>
                    {c.initials}
                  </span>
                  <span className="sidebar-case-info">
                    <span className="sidebar-case-name">{c.name}</span>
                    <span className="sidebar-case-sub">{c.presentingConcern}</span>
                  </span>
                </button>
              );
            })}
        </div>
      </aside>
    );
  }

  return (
    <aside className="section-sidebar">
      <div className="sidebar-section-head">
        <div className="eyebrow">Cases</div>
        <h2>Case Index</h2>
      </div>
      <div className="sidebar-nav">
        <button className="sidebar-item active">
          <FolderIcon /> All cases <span className="count">5</span>
        </button>
        <button className="sidebar-item">
          <ActivityIcon /> In progress <span className="count">2</span>
        </button>
        <button className="sidebar-item">
          <ClockIcon /> Needs attention <span className="count">2</span>
        </button>
      </div>
      <div className="sidebar-group-label">Recent</div>
      <div className="sidebar-case">
        {cases.map((id) => {
          const c = getCase(id);
          if (!c) return null;
          return (
            <button key={id} className="sidebar-case-item" onClick={() => openCase(id, "overview")}>
              <span className="sidebar-case-avatar" style={{ background: c.color }}>
                {c.initials}
              </span>
              <span className="sidebar-case-info">
                <span className="sidebar-case-name">{c.name}</span>
                <span className="sidebar-case-sub">{c.presentingConcern}</span>
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

/* ── Main header ──────────────────────────────────────────────────── */

function MainHeader() {
  const { route } = useNav();
  const record = route.section === "cases" && route.caseId ? getCase(route.caseId) : undefined;

  return (
    <header className="main-header">
      {record ? (
        <div className="crumbs">
          <span>Cases</span> / <b>{record.name}</b>
          <Badge tone={record.stage === "safety" ? "warning" : "neutral"}>
            {record.presentingConcern}
          </Badge>
        </div>
      ) : (
        <h1>Clinical Psychology Assistant</h1>
      )}
      <div className="header-spacer" />
      <div className="search-box" style={{ maxWidth: 260 }}>
        <SearchIcon />
        <input className="input" placeholder="Search… (⌘K)" />
      </div>
      <span
        className="icon-btn"
        title="Command palette (⌘K)"
        onClick={() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))}
      >
        <SparklesIcon />
      </span>
    </header>
  );
}

/* ── Insight rail (case workspace) ────────────────────────────────── */

function InsightRail({ caseId }: { caseId: string }) {
  const record = getCase(caseId);
  const { openCase } = useNav();
  if (!record) return null;

  const done = record.safetyChecklist.filter((s) => s.done).length;
  const total = record.safetyChecklist.length;

  return (
    <aside className="insight">
      <div className="insight-card">
        <h4>
          <ActivityIcon /> Stage progress
        </h4>
        <ProgressSegments value={record.stageProgress} />
        <Meter value={record.stageProgress} label="Overall progress" />
      </div>

      <div className="insight-card">
        <h4>
          <ShieldIcon /> Safety
        </h4>
        <Ring value={total === 0 ? 0 : done / total} size={52} caption="sign-off" />
        <div style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-2)" }}>
          {done}/{total} checklist items completed
        </div>
        {record.riskFlags.length > 0 ? (
          record.riskFlags.map((f, i) => (
            <Badge key={i} tone={f.level === "high" ? "danger" : f.level === "moderate" ? "warning" : "info"} dot>
              {f.label}
            </Badge>
          ))
        ) : (
          <Badge tone="success" dot>
            No active risk flags
          </Badge>
        )}
        <button className="btn btn-sm" onClick={() => openCase(caseId, "safety")}>
          Open safety tab
        </button>
      </div>

      <div className="insight-card">
        <h4>
          <ClockIcon /> Schedule
        </h4>
        <div className="detail">
          <span className="detail-label">Next appointment</span>
          <span className="detail-value">{record.nextAppointment}</span>
        </div>
        <div className="detail">
          <span className="detail-label">Last updated</span>
          <span className="detail-value">{record.updatedAt}</span>
        </div>
        <div className="detail">
          <span className="detail-label">Framework</span>
          <span className="detail-value" style={{ textTransform: "capitalize" }}>
            {record.framework}
          </span>
        </div>
      </div>

      <div className="insight-card">
        <h4>
          <DatabaseIcon /> System
        </h4>
        {Object.entries(SYSTEM_STATUS).map(([key, s]) => (
          <div key={key} className="list-row" style={{ padding: "6px 0", borderBottom: "1px solid var(--c-hairline)" }}>
            <span className="status-dot" style={{ background: "var(--c-success)" }} />
            <span style={{ flex: 1, fontSize: "var(--text-sm)", textTransform: "capitalize" }}>{key}</span>
            <span style={{ fontSize: "var(--text-xs)", color: "var(--c-ink-3)" }}>
              {s.latency !== undefined ? `${s.latency}ms` : s.detail ?? "ok"}
            </span>
          </div>
        ))}
      </div>
    </aside>
  );
}
