import { useMemo, useState } from "react";
import { CASES } from "../lib/sampleData";
import { useNav } from "../lib/nav";
import type { CaseStage } from "../lib/types";
import { Badge, Button, Meter } from "../ui/primitives";
import { ChevronRightIcon, PlusIcon, SearchIcon } from "../ui/icons";

const STAGE_ORDER: CaseStage[] = ["understanding", "evidence", "formulation", "plan", "safety"];

export default function CaseIndexPage() {
  const { openCase } = useNav();
  const [query, setQuery] = useState("");
  const [stage, setStage] = useState<CaseStage | "all">("all");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return CASES.filter((c) => {
      const matchesQuery =
        !q ||
        c.name.toLowerCase().includes(q) ||
        c.presentingConcern.toLowerCase().includes(q) ||
        c.id.toLowerCase().includes(q);
      const matchesStage = stage === "all" || c.stage === stage;
      return matchesQuery && matchesStage;
    });
  }, [query, stage]);

  return (
    <div className="canvas">
      <div className="section-head">
        <div>
          <div className="kicker">Cases</div>
          <h1 className="section-title">Case Index</h1>
          <p className="section-sub">{CASES.length} cases total · {CASES.filter((c) => c.status === "open" || c.status === "in-progress").length} active</p>
        </div>
        <Button variant="primary">
          <PlusIcon /> New case
        </Button>
      </div>

      <div className="toolbar">
        <div className="search-box">
          <SearchIcon />
          <input
            className="input"
            placeholder="Search by client, concern, or ID…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <select className="select" value={stage} onChange={(e) => setStage(e.target.value as CaseStage | "all")}>
          <option value="all">All stages</option>
          {STAGE_ORDER.map((s) => (
            <option key={s} value={s}>
              {s[0].toUpperCase() + s.slice(1)}
            </option>
          ))}
        </select>
        <span style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-3)" }}>
          {filtered.length} result{filtered.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="panel">
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Client</th>
                <th>Presenting concern</th>
                <th>Framework</th>
                <th>Status</th>
                <th>Stage</th>
                <th>Progress</th>
                <th>Next appointment</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {filtered.map((c) => (
                <tr key={c.id} className="table-row-click" onClick={() => openCase(c.id, "overview")}>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span className="avatar" style={{ background: c.color, width: 28, height: 28, fontSize: "var(--text-xs)" }}>
                        {c.initials}
                      </span>
                      <div>
                        <div className="table-cell-strong">{c.name}</div>
                        <div className="table-cell-sub">
                          {c.id} · {c.age}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="table-cell-sub">{c.presentingConcern}</td>
                  <td>
                    <Badge tone={c.framework === "act" ? "info" : c.framework === "humanistic" ? "success" : "primary"}>
                      {c.framework}
                    </Badge>
                  </td>
                  <td>
                    <Badge
                      tone={
                        c.status === "in-progress"
                          ? "primary"
                          : c.status === "open"
                            ? "info"
                            : c.status === "on-hold"
                              ? "warning"
                              : "neutral"
                      }
                      dot
                    >
                      {c.status}
                    </Badge>
                  </td>
                  <td className="table-cell-sub" style={{ textTransform: "capitalize" }}>
                    {c.stage}
                  </td>
                  <td>
                    <div style={{ width: 100 }}>
                      <Meter value={c.stageProgress} showValue={false} />
                    </div>
                  </td>
                  <td className="table-cell-sub">{c.nextAppointment}</td>
                  <td>
                    <ChevronRightIcon style={{ width: 15, height: 15, color: "var(--c-ink-4)" }} />
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={8}>
                    <div className="empty-state" style={{ padding: "var(--space-8)" }}>
                      <SearchIcon />
                      <h3>No cases match</h3>
                      <p>Try a different search or clear the stage filter.</p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
