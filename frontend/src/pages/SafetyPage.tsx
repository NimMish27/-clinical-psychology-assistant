import { CASES } from "../lib/sampleData";
import { useNav } from "../lib/nav";
import { Alert, Badge, Card, Meter, Panel } from "../ui/primitives";
import { CheckIcon, ShieldIcon } from "../ui/icons";

export default function SafetyPage() {
  const { openCase } = useNav();

  const totals = CASES.map((c) => {
    const done = c.safetyChecklist.filter((s) => s.done).length;
    return { ...c, done, total: c.safetyChecklist.length, pct: c.safetyChecklist.length ? (done / c.safetyChecklist.length) * 100 : 0 };
  });

  const avg = Math.round(totals.reduce((a, t) => a + t.pct, 0) / Math.max(1, totals.length));
  const incomplete = totals.filter((t) => t.pct < 100);

  return (
    <div className="canvas">
      <div className="section-head">
        <div>
          <div className="kicker">Safety</div>
          <h1 className="section-title">Safety Center</h1>
          <p className="section-sub">Ambient guardrails — sign-off gates, risk flags, and escalation support.</p>
        </div>
      </div>

      <Alert tone="info" title="How safety works here">
        The Safety stage is a gate: a case cannot progress beyond the formulation and plan work until its
        sign-off checklist is complete. The system surfaces risks but the clinician always owns the judgment.
      </Alert>

      <div className="stat-grid">
        <Card padded>
          <div className="stat-label"><ShieldIcon /> Checklist compliance</div>
          <div className="stat-value">{avg}%</div>
          <div className="stat-trend">across all active cases</div>
        </Card>
        <Card padded>
          <div className="stat-label">Open risk flags</div>
          <div className="stat-value">{CASES.reduce((a, c) => a + c.riskFlags.length, 0)}</div>
          <div className="stat-trend">all low or monitored</div>
        </Card>
        <Card padded>
          <div className="stat-label">Incomplete sign-offs</div>
          <div className="stat-value">{incomplete.length}</div>
          <div className="stat-trend">cases with pending items</div>
        </Card>
      </div>

      <Panel title="Sign-off status by case">
        {totals.map((c) => (
          <div key={c.id} className="list-row clickable" onClick={() => openCase(c.id, "safety")}>
            <span className="avatar" style={{ background: c.color, width: 28, height: 28, fontSize: "var(--text-xs)" }}>
              {c.initials}
            </span>
            <div className="row-main">
              <div className="row-title" style={{ fontSize: "var(--text-sm)" }}>{c.name}</div>
              <div className="row-sub">
                {c.done}/{c.total} items
              </div>
            </div>
            <div style={{ width: 140 }}>
              <Meter value={c.pct} showValue={false} />
            </div>
            <Badge tone={c.pct === 100 ? "success" : "warning"} dot>
              {c.pct === 100 ? "Signed off" : `${Math.round(c.pct)}%`}
            </Badge>
          </div>
        ))}
      </Panel>

      <Panel title="Safety checklist reference">
        <div className="checklist">
          {[
            { label: "Suicidal ideation screen completed", desc: "Direct inquiry + validated screen (e.g. PHQ-9 item 9)" },
            { label: "Risk escalation plan reviewed", desc: "Clinician on-call + 24h contact path documented" },
            { label: "Self-harm history documented", desc: "Including current frequency and intensity" },
            { label: "Crisis resources provided", desc: "Hotline cards, emergency services, safety plan sheet" },
            { label: "Caregiver / support contact logged", desc: "Consented support person on file" },
          ].map((item) => (
            <div key={item.label} className="checklist-item">
              <span className="check done">
                <CheckIcon />
              </span>
              <div>
                <div className="checklist-title">{item.label}</div>
                <div className="checklist-desc">{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
