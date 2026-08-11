import { useCallback, useEffect, useRef, useState } from "react";
import { searchEvidence, type EvidenceItem, type EvidenceSearchResponse } from "../api/client";
import { useNav, CASE_TABS } from "../lib/nav";
import { getCase } from "../lib/sampleData";
import { Alert, Badge, Button, Card, EmptyState, ErrorState, Meter, Panel, Ring, Skeleton, StageBar, Tabs } from "../ui/primitives";
import CaseAnalysisPage from "./CaseAnalysisPage";
import {
  AlertTriangleIcon,
  BookIcon,
  CheckIcon,
  ExternalLinkIcon,
  LibraryIcon,
  SearchIcon,
  ShieldIcon,
} from "../ui/icons";

export default function CaseWorkspacePage({ caseId }: { caseId: string }) {
  const { route, navigate, toCasesIndex } = useNav();
  const record = getCase(caseId);
  const tab = route.caseTab ?? "overview";

  if (!record) {
    return (
      <div className="canvas">
        <ErrorState title="Case not found" detail={`No case exists with ID ${caseId}.`} />
      </div>
    );
  }

  const showStageBar = record.stageProgress < 100;

  return (
    <div className="canvas">
      <div className="rise">
        <button className="btn btn-ghost btn-sm" onClick={toCasesIndex} style={{ marginBottom: "var(--space-3)" }}>
          <span style={{ display: "inline-flex" }}>
            <ExternalLinkIcon style={{ transform: "rotate(180deg)", width: 14, height: 14 }} />
          </span>
          Back to Case Index
        </button>

        <div className="case-hero">
          <span className="avatar" style={{ background: record.color }}>
            {record.initials}
          </span>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
              <h2>{record.name}</h2>
              <Badge tone={record.stage === "safety" ? "warning" : "primary"} dot>
                {record.stage}
              </Badge>
            </div>
            <div className="case-hero-sub">
              <span>{record.age} · {record.gender}</span>
              <span>·</span>
              <span>{record.presentingConcern}</span>
              <span>·</span>
              <span>{record.clinician}</span>
            </div>
          </div>
        </div>

        {showStageBar && (
          <Card padded style={{ marginTop: "var(--space-4)" }}>
            <StageBar current={record.stage} />
          </Card>
        )}
      </div>

      <div className="panel" style={{ overflow: "visible" }}>
        <Tabs
          tabs={CASE_TABS.map((t) => ({ id: t.id, label: t.label }))}
          active={tab}
          onChange={(id) => navigate({ section: "cases", caseId, caseTab: id as never })}
        />
      </div>

      <div className="rise" key={tab}>
        {tab === "overview" && <OverviewTab caseId={caseId} />}
        {tab === "analysis" && <CaseAnalysisPage caseId={caseId} />}
        {tab === "assessment" && <AssessmentTab caseId={caseId} />}
        {tab === "evidence" && <CaseEvidenceTab caseId={caseId} />}
        {tab === "formulation" && <FormulationTab />}
        {tab === "plan" && <PlanTab />}
        {tab === "safety" && <SafetyTab caseId={caseId} />}
      </div>
    </div>
  );
}

/* ── Overview ─────────────────────────────────────────────────────── */

function OverviewTab({ caseId }: { caseId: string }) {
  const { openCase } = useNav();
  const record = getCase(caseId)!;
  const total = record.safetyChecklist.length;
  const done = record.safetyChecklist.filter((s) => s.done).length;

  return (
    <div className="canvas" style={{ padding: 0 }}>
      <div className="grid-2">
        <Panel title="Presenting concerns">
          <div style={{ padding: "var(--space-2) var(--space-5)" }}>
            {record.concerns.map((c) => (
              <div key={c} className="list-row" style={{ paddingLeft: 0, paddingRight: 0 }}>
                <span className="status-dot" style={{ background: "var(--c-primary)" }} />
                <div className="row-main">{c}</div>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="Client snapshot">
          <div className="detail-grid" style={{ padding: "var(--space-4) var(--space-5)" }}>
            <div className="detail">
              <span className="detail-label">Age / gender</span>
              <span className="detail-value">{record.age} / {record.gender}</span>
            </div>
            <div className="detail">
              <span className="detail-label">Framework</span>
              <span className="detail-value" style={{ textTransform: "capitalize" }}>{record.framework}</span>
            </div>
            <div className="detail">
              <span className="detail-label">Clinician</span>
              <span className="detail-value">{record.clinician}</span>
            </div>
            <div className="detail">
              <span className="detail-label">Next appointment</span>
              <span className="detail-value">{record.nextAppointment}</span>
            </div>
          </div>
        </Panel>
      </div>

      <Panel title="Therapeutic goals">
        <div className="checklist">
          {record.goals.map((g) => (
            <div key={g} className="checklist-item">
              <span className="check done">
                <CheckIcon />
              </span>
              <div>
                <div className="checklist-title">{g}</div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid-2">
        <Panel title="Safety summary" actions={<button className="btn btn-sm" onClick={() => openCase(record.id, "safety")}>Open safety</button>}>
          <div style={{ padding: "var(--space-4) var(--space-5)", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            {record.riskFlags.length > 0 ? (
              record.riskFlags.map((f, i) => (
                <Badge key={i} tone={f.level === "high" ? "danger" : f.level === "moderate" ? "warning" : "info"} dot>
                  {f.label}
                </Badge>
              ))
            ) : (
              <Badge tone="success" dot>No active risk flags</Badge>
            )}
            <div>
              <div style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-3)", marginBottom: 4 }}>
                Checklist completion
              </div>
              <Meter value={total ? (done / total) * 100 : 0} />
            </div>
          </div>
        </Panel>
        <Panel title="Next actions">
          <div className="checklist">
            <div className="checklist-item">
              <span className="check" />
              <div>
                <div className="checklist-title">Run evidence search</div>
                <div className="checklist-desc">Refresh literature for the current presenting concern</div>
              </div>
            </div>
            <div className="checklist-item">
              <span className="check" />
              <div>
                <div className="checklist-title">Complete safety sign-off</div>
                <div className="checklist-desc">Finish remaining checklist items before stage gate</div>
              </div>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}

/* ── Assessment ───────────────────────────────────────────────────── */

function AssessmentTab({ caseId }: { caseId: string }) {
  const record = getCase(caseId)!;
  return (
    <div className="canvas" style={{ padding: 0 }}>
      <Alert tone="info" title="This assistant never diagnoses">
        Formulations here are clinical hypotheses for your review — they do not constitute a
        diagnosis and are not a substitute for clinical judgment.
      </Alert>

      <div className="grid-2">
        <Panel title="Intake progress">
          <div style={{ padding: "var(--space-4) var(--space-5)" }}>
            <StepperBlock />
          </div>
        </Panel>
        <Panel title="Clinical history">
          <div style={{ padding: "var(--space-4) var(--space-5)", fontSize: "var(--text-md)", lineHeight: 1.7, color: "var(--c-ink-2)" }}>
            {record.history}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function StepperBlock() {
  const steps = [
    { id: "presenting", title: "Presenting concerns", desc: "Documented symptoms and presentation" },
    { id: "context", title: "History & context", desc: "Onset, trajectory, prior treatment" },
    { id: "risk", title: "Risk assessment", desc: "Safety screening and flags" },
    { id: "strengths", title: "Strengths & resources", desc: "Coping, support system" },
    { id: "goals", title: "Goals", desc: "Collaboratively defined outcomes" },
  ];
  return (
    <div className="stepper">
      {steps.map((s, i) => (
        <div key={s.id} className={`step ${i < 3 ? "done" : i === 3 ? "active" : ""}`}>
          <span className="step-dot">{i < 3 ? <CheckIcon /> : i + 1}</span>
          <div className="step-body">
            <div className="step-title">{s.title}</div>
            <div className="step-desc">{s.desc}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Evidence tab (real retrieval) ────────────────────────────────── */

function CaseEvidenceTab({ caseId }: { caseId: string }) {
  const record = getCase(caseId);
  const framework = record?.framework === "general" ? null : (record?.framework ?? null);
  const defaultQuery = record?.presentingConcern ?? "";

  const [query, setQuery] = useState(defaultQuery);
  const [includeUserDocs, setIncludeUserDocs] = useState(false);
  const [result, setResult] = useState<EvidenceSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const ranOnce = useRef(false);

  const run = useCallback(
    async (q: string, includeDocs: boolean) => {
      if (q.trim().length < 3) return;
      setLoading(true);
      setError(null);
      try {
        const res = await searchEvidence(q, {
          framework,
          includeUserDocuments: includeDocs,
          nResults: 5,
        });
        setResult(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [framework],
  );

  useEffect(() => {
    if (ranOnce.current) return;
    ranOnce.current = true;
    void run(defaultQuery, false);
  }, [run, defaultQuery]);

  const toggleExpand = (chunkId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(chunkId)) next.delete(chunkId);
      else next.add(chunkId);
      return next;
    });
  };

  return (
    <div className="canvas" style={{ padding: 0 }}>
      <Card padded>
        <div className="toolbar">
          <div className="search-box" style={{ maxWidth: "none", flex: 3 }}>
            <SearchIcon />
            <input
              className="input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run(query, includeUserDocs)}
              placeholder="Ask the literature about this case…"
            />
          </div>
          <Button variant="primary" disabled={loading} onClick={() => run(query, includeUserDocs)}>
            {loading ? "Searching…" : "Search literature"}
          </Button>
        </div>
        <div className="toolbar" style={{ marginTop: "var(--space-3)" }}>
          <Badge tone="primary">{record?.framework ?? "general"} knowledge base</Badge>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={includeUserDocs}
              onChange={(e) => setIncludeUserDocs(e.target.checked)}
            />
            Include my documents
          </label>
          <span style={{ marginLeft: "auto", fontSize: "var(--text-xs)", color: "var(--c-ink-3)" }}>
            {result ? `${result.sources.length} sources · ${Math.round(result.elapsed_ms)}ms` : ""}
          </span>
        </div>
      </Card>

      {loading && (
        <Card padded>
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            <Skeleton width="100%" height={16} />
            <Skeleton width="85%" height={16} />
            <Skeleton width="92%" height={16} />
          </div>
        </Card>
      )}

      {error && !loading && <ErrorState title="Evidence search failed" detail={error} />}

      {!loading && !error && result && result.framework && (
        <div className="grid-2">
          <Card padded>
            <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)" }}>
              <Ring value={result.confidence ?? 0} size={64} />
              <div>
                <div className="stat-label">Evidence confidence</div>
                <div style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-3)", marginTop: 2 }}>
                  {result.total_candidates} candidates considered
                </div>
              </div>
            </div>
          </Card>
          <Card padded>
            <div className="synthesis-box" style={{ background: "transparent", border: "none", padding: 0 }}>
              <div className="synthesis-title">
                <BookIcon /> Synthesis
              </div>
              <p>{result.synthesis}</p>
            </div>
          </Card>
        </div>
      )}

      {!loading && !error && result && (result.key_findings.length > 0 || result.evidence_gaps.length > 0) && (
        <div className="grid-2">
          {result.key_findings.length > 0 && (
            <Panel title="Key findings">
              <div className="findings-list" style={{ padding: "var(--space-4) var(--space-5)" }}>
                {result.key_findings.map((f, i) => (
                  <div key={i} className="finding">
                    <span>{f}</span>
                  </div>
                ))}
              </div>
            </Panel>
          )}
          {result.evidence_gaps.length > 0 && (
            <Panel title="Evidence gaps">
              <div style={{ padding: "var(--space-4) var(--space-5)", display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
                {result.evidence_gaps.map((g, i) => (
                  <div key={i} className="gap-item">
                    <span>{g}</span>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>
      )}

      {!loading && !error && result && result.sources.length === 0 && (
        <Card padded>
          <EmptyState icon={<LibraryIcon />} title="No evidence found">
            Try a broader query, switch frameworks, or include your own documents.
          </EmptyState>
        </Card>
      )}

      {!loading &&
        !error &&
        result &&
        result.sources.length > 0 &&
        result.sources.map((item) => (
          <EvidenceCard key={item.chunk_id || item.source + item.page} item={item} expanded={expanded.has(item.chunk_id || item.source + item.page)} onToggle={() => toggleExpand(item.chunk_id || item.source + item.page)} />
        ))}
    </div>
  );
}

export function EvidenceCard({
  item,
  expanded,
  onToggle,
}: {
  item: EvidenceItem;
  expanded: boolean;
  onToggle: () => void;
}) {
  const quality = item.quality_rating ?? 0;
  const tone =
    quality >= 0.75 ? "success" : quality >= 0.6 ? "primary" : quality >= 0.45 ? "warning" : "neutral";
  return (
    <div className={`evidence-card ${expanded ? "expanded" : ""} ${item.is_user_document ? "user" : ""}`}>
      <div className="evidence-card-head">
        <div className="evidence-source">
          <span className="filename">{item.source}</span>
          {item.is_user_document && <Badge tone="warning">My documents</Badge>}
        </div>
        <Badge tone={tone} dot>
          {Math.round(quality * 100)}% match
        </Badge>
      </div>
      <div className="evidence-meta">
        {item.framework && <Tag>{item.framework}</Tag>}
        {item.chapter && <Tag>Ch. {item.chapter}</Tag>}
        {item.topic && <Tag>{item.topic}</Tag>}
        {item.intervention && <Tag>{item.intervention}</Tag>}
        {item.symptom && <Tag>{item.symptom}</Tag>}
        {item.page > 0 && <Tag>p. {item.page}</Tag>}
      </div>
      <div className="evidence-content">{item.content}</div>
      <div className="evidence-footer">
        <div className="meter" style={{ flex: 1 }}>
          <div className="meter-track">
            <div className="meter-fill" style={{ width: `${Math.round(quality * 100)}%` }} />
          </div>
        </div>
        <button className="btn btn-sm btn-ghost" onClick={onToggle}>
          {expanded ? "Collapse" : "Expand"}
        </button>
        <button className="btn btn-sm btn-ghost" title="Copy citation">
          <ExternalLinkIcon />
        </button>
      </div>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return <span className="tag">{children}</span>;
}

/* ── Formulation ──────────────────────────────────────────────────── */

function FormulationTab() {
  return (
    <div className="canvas" style={{ padding: 0 }}>
      <Alert tone="info" title="Evidence-anchored formulation">
        Each hypothesis below links back to retrieved evidence. Ratings reflect the quality of the
        supporting literature, not the certainty of the hypothesis itself.
      </Alert>
      <div className="grid-2">
        <Panel title="Maintaining factors" actions={<Badge tone="info">hypothesis</Badge>}>
          <div style={{ padding: "var(--space-4) var(--space-5)", display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
            {[
              { label: "Rumination maintains worry cycle", score: 0.82 },
              { label: "Sleep disruption amplifies daytime anxiety", score: 0.74 },
              { label: "Avoidance of sleep onset fuels arousal", score: 0.61 },
            ].map((h) => (
              <div key={h.label}>
                <div style={{ fontSize: "var(--text-md)", fontWeight: 550, marginBottom: 4 }}>{h.label}</div>
                <Meter value={h.score * 100} label={`Evidence support · ${h.score.toFixed(2)}`} />
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="Protective factors">
          <div className="checklist">
            {["Strong engagement and motivation", "Supportive family network", "No substance use"].map((p) => (
              <div key={p} className="checklist-item">
                <span className="check done">
                  <CheckIcon />
                </span>
                <div className="checklist-title">{p}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

/* ── Plan ─────────────────────────────────────────────────────────── */

function PlanTab() {
  return (
    <div className="canvas" style={{ padding: 0 }}>
      <div className="form-grid">
        <Panel title="Objectives" actions={<Badge tone="primary">3 active</Badge>}>
          <div className="checklist">
            {["Reduce nightly rumination to ≤2 nights/week by 6 weeks", "Master cognitive restructuring for work thoughts", "Stabilize sleep-wake schedule"].map((o) => (
              <div key={o} className="checklist-item">
                <span className="check done">
                  <CheckIcon />
                </span>
                <div>
                  <div className="checklist-title">{o}</div>
                  <div className="checklist-desc">Measured via worry log + sleep diary</div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="Interventions">
          <div className="checklist">
            {["Cognitive restructuring (weekly)", "Worry-time scheduling (daily)", "Stimulus control for sleep (daily)"].map((o) => (
              <div key={o} className="checklist-item">
                <span className="check" />
                <div>
                  <div className="checklist-title">{o}</div>
                  <div className="checklist-desc">Cadence: weekly session + daily practice</div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
      <Panel title="Outcome monitoring">
        <div className="detail-grid" style={{ padding: "var(--space-4) var(--space-5)" }}>
          <div className="detail">
            <span className="detail-label">Measure</span>
            <span className="detail-value">GAD-7</span>
          </div>
          <div className="detail">
            <span className="detail-label">Baseline</span>
            <span className="detail-value">14 (moderate)</span>
          </div>
          <div className="detail">
            <span className="detail-label">Review cadence</span>
            <span className="detail-value">Every 2 weeks</span>
          </div>
          <div className="detail">
            <span className="detail-label">Target</span>
            <span className="detail-value">≤ 8</span>
          </div>
        </div>
      </Panel>
    </div>
  );
}

/* ── Safety ───────────────────────────────────────────────────────── */

function SafetyTab({ caseId }: { caseId: string }) {
  const record = getCase(caseId)!;
  const done = record.safetyChecklist.filter((s) => s.done).length;
  const total = record.safetyChecklist.length;

  return (
    <div className="canvas" style={{ padding: 0 }}>
      <Card padded>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-5)", flexWrap: "wrap" }}>
          <Ring value={total === 0 ? 0 : done / total} size={72} caption="sign-off" />
          <div style={{ flex: 1, minWidth: 220 }}>
            <div style={{ fontSize: "var(--text-md)", fontWeight: 600, marginBottom: "var(--space-2)" }}>
              {done} of {total} safety items completed
            </div>
            <Meter value={total === 0 ? 0 : (done / total) * 100} label="Stage gate readiness" />
          </div>
          <Button variant={done === total ? "primary" : "default"} disabled={done < total} title={done < total ? "Complete all items to unlock sign-off" : undefined}>
            <ShieldIcon /> {done === total ? "Sign off case" : "Locked until complete"}
          </Button>
        </div>
      </Card>

      {done < total && (
        <Alert tone="warning" title="Stage gate locked">
          The Safety stage requires a full sign-off before the case can progress. Review the items below.
        </Alert>
      )}

      <Panel title="Safety checklist">
        <div className="checklist">
          {record.safetyChecklist.map((item) => (
            <div key={item.id} className="checklist-item">
              <span className={`check ${item.done ? "done" : ""}`}>{item.done && <CheckIcon />}</span>
              <div>
                <div className="checklist-title">{item.label}</div>
                <div className="checklist-desc">{item.detail}</div>
              </div>
              <span style={{ marginLeft: "auto" }}>
                <Badge tone={item.done ? "success" : "neutral"} dot>
                  {item.done ? "Complete" : "Pending"}
                </Badge>
              </span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Safety notes">
        <div style={{ padding: "var(--space-4) var(--space-5)" }}>
          {record.safetyNotes.map((n, i) => (
            <div key={i} className="list-row" style={{ paddingLeft: 0, paddingRight: 0 }}>
              <AlertTriangleIcon style={{ width: 15, height: 15, color: "var(--c-warning)", flexShrink: 0 }} />
              <div className="row-main">{n}</div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
