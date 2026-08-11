import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  FRAMEWORKS,
  searchEvidence,
  type EvidenceItem,
  type EvidenceSearchResponse,
  type Framework,
} from "../api/client";
import { analyzeFrameworks, bestFit } from "../lib/analysis";
import { getCase } from "../lib/sampleData";
import {
  Alert,
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  Meter,
  Panel,
  Skeleton,
  Tag,
} from "../ui/primitives";
import {
  CheckIcon,
  LibraryIcon,
  RefreshIcon,
} from "../ui/icons";

interface FwEntry {
  loading: boolean;
  result: EvidenceSearchResponse | null;
  error: string | null;
}

const EMPTY: Record<Framework, FwEntry> = {
  cbt: { loading: false, result: null, error: null },
  act: { loading: false, result: null, error: null },
  humanistic: { loading: false, result: null, error: null },
};

function entryConfidence(e: FwEntry): number | null {
  if (!e.result) return null;
  if (e.result.confidence != null) return e.result.confidence;
  const qs = e.result.sources.map((s) => s.quality_rating ?? 0);
  return qs.length ? qs.reduce((a, b) => a + b, 0) / qs.length : null;
}

export default function CaseAnalysisPage({ caseId }: { caseId: string }) {
  const record = getCase(caseId);
  const [entries, setEntries] = useState<Record<Framework, FwEntry>>(EMPTY);
  const [lastRun, setLastRun] = useState<string | null>(null);
  const ranOnce = useRef(false);

  const query = record?.presentingConcern ?? "";

  const run = useCallback(async () => {
    if (query.trim().length < 3) return;
    setLastRun(null);
    setEntries((prev) => {
      const next = { ...prev };
      for (const f of FRAMEWORKS) next[f.id] = { ...next[f.id], loading: true, error: null };
      return next;
    });
    const settled = await Promise.allSettled(
      FRAMEWORKS.map((f) =>
        searchEvidence(query, { framework: f.id, nResults: 4 }),
      ),
    );
    setEntries((prev) => {
      const next = { ...prev };
      FRAMEWORKS.forEach((f, i) => {
        const s = settled[i];
        next[f.id] =
          s.status === "fulfilled"
            ? { loading: false, result: s.value, error: null }
            : {
                loading: false,
                result: null,
                error: s.reason instanceof Error ? s.reason.message : String(s.reason),
              };
      });
      return next;
    });
    setLastRun(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
  }, [query]);

  useEffect(() => {
    if (ranOnce.current) return;
    ranOnce.current = true;
    void run();
  }, [run]);

  const running = FRAMEWORKS.some((f) => entries[f.id].loading);

  const allSources = useMemo(() => {
    const seen = new Set<string>();
    const list: EvidenceItem[] = [];
    for (const f of FRAMEWORKS) {
      const res = entries[f.id].result;
      if (!res) continue;
      for (const it of res.sources) {
        const key = it.chunk_id || `${it.source}:${it.page}`;
        if (seen.has(key)) continue;
        seen.add(key);
        list.push(it);
      }
    }
    return list.sort((a, b) => (b.quality_rating ?? 0) - (a.quality_rating ?? 0));
  }, [entries]);

  if (!record) {
    return (
      <div className="canvas">
        <ErrorState title="Case not found" detail={`No case exists with ID ${caseId}.`} />
      </div>
    );
  }

  const assessments = analyzeFrameworks(record);
  const best = bestFit(assessments);

  return (
    <div className="canvas" style={{ padding: 0 }}>
      <Card padded>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "var(--space-4)", flexWrap: "wrap" }}>
          <div>
            <div className="eyebrow">Case Analysis</div>
            <div style={{ fontSize: "var(--text-md)", fontWeight: 650, marginTop: 2 }}>
              Framework comparison for {record.name}
            </div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--c-ink-3)", marginTop: 2 }}>
              {lastRun ? `Updated ${lastRun}` : "Running framework evidence searches…"}
            </div>
          </div>
          <Button disabled={running} onClick={() => void run()}>
            <RefreshIcon /> {running ? "Analyzing…" : "Refresh analysis"}
          </Button>
        </div>
      </Card>

      <Alert tone="info" title="This assistant never diagnoses">
        The suitability scores below reflect how the retrieved literature maps onto this case —
        they are clinical hypotheses for your review, not a diagnosis.
      </Alert>

      {/* Left: case summary · Right: framework cards */}
      <div className="analysis-grid">
        <div className="analysis-col">
          <Panel title="Case Summary">
            <div style={{ padding: "var(--space-4) var(--space-5)", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
                <span className="avatar" style={{ background: record.color, width: 40, height: 40 }}>
                  {record.initials}
                </span>
                <div>
                  <div style={{ fontSize: "var(--text-md)", fontWeight: 650 }}>{record.name}</div>
                  <div style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-3)" }}>
                    {record.age} · {record.gender} · {record.clinician}
                  </div>
                </div>
              </div>
              <div>
                <div className="analysis-sub">Presenting concern</div>
                <div style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-2)", marginTop: 2 }}>
                  {record.presentingConcern}
                </div>
              </div>
              <div>
                <div className="analysis-sub">History</div>
                <div style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-2)", lineHeight: 1.6, marginTop: 2 }}>
                  {record.history}
                </div>
              </div>
              <div className="detail-grid">
                <div className="detail">
                  <span className="detail-label">Status</span>
                  <span className="detail-value" style={{ textTransform: "capitalize" }}>{record.status}</span>
                </div>
                <div className="detail">
                  <span className="detail-label">Stage</span>
                  <span className="detail-value" style={{ textTransform: "capitalize" }}>{record.stage}</span>
                </div>
                <div className="detail">
                  <span className="detail-label">Framework</span>
                  <span className="detail-value" style={{ textTransform: "capitalize" }}>{record.framework}</span>
                </div>
                <div className="detail">
                  <span className="detail-label">Next session</span>
                  <span className="detail-value">{record.nextAppointment}</span>
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="Symptoms">
            <div style={{ padding: "var(--space-2) var(--space-5)" }}>
              {record.symptoms.map((s) => (
                <div key={s} className="list-row" style={{ paddingLeft: 0, paddingRight: 0 }}>
                  <span className="status-dot" style={{ background: "var(--c-primary)" }} />
                  <div className="row-main">{s}</div>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Stressors">
            <div style={{ padding: "var(--space-2) var(--space-5)" }}>
              {record.stressors.map((s) => (
                <div key={s} className="list-row" style={{ paddingLeft: 0, paddingRight: 0 }}>
                  <span className="status-dot" style={{ background: "var(--c-warning)" }} />
                  <div className="row-main">{s}</div>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Protective Factors">
            <div style={{ padding: "var(--space-2) var(--space-5)" }}>
              {record.protectiveFactors.map((s) => (
                <div key={s} className="list-row" style={{ paddingLeft: 0, paddingRight: 0 }}>
                  <span className="status-dot" style={{ background: "var(--c-success)" }} />
                  <div className="row-main">{s}</div>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        {/* Framework cards */}
        <div className="analysis-col">
          {FRAMEWORKS.map((fw) => (
            <FrameworkCard key={fw.id} assessment={assessments[fw.id]} entry={entries[fw.id]} />
          ))}
        </div>
      </div>

      {/* Bottom: evidence preview + confidence scores */}
      <div className="analysis-preview-row">
        <Panel title="Evidence Preview" actions={<Badge tone="primary" dot>top sources</Badge>}>
          <div style={{ padding: "var(--space-4) var(--space-5)", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            {running && allSources.length === 0 && (
              <>
                <Skeleton width="100%" height={64} />
                <Skeleton width="100%" height={64} />
                <Skeleton width="100%" height={64} />
              </>
            )}
            {!running && allSources.length === 0 && (
              <EmptyState icon={<LibraryIcon />} title="No evidence retrieved">
                Run a refresh once a framework knowledge base is indexed.
              </EmptyState>
            )}
            {allSources.slice(0, 3).map((it) => (
              <PreviewCard key={it.chunk_id || `${it.source}:${it.page}`} item={it} />
            ))}
          </div>
        </Panel>

        <Panel title="Confidence Scores" actions={<Badge tone="info">KB retrieval</Badge>}>
          <div style={{ padding: "var(--space-4) var(--space-5)", display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
            {FRAMEWORKS.map((fw) => {
              const e = entries[fw.id];
              const conf = entryConfidence(e);
              return (
                <div key={fw.id}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: "var(--text-sm)", fontWeight: 600, textTransform: "capitalize" }}>
                      {fw.id}
                    </span>
                    {e.loading ? (
                      <span className="analysis-sub">searching…</span>
                    ) : e.error ? (
                      <Badge tone="neutral">not indexed</Badge>
                    ) : conf != null ? (
                      <span className="analysis-sub">{Math.round(conf * 100)}%</span>
                    ) : (
                      <span className="analysis-sub">—</span>
                    )}
                  </div>
                  {e.loading ? (
                    <Skeleton width="100%" height={8} />
                  ) : e.error ? (
                    <div style={{ fontSize: "var(--text-xs)", color: "var(--c-ink-3)" }}>{e.error}</div>
                  ) : conf != null ? (
                    <Meter value={conf * 100} />
                  ) : (
                    <Meter value={0} />
                  )}
                </div>
              );
            })}
            <div style={{ borderTop: "1px solid var(--c-hairline)", paddingTop: "var(--space-3)" }}>
              <div style={{ fontSize: "var(--text-sm)", fontWeight: 600 }}>
                Best fit: <span style={{ textTransform: "capitalize" }}>{best.framework}</span>
              </div>
              <div style={{ fontSize: "var(--text-xs)", color: "var(--c-ink-3)", marginTop: 2 }}>
                Suitability {best.score}/100 · {best.verdict}
              </div>
            </div>
          </div>
        </Panel>
      </div>

      {/* Bottom: supporting literature */}
      <Panel title="Supporting Literature" actions={<Badge tone="neutral">{allSources.length} sources</Badge>}>
        {allSources.length === 0 && !running ? (
          <div style={{ padding: "var(--space-4) var(--space-5)" }}>
            <EmptyState icon={<LibraryIcon />} title="No literature yet">
              Retrieved sources from each framework search will appear here.
            </EmptyState>
          </div>
        ) : (
          allSources.map((it) => <LiteratureRow key={it.chunk_id || `${it.source}:${it.page}`} item={it} />)
        )}
      </Panel>
    </div>
  );
}

/* ── Framework card ───────────────────────────────────────────────── */

const VERDICT_TONE: Record<string, "success" | "info" | "neutral"> = {
  recommended: "success",
  consider: "info",
  "not-indicated": "neutral",
};

function FrameworkCard({
  assessment,
  entry,
}: {
  assessment: ReturnType<typeof analyzeFrameworks>[Framework];
  entry: FwEntry;
}) {
  const conf = entryConfidence(entry);
  return (
    <Panel
      title={
        <span className="analysis-fw-name">
          <span className="analysis-fw-dot" style={{ background: CONFIG_COLOR[assessment.framework] }} />
          {assessment.framework}
        </span>
      }
      actions={<Badge tone={VERDICT_TONE[assessment.verdict]} dot>{assessment.verdict}</Badge>}
    >
      <div className="analysis-fw-card">
        <div className="analysis-lead">
          <span className="analysis-score">{assessment.score}</span>
          <span className="analysis-sub">/ 100 suitability</span>
        </div>

        <div>
          <div className="analysis-sub" style={{ marginBottom: 4 }}>Evidence support · KB retrieval</div>
          {entry.loading ? (
            <Skeleton width="100%" height={8} />
          ) : entry.error ? (
            <div style={{ fontSize: "var(--text-xs)", color: "var(--c-ink-3)" }}>
              Knowledge base not indexed for this framework.
            </div>
          ) : conf != null ? (
            <Meter value={conf * 100} label={`confidence · ${Math.round(conf * 100)}%`} />
          ) : (
            <Meter value={0} />
          )}
        </div>

        {assessment.matched.length > 0 && (
          <div>
            <div className="analysis-sub" style={{ marginBottom: 4 }}>Alignment</div>
            <div className="analysis-matches">
              {assessment.matched.map((m) => (
                <Tag key={m}>{m}</Tag>
              ))}
            </div>
          </div>
        )}

        <div>
          <div className="analysis-sub" style={{ marginBottom: 4 }}>Suggested interventions</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {assessment.interventions.map((i) => (
              <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                <CheckIcon style={{ width: 14, height: 14, color: "var(--c-success)", flexShrink: 0, marginTop: 2 }} />
                <span style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-2)" }}>{i}</span>
              </div>
            ))}
          </div>
        </div>

        <ul className="analysis-reasons">
          {assessment.reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      </div>
    </Panel>
  );
}

const CONFIG_COLOR: Record<Framework, string> = {
  cbt: "#6366f1",
  act: "#0ea5e9",
  humanistic: "#10b981",
};

/* ── Evidence preview card ────────────────────────────────────────── */

function PreviewCard({ item }: { item: EvidenceItem }) {
  const quality = Math.round((item.quality_rating ?? 0) * 100);
  return (
    <div className="evidence-card">
      <div className="evidence-card-head">
        <div className="evidence-source">
          <span className="filename">{item.source}</span>
          {item.framework && <Tag>{item.framework}</Tag>}
        </div>
        <Badge tone={quality >= 75 ? "success" : quality >= 60 ? "primary" : "warning"} dot>
          {quality}% match
        </Badge>
      </div>
      <div className="evidence-meta">
        {item.chapter && <Tag>Ch. {item.chapter}</Tag>}
        {item.topic && <Tag>{item.topic}</Tag>}
        {item.page > 0 && <Tag>p. {item.page}</Tag>}
      </div>
      <div className="evidence-content">
        {item.content.length > 180 ? `${item.content.slice(0, 180)}…` : item.content}
      </div>
    </div>
  );
}

/* ── Supporting literature row ────────────────────────────────────── */

function LiteratureRow({ item }: { item: EvidenceItem }) {
  const [open, setOpen] = useState(false);
  const quality = Math.round((item.quality_rating ?? 0) * 100);
  return (
    <div className="list-row" style={{ alignItems: "flex-start", flexDirection: "column", gap: "var(--space-2)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", width: "100%" }}>
        <span className="status-dot" style={{ background: item.is_user_document ? "var(--c-warning)" : "var(--c-primary)" }} />
        <div className="row-main">
          <div className="row-title" style={{ fontSize: "var(--text-sm)" }}>{item.source}</div>
          <div className="row-sub" style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center", marginTop: 3 }}>
            {item.is_user_document && <Tag>My documents</Tag>}
            {item.framework && <Tag>{item.framework}</Tag>}
            {item.chapter && <Tag>Ch. {item.chapter}</Tag>}
            {item.topic && <Tag>{item.topic}</Tag>}
            {item.page > 0 && <Tag>p. {item.page}</Tag>}
          </div>
        </div>
        <Badge tone={quality >= 75 ? "success" : quality >= 60 ? "primary" : "warning"} dot>
          {quality}% match
        </Badge>
        <button className="btn btn-sm btn-ghost" onClick={() => setOpen((o) => !o)}>
          {open ? "Collapse" : "Expand"}
        </button>
      </div>
      {open && (
        <div style={{ paddingLeft: 21, fontSize: "var(--text-sm)", color: "var(--c-ink-2)", lineHeight: 1.65 }}>
          {item.content}
        </div>
      )}
    </div>
  );
}
