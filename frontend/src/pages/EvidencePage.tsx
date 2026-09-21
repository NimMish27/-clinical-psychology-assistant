import { useState } from "react";
import { FRAMEWORKS, searchEvidence, type EvidenceSearchResponse } from "../api/client";
import { Badge, Button, Card, EmptyState, ErrorState, Panel, Ring, Skeleton } from "../ui/primitives";
import { EvidenceCard } from "./CaseWorkspacePage";
import { LibraryIcon, SearchIcon, SparklesIcon } from "../ui/icons";

export default function EvidencePage() {
  const [query, setQuery] = useState("");
  const [framework, setFramework] = useState<string>("general");
  const [includeUserDocs, setIncludeUserDocs] = useState(false);
  const [result, setResult] = useState<EvidenceSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const run = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (query.trim().length < 3) return;
    setLoading(true);
    setError(null);
    setHasSearched(true);
    try {
      const res = await searchEvidence(query, {
        framework: framework === "general" ? null : (framework as never),
        includeUserDocuments: includeUserDocs,
        nResults: 6,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div className="canvas canvas-narrow">
      <div className="section-head">
        <div>
          <div className="kicker">Evidence</div>
          <h1 className="section-title">Evidence Explorer</h1>
          <p className="section-sub">
            Ask the literature. Retrieval-only — results are structured evidence cards, not a chat transcript.
          </p>
        </div>
      </div>

      <Card padded>
        <form onSubmit={run}>
          <div className="search-box" style={{ maxWidth: "none" }}>
            <SearchIcon />
            <input
              className="input"
              style={{ height: 44, fontSize: "var(--text-lg)" }}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search clinical literature… e.g. exposure therapy for panic with agoraphobia"
            />
          </div>
          <div className="toolbar" style={{ marginTop: "var(--space-4)" }}>
            <select
              className="select"
              value={framework}
              onChange={(e) => setFramework(e.target.value)}
              aria-label="Knowledge base"
            >
              <option value="general">General clinical knowledge base</option>
              {FRAMEWORKS.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label} literature
                </option>
              ))}
            </select>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={includeUserDocs}
                onChange={(e) => setIncludeUserDocs(e.target.checked)}
              />
              Include my documents
            </label>
            <span style={{ flex: 1 }} />
            <Button type="submit" variant="primary" size="lg" disabled={loading}>
              <SparklesIcon /> {loading ? "Searching…" : "Search"}
            </Button>
          </div>
        </form>
      </Card>

      {loading && (
        <Card padded>
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            <Skeleton width="100%" height={16} />
            <Skeleton width="80%" height={16} />
            <Skeleton width="90%" height={16} />
          </div>
        </Card>
      )}

      {error && !loading && <ErrorState title="Evidence search failed" detail={error} />}

      {!loading && !error && result && result.framework && (
        <Card padded>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-5)", flexWrap: "wrap" }}>
            <Ring value={result.confidence ?? 0} size={72} />
            <div style={{ flex: 1, minWidth: 200 }}>
              <div className="stat-label">Evidence confidence</div>
              <p style={{ fontSize: "var(--text-sm)", color: "var(--c-ink-3)", marginTop: 4 }}>
                {result.total_candidates} candidates considered in {Math.round(result.elapsed_ms)}ms
              </p>
            </div>
            {result.user_documents_returned > 0 && (
              <Badge tone="warning">{result.user_documents_returned} from my documents</Badge>
            )}
          </div>
        </Card>
      )}

      {!loading && !error && result && result.synthesis && (
        <div className="synthesis-box">
          <div className="synthesis-title">Synthesis</div>
          <p>{result.synthesis}</p>
        </div>
      )}

      {!loading && !error && result && result.key_findings.length > 0 && (
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

      {!loading && !error && result && result.evidence_gaps.length > 0 && (
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

      {!loading && !error && hasSearched && result && result.sources.length === 0 && (
        <Card padded>
          <EmptyState icon={<LibraryIcon />} title="No evidence found">
            Broaden the query, switch knowledge base, or include your own documents.
          </EmptyState>
        </Card>
      )}

      {!loading &&
        !error &&
        result &&
        result.sources.map((item) => {
          const key = item.chunk_id || item.source + item.page;
          return (
            <EvidenceCard
              key={key}
              item={item}
              expanded={expanded.has(key)}
              onToggle={() => toggleExpand(key)}
            />
          );
        })}
    </div>
  );
}
