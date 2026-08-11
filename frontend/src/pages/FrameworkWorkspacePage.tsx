import { useCallback, useEffect, useRef, useState } from "react";
import { searchEvidence, type EvidenceItem, type EvidenceSearchResponse } from "../api/client";
import { FRAMEWORK_CARDS, FRAMEWORK_ID_TO_API } from "../lib/frameworks";
import { useNav } from "../lib/nav";
import { FrameworkGlyph } from "../ui/FrameworkCard";
import {
  AlertTriangleIcon,
  ArrowLeftIcon,
  BookIcon,
  CheckIcon,
  ChevronDownIcon,
  ExternalLinkIcon,
  LayersIcon,
  RefreshIcon,
  SearchIcon,
  ShieldIcon,
  SparklesIcon,
} from "../ui/icons";

type TabId = "conceptualization" | "evidence" | "plan" | "assessment";

const TABS: Array<{ id: TabId; label: string }> = [
  { id: "conceptualization", label: "Conceptualization" },
  { id: "evidence", label: "Evidence" },
  { id: "plan", label: "Plan" },
  { id: "assessment", label: "Assessment" },
];

export default function FrameworkWorkspacePage({ frameworkId }: { frameworkId: string }) {
  const { openSection } = useNav();
  const data = FRAMEWORK_CARDS.find((c) => c.id === frameworkId);

  const [tab, setTab] = useState<TabId>("conceptualization");

  const apiFramework = data ? FRAMEWORK_ID_TO_API[data.id] : null;
  const defaultQuery = data
    ? `Effectiveness and core mechanisms of ${data.name}${data.tagline && data.tagline !== data.name ? ` (${data.tagline})` : ""}`
    : "";

  const [evQuery, setEvQuery] = useState(defaultQuery);
  const [evResult, setEvResult] = useState<EvidenceSearchResponse | null>(null);
  const [evLoading, setEvLoading] = useState(false);
  const [evError, setEvError] = useState<string | null>(null);
  const ranOnce = useRef(false);

  const run = useCallback(async () => {
    if (evQuery.trim().length < 3) return;
    setEvLoading(true);
    setEvError(null);
    try {
      const res = await searchEvidence(evQuery, { framework: apiFramework, nResults: 5 });
      setEvResult(res);
    } catch (e) {
      setEvError(e instanceof Error ? e.message : String(e));
    } finally {
      setEvLoading(false);
    }
  }, [evQuery, apiFramework]);

  useEffect(() => {
    if (ranOnce.current) return;
    ranOnce.current = true;
    void run();
  }, [run]);

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-24 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-stone-100 text-stone-400">
          <LayersIcon className="h-6 w-6" />
        </div>
        <h1 className="text-lg font-semibold text-stone-800">Framework not found</h1>
        <p className="text-sm text-stone-500">No workspace exists for "{frameworkId}".</p>
        <button
          onClick={() => openSection("frameworks")}
          className="mt-2 inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700"
        >
          <ArrowLeftIcon className="h-4 w-4" /> Back to framework cards
        </button>
      </div>
    );
  }

  const w = data.workspace;

  return (
    <div className="relative">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div
          className="absolute -right-16 -top-20 h-72 w-72 rounded-full opacity-20 blur-3xl"
          style={{ background: data.color }}
        />
        <div className="absolute left-1/3 top-1/2 h-64 w-64 rounded-full bg-indigo-100/60 blur-3xl" />
      </div>

      <div className="relative">
        <button
          onClick={() => openSection("frameworks")}
          className="inline-flex items-center gap-1.5 text-sm font-medium text-stone-500 transition hover:text-stone-800"
        >
          <ArrowLeftIcon className="h-4 w-4" /> Framework cards
        </button>

        {/* Header */}
        <header className="mt-4 overflow-hidden rounded-2xl border border-white/70 bg-white/60 p-5 shadow-[0_1px_2px_rgba(15,23,42,0.05),0_10px_30px_-12px_rgba(15,23,42,0.12)] backdrop-blur-xl sm:p-6">
          <div className="flex flex-wrap items-center gap-4">
            <span
              className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl text-white shadow-lg"
              style={{
                background: `linear-gradient(135deg, ${data.color}, ${data.color}cc)`,
                boxShadow: `0 10px 24px -10px ${data.color}b3`,
              }}
            >
              <FrameworkGlyph id={data.id} className="h-7 w-7" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-stone-900">{data.name}</h1>
                {data.featured && (
                  <span
                    className="rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-white"
                    style={{ background: data.color }}
                  >
                    Core
                  </span>
                )}
              </div>
              <p className="text-sm text-stone-400">{data.tagline}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <HeaderChip label="Confidence" value={`${Math.round(data.confidence * 100)}%`} color={data.color} />
              <HeaderChip label="Sources" value={`${data.evidenceCount}`} />
              <HeaderChip
                label="KB"
                value={apiFramework ? `${apiFramework} indexed` : "general KB"}
              />
            </div>
          </div>
        </header>

        {/* Tabs */}
        <nav className="sticky top-0 z-20 mt-5 flex gap-1 rounded-xl border border-white/70 bg-white/70 p-1 shadow-sm backdrop-blur-xl">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 ${
                tab === t.id
                  ? "bg-white text-stone-900 shadow-sm ring-1 ring-stone-200/70"
                  : "text-stone-500 hover:bg-white/60 hover:text-stone-700"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {/* Tab content */}
        {tab === "conceptualization" && (
          <div className="mt-5 flex flex-col gap-5">
            <SectionCard title="Case Conceptualization" icon={<BookIcon className="h-4 w-4" />}>
              <p className="text-sm leading-relaxed text-stone-600">{w.conceptualization}</p>
              <div className="mt-4 rounded-lg bg-white/60 p-3 text-[11px] font-semibold uppercase tracking-[0.08em] text-stone-400 ring-1 ring-stone-200/60">
                {w.model.label}
              </div>
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                {w.model.points.map((p, i) => (
                  <div key={i} className="rounded-xl bg-white/70 p-3.5 ring-1 ring-stone-200/60">
                    <span
                      className="mb-2 flex h-6 w-6 items-center justify-center rounded-md text-[11px] font-bold text-white"
                      style={{ background: data.color }}
                    >
                      {i + 1}
                    </span>
                    <p className="text-xs leading-relaxed text-stone-600">{p}</p>
                  </div>
                ))}
              </div>
            </SectionCard>

            <SectionCard title="Therapeutic Focus" icon={<SparklesIcon className="h-4 w-4" />}>
              <p className="text-sm leading-relaxed text-stone-600">{w.therapeuticFocus}</p>
            </SectionCard>

            <div className="grid gap-5 lg:grid-cols-2">
              <SectionCard title="Strengths" icon={<CheckIcon className="h-4 w-4" />}>
                <ul className="flex flex-col gap-2.5">
                  {w.strengths.map((s) => (
                    <li key={s} className="flex items-start gap-2.5">
                      <CheckIcon className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
                      <span className="text-sm leading-relaxed text-stone-600">{s}</span>
                    </li>
                  ))}
                </ul>
              </SectionCard>
              <SectionCard title="Limitations" icon={<AlertTriangleIcon className="h-4 w-4" />}>
                <ul className="flex flex-col gap-2.5">
                  {w.limitations.map((s) => (
                    <li key={s} className="flex items-start gap-2.5">
                      <AlertTriangleIcon className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
                      <span className="text-sm leading-relaxed text-stone-600">{s}</span>
                    </li>
                  ))}
                </ul>
              </SectionCard>
            </div>
          </div>
        )}

        {tab === "evidence" && <EvidenceTab dataName={data.name} query={evQuery} setQuery={setEvQuery} onRun={() => void run()} loading={evLoading} error={evError} result={evResult} color={data.color} kbLabel={apiFramework ? `${apiFramework} knowledge base` : "general clinical knowledge base"} />}

        {tab === "plan" && (
          <div className="mt-5 flex flex-col gap-5">
            <SectionCard title="Suggested Interventions" icon={<SparklesIcon className="h-4 w-4" />}>
              <div className="grid gap-3 sm:grid-cols-3">
                {w.suggestedInterventions.map((iv) => (
                  <div key={iv.name} className="rounded-xl bg-white/70 p-4 ring-1 ring-stone-200/60 transition-colors hover:bg-white">
                    <div className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full" style={{ background: data.color }} />
                      <div className="text-sm font-semibold text-stone-800">{iv.name}</div>
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-stone-500">{iv.desc}</p>
                  </div>
                ))}
              </div>
            </SectionCard>

            <SectionCard title="Session Planning" icon={<LayersIcon className="h-4 w-4" />}>
              <ol className="flex flex-col">
                {w.sessionPlan.map((ph, i) => (
                  <li key={ph.phase} className="relative flex gap-4 pb-5 last:pb-0">
                    {i < w.sessionPlan.length - 1 && (
                      <span className="absolute left-[9px] top-6 h-full w-px bg-stone-200" />
                    )}
                    <span
                      className="relative mt-1 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white"
                      style={{ background: data.color }}
                    >
                      {i + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-semibold text-stone-800">{ph.phase}</span>
                        <span className="rounded-full bg-stone-100 px-2 py-0.5 text-[10px] font-medium text-stone-500">
                          {ph.weeks}
                        </span>
                      </div>
                      <p className="mt-0.5 text-xs text-stone-400">{ph.focus}</p>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {ph.items.map((it) => (
                          <span
                            key={it}
                            className="rounded-md bg-white/80 px-2 py-1 text-[11px] font-medium text-stone-600 ring-1 ring-stone-200/60"
                          >
                            {it}
                          </span>
                        ))}
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
            </SectionCard>

            <SectionCard title="Homework" icon={<CheckIcon className="h-4 w-4" />}>
              <div className="grid gap-3 sm:grid-cols-3">
                {w.homework.map((h) => (
                  <div key={h.name} className="rounded-xl bg-white/70 p-4 ring-1 ring-stone-200/60">
                    <div className="flex items-start justify-between gap-2">
                      <div className="text-sm font-semibold text-stone-800">{h.name}</div>
                      <span className="shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold text-indigo-600 ring-1 ring-indigo-100">
                        {h.cadence}
                      </span>
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-stone-500">{h.desc}</p>
                  </div>
                ))}
              </div>
            </SectionCard>
          </div>
        )}

        {tab === "assessment" && (
          <div className="mt-5 flex flex-col gap-5">
            <SectionCard title="Assessment Questions" icon={<ShieldIcon className="h-4 w-4" />}>
              <p className="mb-4 text-xs text-stone-400">
                Collaborative prompts to open dialogue and refine the formulation under this framework.
              </p>
              <ol className="flex flex-col gap-2.5">
                {w.assessmentQuestions.map((q, i) => (
                  <li key={q} className="flex items-start gap-3 rounded-xl bg-white/70 px-4 py-3 ring-1 ring-stone-200/60">
                    <span
                      className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[11px] font-bold text-white"
                      style={{ background: data.color }}
                    >
                      {i + 1}
                    </span>
                    <span className="text-sm leading-relaxed text-stone-700">"{q}"</span>
                  </li>
                ))}
              </ol>
            </SectionCard>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Small building blocks ────────────────────────────────────────── */

function HeaderChip({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center gap-2 rounded-xl bg-white/70 px-3 py-2 ring-1 ring-stone-200/60">
      <span className="text-[11px] text-stone-400">{label}</span>
      <span className="text-xs font-bold text-stone-700" style={color ? { color } : undefined}>
        {value}
      </span>
    </div>
  );
}

function SectionCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-white/70 bg-white/60 p-5 shadow-[0_1px_2px_rgba(15,23,42,0.04),0_8px_24px_-12px_rgba(15,23,42,0.1)] backdrop-blur-xl">
      <h2 className="mb-3.5 flex items-center gap-2 text-sm font-semibold text-stone-800">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
          {icon}
        </span>
        {title}
      </h2>
      {children}
    </section>
  );
}

/* ── Evidence tab ─────────────────────────────────────────────────── */

function EvidenceTab({
  dataName,
  query,
  setQuery,
  onRun,
  loading,
  error,
  result,
  color,
  kbLabel,
}: {
  dataName: string;
  query: string;
  setQuery: (q: string) => void;
  onRun: () => void;
  loading: boolean;
  error: string | null;
  result: EvidenceSearchResponse | null;
  color: string;
  kbLabel: string;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = (key: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const conf = result?.confidence ?? 0;
  const pct = Math.round(conf * 100);
  const sources = result?.sources ?? [];

  return (
    <div className="mt-5 flex flex-col gap-5">
      <SectionCard title="Evidence" icon={<SearchIcon className="h-4 w-4" />}>
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="flex flex-1 items-center gap-2 rounded-xl bg-white/80 px-3.5 py-2.5 ring-1 ring-stone-200/70 focus-within:ring-2">
            <SearchIcon className="h-4 w-4 shrink-0 text-stone-400" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onRun()}
              placeholder={`Search literature for ${dataName}…`}
              className="w-full bg-transparent text-sm text-stone-800 placeholder:text-stone-300 focus:outline-none"
            />
          </div>
          <button
            onClick={onRun}
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-60"
            style={{ background: color }}
          >
            <RefreshIcon className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
        <div className="mt-2.5 flex flex-wrap items-center gap-2 text-[11px] text-stone-400">
          <span className="rounded-full bg-stone-100 px-2 py-0.5 font-medium text-stone-500">{kbLabel}</span>
          {result && <span>{result.sources.length} sources · {Math.round(result.elapsed_ms)}ms</span>}
        </div>

        {error && !loading && (
          <div className="mt-4 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600 ring-1 ring-red-100">{error}</div>
        )}

        {!loading && !error && result && result.framework && (
          <div className="mt-5 grid gap-4 lg:grid-cols-[auto_1fr]">
            <div className="flex items-center justify-center rounded-xl bg-white/70 p-5 ring-1 ring-stone-200/60">
              <ConfidenceRing pct={pct} color={color} />
            </div>
            <div className="rounded-xl bg-white/70 p-4 ring-1 ring-stone-200/60">
              <div className="mb-1.5 flex items-center gap-1.5 text-sm font-semibold text-stone-800">
                <BookIcon className="h-4 w-4" style={{ color }} /> Synthesis
              </div>
              <p className="text-sm leading-relaxed text-stone-600">{result.synthesis}</p>
            </div>
          </div>
        )}

        {!loading && !error && result && result.key_findings.length > 0 && (
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <div>
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-stone-400">
                Key findings
              </div>
              <ul className="flex flex-col gap-1.5">
                {result.key_findings.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm leading-relaxed text-stone-600">
                    <CheckIcon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />
                    {f}
                  </li>
                ))}
              </ul>
            </div>
            {result.evidence_gaps.length > 0 && (
              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-stone-400">
                  Evidence gaps
                </div>
                <ul className="flex flex-col gap-1.5">
                  {result.evidence_gaps.map((g, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm leading-relaxed text-stone-500">
                      <AlertTriangleIcon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                      {g}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {!loading && !error && result && sources.length === 0 && (
          <div className="mt-5 rounded-xl bg-stone-50 px-4 py-8 text-center text-sm text-stone-400 ring-1 ring-stone-100">
            No evidence retrieved yet. Run a search above.
          </div>
        )}
      </SectionCard>

      {/* Supporting literature */}
      <SectionCard title="Supporting Literature" icon={<LayersIcon className="h-4 w-4" />}>
        {loading && sources.length === 0 && (
          <div className="flex flex-col gap-3">
            <div className="h-12 animate-pulse rounded-xl bg-stone-100" />
            <div className="h-12 animate-pulse rounded-xl bg-stone-100" />
            <div className="h-12 animate-pulse rounded-xl bg-stone-100" />
          </div>
        )}
        {sources.length === 0 && !loading ? (
          <div className="rounded-xl bg-stone-50 px-4 py-8 text-center text-sm text-stone-400 ring-1 ring-stone-100">
            Retrieved sources will appear here.
          </div>
        ) : (
          sources.map((it) => <LiteratureRow key={it.chunk_id || `${it.source}:${it.page}`} item={it} expanded={expanded.has(it.chunk_id || `${it.source}:${it.page}`)} onToggle={() => toggle(it.chunk_id || `${it.source}:${it.page}`)} />)
        )}
      </SectionCard>
    </div>
  );
}

function ConfidenceRing({ pct, color, size = 104 }: { pct: number; color: string; size?: number }) {
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <div
        className="absolute inset-0 rounded-full transition-all duration-700"
        style={{ background: `conic-gradient(${color} ${pct * 3.6}deg, rgba(0,0,0,0.07) 0deg)` }}
      />
      <div className="absolute inset-[7px] flex flex-col items-center justify-center rounded-full bg-white/85 backdrop-blur">
        <span className="text-2xl font-bold tabular-nums" style={{ color }}>
          {pct}%
        </span>
        <span className="text-[10px] text-stone-400">confidence</span>
      </div>
    </div>
  );
}

function LiteratureRow({
  item,
  expanded,
  onToggle,
}: {
  item: EvidenceItem;
  expanded: boolean;
  onToggle: () => void;
}) {
  const quality = Math.round((item.quality_rating ?? 0) * 100);
  return (
    <div className="rounded-xl bg-white/70 px-4 py-3 ring-1 ring-stone-200/60">
      <div className="flex flex-wrap items-center gap-2">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
          <BookIcon className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-stone-800">{item.source}</div>
          <div className="text-[11px] text-stone-400">
            {item.framework && `${item.framework} · `}
            {item.chapter && `Ch. ${item.chapter} · `}
            {item.page > 0 && `p. ${item.page} · `}
            {quality}% match
          </div>
        </div>
        <button
          onClick={onToggle}
          className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-stone-500 transition hover:bg-stone-100 hover:text-stone-700"
        >
          {expanded ? "Collapse" : "Expand"}
          <ChevronDownIcon className={`h-3.5 w-3.5 transition-transform ${expanded ? "rotate-180" : ""}`} />
        </button>
        <button
          title="Copy citation"
          className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-stone-400 transition hover:bg-stone-100 hover:text-stone-600"
        >
          <ExternalLinkIcon className="h-3.5 w-3.5" />
        </button>
      </div>
      {expanded && (
        <p className="mt-2.5 border-t border-stone-100 pt-2.5 text-sm leading-relaxed text-stone-500">
          {item.content}
        </p>
      )}
    </div>
  );
}
