import { useCallback, useEffect, useRef, useState } from "react";
import { searchEvidence, type EvidenceSearchResponse } from "../api/client";
import { compareFrameworks, type ItemKind } from "../lib/compare";
import { FRAMEWORK_CARDS, FRAMEWORK_ID_TO_API, type FrameworkCardData, type FrameworkCardId } from "../lib/frameworks";
import { useNav } from "../lib/nav";
import { FrameworkGlyph } from "../ui/FrameworkCard";
import {
  AlertTriangleIcon,
  ArrowLeftIcon,
  BookIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  GitCompareIcon,
  PinIcon,
  RefreshIcon,
} from "../ui/icons";

type Mode = "all" | "pair";
type Highlight = "none" | "similar" | "diff";

export default function FrameworkComparePage() {
  const { navigate } = useNav();
  const [mode, setMode] = useState<Mode>("pair");
  const [highlight, setHighlight] = useState<Highlight>("none");
  const [selected, setSelected] = useState<FrameworkCardId[]>(["cbt", "act"]);
  const [pinned, setPinned] = useState<Set<FrameworkCardId>>(new Set(["general"]));
  const [expanded, setExpanded] = useState<Set<FrameworkCardId>>(new Set());

  const [evidence, setEvidence] = useState<Record<string, EvidenceSearchResponse | null>>({});
  const [loadingEv, setLoadingEv] = useState<Set<string>>(new Set());
  const cacheRef = useRef<Record<string, EvidenceSearchResponse | null>>({});
  const inFlight = useRef<Set<string>>(new Set());

  const ensureEvidence = useCallback((id: FrameworkCardId) => {
    if (cacheRef.current[id] !== undefined || inFlight.current.has(id)) return;
    inFlight.current.add(id);
    setLoadingEv((prev) => new Set(prev).add(id));
    const data = FRAMEWORK_CARDS.find((c) => c.id === id)!;
    const apiFw = FRAMEWORK_ID_TO_API[id];
    searchEvidence(
      `Effectiveness and core mechanisms of ${data.name} (${data.tagline})`,
      apiFw ? { framework: apiFw, nResults: 3 } : { nResults: 3 },
    )
      .then((res) => {
        cacheRef.current[id] = res;
        setEvidence((prev) => ({ ...prev, [id]: res }));
      })
      .catch(() => {
        cacheRef.current[id] = null;
        setEvidence((prev) => ({ ...prev, [id]: null }));
      })
      .finally(() => {
        inFlight.current.delete(id);
        setLoadingEv((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      });
  }, []);

  useEffect(() => {
    selected.forEach((id) => ensureEvidence(id));
  }, [selected, ensureEvidence]);

  const togglePinned = (id: FrameworkCardId) =>
    setPinned((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleExpanded = (id: FrameworkCardId) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    ensureEvidence(id);
  };

  const pick = (id: FrameworkCardId) => {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 2) return prev;
      return [...prev, id];
    });
  };

  const ordered = [...FRAMEWORK_CARDS].sort((a, b) => {
    const pa = pinned.has(a.id) ? 0 : 1;
    const pb = pinned.has(b.id) ? 0 : 1;
    return pa - pb;
  });

  const comparison = selected.length === 2 ? compareFrameworks(selected[0], selected[1]) : null;

  return (
    <div className="relative">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-16 top-10 h-64 w-64 rounded-full bg-indigo-200/40 blur-3xl" />
        <div className="absolute -right-16 top-1/3 h-72 w-72 rounded-full bg-sky-200/40 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-56 w-56 rounded-full bg-emerald-200/30 blur-3xl" />
      </div>

      <div className="relative">
        <button
          onClick={() => navigate({ section: "frameworks", frameworkMode: "cards" })}
          className="inline-flex items-center gap-1.5 text-sm font-medium text-stone-500 transition hover:text-stone-800"
        >
          <ArrowLeftIcon className="h-4 w-4" /> Framework cards
        </button>

        {/* Header */}
        <header className="mt-4 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-indigo-600">
              <GitCompareIcon className="h-3.5 w-3.5" /> Framework Comparison
            </p>
            <h1 className="mt-1 text-2xl font-bold tracking-tight text-stone-900">Compare frameworks</h1>
            <p className="mt-1 max-w-2xl text-sm text-stone-500">
              Analyses side-by-side — expand any framework, or pick two to highlight what they share
              and where they diverge.
            </p>
          </div>
        </header>

        {/* Toolbar */}
        <div className="mt-5 flex flex-wrap items-center gap-3 rounded-2xl border border-white/70 bg-white/60 p-3 shadow-[0_1px_2px_rgba(15,23,42,0.04),0_8px_24px_-12px_rgba(15,23,42,0.1)] backdrop-blur-xl">
          <div className="flex gap-1 rounded-lg bg-stone-100/80 p-1">
            {[
              { id: "all" as Mode, label: "All frameworks" },
              { id: "pair" as Mode, label: "Compare two" },
            ].map((m) => (
              <button
                key={m.id}
                onClick={() => setMode(m.id)}
                className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                  mode === m.id ? "bg-white text-stone-900 shadow-sm" : "text-stone-500 hover:text-stone-700"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {mode === "all" ? (
            <div className="flex gap-1.5">
              <button
                onClick={() => setExpanded(new Set(FRAMEWORK_CARDS.map((c) => c.id)))}
                className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-stone-500 transition hover:bg-stone-100 hover:text-stone-700"
              >
                Expand all
              </button>
              <button
                onClick={() => setExpanded(new Set())}
                className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-stone-500 transition hover:bg-stone-100 hover:text-stone-700"
              >
                Collapse all
              </button>
            </div>
          ) : (
            <>
              <div className="hidden h-5 w-px bg-stone-200 sm:block" />
              <div className="flex flex-wrap gap-1.5">
                {FRAMEWORK_CARDS.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => pick(c.id)}
                    className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold transition ${
                      selected.includes(c.id)
                        ? "text-white shadow-sm"
                        : "bg-white/70 text-stone-500 ring-1 ring-stone-200/70 hover:text-stone-700"
                    }`}
                    style={selected.includes(c.id) ? { background: c.color } : undefined}
                  >
                    {pinned.has(c.id) && <PinIcon className="h-3 w-3" />}
                    {c.name}
                  </button>
                ))}
              </div>
              <button
                onClick={() => setSelected([])}
                className="ml-auto rounded-lg px-2.5 py-1.5 text-xs font-medium text-stone-400 transition hover:text-stone-600"
              >
                Clear
              </button>
            </>
          )}

          <div className="ml-auto hidden h-5 w-px bg-stone-200 lg:block" />
          <div className="flex gap-1 rounded-lg bg-stone-100/80 p-1">
            {[
              { id: "none" as Highlight, label: "No highlight" },
              { id: "similar" as Highlight, label: "Similarities" },
              { id: "diff" as Highlight, label: "Differences" },
            ].map((h) => (
              <button
                key={h.id}
                onClick={() => setHighlight(h.id)}
                disabled={mode === "all"}
                title={mode === "all" ? "Select Compare two to highlight" : undefined}
                className={`rounded-md px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${
                  highlight === h.id ? "bg-white text-stone-900 shadow-sm" : "text-stone-500 hover:text-stone-700"
                }`}
              >
                {h.label}
              </button>
            ))}
          </div>
        </div>

        {mode === "all" && (
          <div className="mt-8 flex gap-6 overflow-x-auto pb-6 pt-2 pl-2">
            {ordered.map((c) => (
              <CompareCard
                key={c.id}
                data={c}
                expanded={expanded.has(c.id)}
                pinned={pinned.has(c.id)}
                onExpand={() => toggleExpanded(c.id)}
                onPin={() => togglePinned(c.id)}
                evidence={evidence[c.id]}
                loadingEv={loadingEv.has(c.id)}
              />
            ))}
          </div>
        )}

        {/* Compare two */}
        {mode === "pair" && comparison && (
          <>
            {/* Stats strip */}
            <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
              <StatChip
                label="Shared features"
                value={String(comparison.sharedCount)}
                icon={<CheckIcon className="h-4 w-4" />}
                tone="emerald"
              />
              <StatChip
                label={`Unique to ${comparison.a.name}`}
                value={String(uniqueCountFor(comparison, "a"))}
                icon={<GitCompareIcon className="h-4 w-4" />}
                color={comparison.a.color}
              />
              <StatChip
                label={`Unique to ${comparison.b.name}`}
                value={String(uniqueCountFor(comparison, "b"))}
                icon={<GitCompareIcon className="h-4 w-4" />}
                color={comparison.b.color}
              />
              <StatChip
                label="Concept overlap"
                value={`${comparison.overlapPct}%`}
                icon={<RefreshIcon className="h-4 w-4" />}
                tone="indigo"
              />
            </div>

            <div className="mt-8 grid gap-8 lg:grid-cols-2">
              <PairColumn
                data={comparison.a}
                comparison={comparison}
                side="a"
                highlight={highlight}
                evidence={evidence[comparison.a.id]}
                loadingEv={loadingEv.has(comparison.a.id)}
                onOpenWorkspace={(id) => navigate({ section: "frameworks", frameworkId: id })}
              />
              <PairColumn
                data={comparison.b}
                comparison={comparison}
                side="b"
                highlight={highlight}
                evidence={evidence[comparison.b.id]}
                loadingEv={loadingEv.has(comparison.b.id)}
                onOpenWorkspace={(id) => navigate({ section: "frameworks", frameworkId: id })}
              />
            </div>

            <p className="mt-4 flex items-center gap-1.5 text-[11px] text-stone-400">
              <AlertTriangleIcon className="h-3.5 w-3.5 text-amber-500" />
              Similarity matching is based on normalized wording of features — a clinical aid, not a
              judgment of superiority.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

/* ── Helpers ───────────────────────────────────────────────────────── */

function uniqueCountFor(c: ReturnType<typeof compareFrameworks>, side: "a" | "b"): number {
  if (side === "a") {
    return c.onlyATags.length + c.onlyAStrengths.length + c.onlyALimitations.length + c.onlyAInterventions.length;
  }
  return c.onlyBTags.length + c.onlyBStrengths.length + c.onlyBLimitations.length + c.onlyBInterventions.length;
}

function StatChip({
  label,
  value,
  icon,
  tone,
  color,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  tone?: "emerald" | "indigo" | "amber";
  color?: string;
}) {
  const toneCls =
    tone === "emerald"
      ? "bg-emerald-50 text-emerald-600"
      : tone === "amber"
        ? "bg-amber-50 text-amber-600"
        : "bg-indigo-50 text-indigo-600";
  return (
    <div className="flex items-center gap-3 rounded-xl border border-white/70 bg-white/60 p-3.5 backdrop-blur-xl">
      <span className={`flex h-9 w-9 items-center justify-center rounded-lg ${toneCls}`} style={color ? { background: `${color}1a`, color } : undefined}>
        {icon}
      </span>
      <div className="min-w-0">
        <div className="text-lg font-bold tabular-nums leading-tight text-stone-800">{value}</div>
        <div className="truncate text-[11px] text-stone-400">{label}</div>
      </div>
    </div>
  );
}

/* ── All-mode expandable card ──────────────────────────────────────── */

function CompareCard({
  data,
  expanded,
  pinned,
  onExpand,
  onPin,
  evidence,
  loadingEv,
}: {
  data: FrameworkCardData;
  expanded: boolean;
  pinned: boolean;
  onExpand: () => void;
  onPin: () => void;
  evidence: EvidenceSearchResponse | null | undefined;
  loadingEv: boolean;
}) {
  const pct = Math.round(data.confidence * 100);
  return (
    <div
      className={`flex w-[380px] shrink-0 flex-col rounded-2xl border bg-white/70 backdrop-blur-xl transition-all duration-300 ${
        expanded ? "border-white shadow-[0_12px_40px_-12px_rgba(15,23,42,0.15)]" : "border-white/80 shadow-md"
      } ${pinned ? "ring-2 ring-offset-1" : ""}`}
      style={pinned ? { borderColor: data.color } : undefined}
    >
      {/* head */}
      <div className="flex items-center gap-3 p-4 pb-3">
        <span
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-white"
          style={{ background: `linear-gradient(135deg, ${data.color}, ${data.color}cc)` }}
        >
          <FrameworkGlyph id={data.id} className="h-4.5 w-4.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="text-sm font-semibold text-stone-800">{data.name}</span>
            {data.featured && (
              <span className="rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase text-white" style={{ background: data.color }}>
                Core
              </span>
            )}
          </div>
          <p className="truncate text-[11px] text-stone-400">{data.tagline}</p>
        </div>
        <button
          onClick={onPin}
          title={pinned ? "Unpin" : "Pin"}
          className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg transition ${
            pinned ? "text-stone-800" : "text-stone-300 hover:bg-white/80 hover:text-stone-500"
          }`}
        >
          <PinIcon className={`h-3.5 w-3.5 ${pinned ? "fill-current" : ""}`} />
        </button>
        <button
          onClick={onExpand}
          aria-expanded={expanded}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-stone-400 transition hover:bg-white/80 hover:text-stone-700 active:scale-90"
        >
          <ChevronDownIcon className={`h-4 w-4 transition-transform duration-300 ${expanded ? "rotate-180" : ""}`} />
        </button>
      </div>

      {/* collapsed meta */}
      {!expanded && (
        <div className="px-4 pb-4">
          <div className="flex items-center gap-2">
            <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-stone-200/70">
              <span className="block h-full rounded-full" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${data.color}77, ${data.color})` }} />
            </span>
            <span className="text-[11px] font-semibold tabular-nums" style={{ color: data.color }}>
              {pct}%
            </span>
          </div>
          <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-stone-500">{data.description}</p>
        </div>
      )}

      {/* expanded */}
      {expanded && (
        <div className="flex flex-col gap-3 border-t border-stone-200/70 px-4 py-3.5">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-stone-400">Primary focus</div>
            <p className="mt-0.5 text-xs font-semibold text-stone-700">{data.focus}</p>
          </div>
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-stone-400">When to consider</div>
            <p className="mt-0.5 text-xs leading-relaxed text-stone-500">{data.whenToUse}</p>
          </div>
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-stone-400">Suggested interventions</div>
            <ul className="mt-1 flex flex-col gap-1">
              {data.workspace.suggestedInterventions.map((i) => (
                <li key={i.name} className="flex items-start gap-1.5 text-xs text-stone-600">
                  <CheckIcon className="mt-0.5 h-3 w-3 shrink-0 text-emerald-500" />
                  {i.name}
                </li>
              ))}
            </ul>
          </div>

          {/* Evidence preview */}
          <MiniEvidence data={data} evidence={evidence} loading={loadingEv} />
        </div>
      )}
    </div>
  );
}

/* ── Pair-mode column ─────────────────────────────────────────────── */

function PairColumn({
  data,
  comparison,
  side,
  highlight,
  evidence,
  loadingEv,
  onOpenWorkspace,
}: {
  data: FrameworkCardData;
  comparison: ReturnType<typeof compareFrameworks>;
  side: "a" | "b";
  highlight: Highlight;
  evidence: EvidenceSearchResponse | null | undefined;
  loadingEv: boolean;
  onOpenWorkspace: (id: FrameworkCardId) => void;
}) {
  const onlyLabel = `Only ${data.name}`;
  const sharedTags = comparison.sharedTags;
  const onlyTags = side === "a" ? comparison.onlyATags : comparison.onlyBTags;
  const interventions = data.workspace.suggestedInterventions.map((i) => i.name);
  const sharedInterv = comparison.sharedInterventions;

  const pct = Math.round(data.confidence * 100);
  const focusShared = comparison.focusShared;

  return (
    <div className="flex flex-col overflow-hidden rounded-3xl border border-white/80 bg-white/70 shadow-lg backdrop-blur-xl">
      {/* head */}
      <div className="flex items-center gap-4 p-5" style={{ borderBottom: `3px solid ${data.color}` }}>
        <span
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-white"
          style={{ background: `linear-gradient(135deg, ${data.color}, ${data.color}cc)` }}
        >
          <FrameworkGlyph id={data.id} className="h-5 w-5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="text-base font-semibold text-stone-800">{data.name}</span>
            {data.featured && (
              <span className="rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase text-white" style={{ background: data.color }}>
                Core
              </span>
            )}
          </div>
          <p className="truncate text-[11px] text-stone-400">{data.tagline}</p>
        </div>
        <button
          onClick={() => onOpenWorkspace(data.id)}
          title="Open workspace"
          className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-semibold transition hover:bg-white/80"
          style={{ color: data.color }}
        >
          Workspace <ChevronRightIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex flex-col gap-4 p-4">
        {/* Focus */}
        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-stone-400">Primary focus</span>
            {focusShared && highlight === "similar" && <BadgePill tone="emerald">Shared</BadgePill>}
          </div>
          <p className="text-sm font-semibold text-stone-700">{data.focus}</p>
        </div>

        {/* Confidence */}
        <div>
          <div className="mb-1 flex items-baseline justify-between text-[11px]">
            <span className="font-medium text-stone-400">Confidence</span>
            <span className="font-semibold tabular-nums" style={{ color: data.color }}>
              {pct}% {comparison.confidenceClose && highlight === "similar" && <span className="text-emerald-500">· close</span>}
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-stone-200/70">
            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${data.color}77, ${data.color})` }} />
          </div>
        </div>

        {/* Tags */}
        <div>
          <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-stone-400">Approach tags</div>
          <div className="flex flex-wrap gap-1.5">
            {[...sharedTags.map((t) => ({ t, kind: "shared" as ItemKind })), ...onlyTags.map((t) => ({ t, kind: "only" as ItemKind }))].map(({ t, kind }) => (
              <CompareTag key={t} text={t} kind={kind} highlight={highlight} onlyLabel={onlyLabel} />
            ))}
          </div>
        </div>

        {/* Suggested interventions */}
        <Block title="Suggested interventions">
          {interventions.map((name) => (
            <CompareItem
              key={name}
              text={name}
              kind={sharedInterv.includes(name) ? "shared" : "only"}
              highlight={highlight}
              onlyLabel={onlyLabel}
            />
          ))}
        </Block>

        {/* Strengths */}
        <Block title="Strengths">
          {data.workspace.strengths.map((s) => {
            const kind = (side === "a" ? comparison.sharedStrengths : comparison.sharedStrengths).includes(s) ? "shared" : "only";
            return <CompareItem key={s} text={s} kind={kind} highlight={highlight} onlyLabel={onlyLabel} />;
          })}
        </Block>

        {/* Limitations */}
        <Block title="Limitations">
          {data.workspace.limitations.map((s) => {
            const kind = (side === "a" ? comparison.sharedLimitations : comparison.sharedLimitations).includes(s) ? "shared" : "only";
            return <CompareItem key={s} text={s} kind={kind} highlight={highlight} onlyLabel={onlyLabel} />;
          })}
        </Block>

        {/* Evidence preview */}
        <Block title="Evidence preview">
          <MiniEvidence data={data} evidence={evidence} loading={loadingEv} compact />
        </Block>
      </div>
    </div>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-stone-400">{title}</div>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

function BadgePill({ tone, children }: { tone: "emerald" | "amber"; children: React.ReactNode }) {
  return (
    <span className={`rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide ${tone === "emerald" ? "bg-emerald-100 text-emerald-600" : "bg-amber-100 text-amber-600"}`}>
      {children}
    </span>
  );
}

function CompareTag({
  text,
  kind,
  highlight,
  onlyLabel,
}: {
  text: string;
  kind: ItemKind;
  highlight: Highlight;
  onlyLabel: string;
}) {
  const s = itemStyle(kind, highlight, onlyLabel);
  return (
    <span className={`rounded-md px-2 py-1 text-[11px] font-medium ring-1 ${s.cls}`}>
      {text}
      {s.label && <span className={`ml-1 text-[9px] font-bold uppercase ${s.tone}`}>{s.label}</span>}
    </span>
  );
}

function CompareItem({
  text,
  kind,
  highlight,
  onlyLabel,
}: {
  text: string;
  kind: ItemKind;
  highlight: Highlight;
  onlyLabel: string;
}) {
  const s = itemStyle(kind, highlight, onlyLabel);
  return (
    <div className={`flex items-start gap-2 rounded-lg px-2.5 py-2 text-xs leading-relaxed text-stone-600 ring-1 transition-colors ${s.cls}`}>
      <span className={`mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full ${s.dot}`} />
      <span className="flex-1">{text}</span>
      {s.label && <span className={`shrink-0 rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase ${s.tone} ${s.badgeBg}`}>{s.label}</span>}
    </div>
  );
}

function itemStyle(kind: ItemKind, highlight: Highlight, onlyLabel: string) {
  if (highlight === "similar" && kind === "shared") {
    return {
      cls: "bg-emerald-50/80 ring-emerald-200",
      dot: "bg-emerald-500",
      tone: "text-emerald-600",
      badgeBg: "bg-emerald-100",
      label: "Similar",
    };
  }
  if (highlight === "diff" && kind === "only") {
    return {
      cls: "bg-amber-50/80 ring-amber-200",
      dot: "bg-amber-500",
      tone: "text-amber-600",
      badgeBg: "bg-amber-100",
      label: onlyLabel,
    };
  }
  return {
    cls: "bg-white/70 ring-stone-200/60",
    dot: "bg-stone-300",
    tone: "",
    badgeBg: "",
    label: null,
  };
}

/* ── Evidence preview ─────────────────────────────────────────────── */

function MiniEvidence({
  data,
  evidence,
  loading,
  compact = false,
}: {
  data: FrameworkCardData;
  evidence: EvidenceSearchResponse | null | undefined;
  loading: boolean;
  compact?: boolean;
}) {
  const sources = evidence?.sources ?? [];
  const conf = evidence?.confidence ?? null;
  const pct = conf != null ? Math.round(conf * 100) : null;

  return (
    <div className="rounded-xl bg-white/70 p-3 ring-1 ring-stone-200/60">
      <div className="mb-2 flex items-center justify-between">
        <span className="inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-stone-400">
          <BookIcon className="h-3 w-3" /> Evidence preview
        </span>
        {loading && <RefreshIcon className="h-3 w-3 animate-spin text-stone-300" />}
        {!loading && pct != null && (
          <span className="text-[10px] font-bold tabular-nums" style={{ color: data.color }}>
            {pct}% confidence
          </span>
        )}
      </div>

      {loading && sources.length === 0 && (
        <div className="flex flex-col gap-1.5">
          <div className="h-3 animate-pulse rounded bg-stone-100" />
          <div className="h-3 animate-pulse rounded bg-stone-100" />
        </div>
      )}

      {!loading && sources.length === 0 && (
        <p className="text-[11px] text-stone-400">
          {evidence === null ? "No indexed literature available." : "Run a search in the workspace for evidence."}
        </p>
      )}

      {sources.length > 0 && (
        <ul className="flex flex-col gap-1.5">
          {sources.slice(0, compact ? 2 : 3).map((s) => (
            <li key={s.chunk_id || `${s.source}:${s.page}`} className="flex items-center gap-1.5 text-[11px] text-stone-500">
              <span className="h-1 w-1 shrink-0 rounded-full" style={{ background: data.color }} />
              <span className="truncate">{s.source}</span>
              <span className="shrink-0 tabular-nums text-stone-400">{Math.round((s.quality_rating ?? 0) * 100)}%</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
