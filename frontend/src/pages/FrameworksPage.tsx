import { useEffect, useState } from "react";
import { fetchKbCollections, type KbCollectionInfo } from "../api/client";
import { FRAMEWORK_CARDS, FRAMEWORK_ID_TO_KB } from "../lib/frameworks";
import { useNav } from "../lib/nav";
import FrameworkCard from "../ui/FrameworkCard";
import { DatabaseIcon, GitCompareIcon, LayersIcon, ShieldIcon, SparklesIcon } from "../ui/icons";

export default function FrameworksPage() {
  const { openFramework, navigate } = useNav();
  const [collections, setCollections] = useState<KbCollectionInfo[] | null>(null);

  useEffect(() => {
    let active = true;
    fetchKbCollections()
      .then((r) => active && setCollections(r.collections))
      .catch(() => active && setCollections([]));
    return () => {
      active = false;
    };
  }, []);

  const isIndexed = (id: string): boolean => {
    const kb = FRAMEWORK_ID_TO_KB[id as keyof typeof FRAMEWORK_ID_TO_KB];
    if (!kb || !collections) return false;
    return collections.some((c) => c.framework === kb && c.built);
  };

  const indexedCount = FRAMEWORK_CARDS.filter((c) => isIndexed(c.id)).length;
  const avgConfidence = Math.round(
    (FRAMEWORK_CARDS.reduce((a, c) => a + c.confidence, 0) / FRAMEWORK_CARDS.length) * 100,
  );
  const totalSources = FRAMEWORK_CARDS.reduce((a, c) => a + c.evidenceCount, 0);

  return (
    <div className="relative">
      {/* decorative page background */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-24 left-1/4 h-72 w-72 rounded-full bg-indigo-200/40 blur-3xl" />
        <div className="absolute right-0 top-1/3 h-80 w-80 rounded-full bg-sky-200/40 blur-3xl" />
        <div className="absolute -bottom-24 left-10 h-64 w-64 rounded-full bg-emerald-200/30 blur-3xl" />
      </div>

      <div className="relative">
        {/* Header */}
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-indigo-600">
              <LayersIcon className="h-3.5 w-3.5" /> Framework Knowledge
            </p>
            <h1 className="mt-1 text-2xl font-bold tracking-tight text-stone-900">Framework cards</h1>
            <p className="mt-1 max-w-xl text-sm text-stone-500">
              Evidence-graded therapeutic frameworks for case comparison — suitability, confidence,
              and the literature behind each.
            </p>
          </div>
          <button
            onClick={() => navigate({ section: "frameworks", frameworkMode: "compare" })}
            className="inline-flex items-center gap-2 rounded-xl bg-stone-900 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-stone-800"
          >
            <GitCompareIcon className="h-4 w-4" /> Compare frameworks
          </button>
        </header>

        {/* Summary strip */}
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: "Frameworks", value: String(FRAMEWORK_CARDS.length), icon: <LayersIcon className="h-4 w-4" /> },
            { label: "Indexed in KB", value: collections === null ? "…" : String(indexedCount), icon: <DatabaseIcon className="h-4 w-4" /> },
            { label: "Avg confidence", value: `${avgConfidence}%`, icon: <SparklesIcon className="h-4 w-4" /> },
            { label: "Literature sources", value: String(totalSources), icon: <ShieldIcon className="h-4 w-4" /> },
          ].map((s) => (
            <div
              key={s.label}
              className="flex items-center gap-3 rounded-xl border border-white/70 bg-white/60 p-3.5 backdrop-blur-xl transition-colors hover:bg-white/80"
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
                {s.icon}
              </span>
              <div>
                <div className="text-lg font-bold tabular-nums leading-tight text-stone-800">{s.value}</div>
                <div className="text-[11px] text-stone-400">{s.label}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Card grid */}
        <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {FRAMEWORK_CARDS.map((fw) => (
            <FrameworkCard key={fw.id} data={fw} onOpen={() => openFramework(fw.id)} />
          ))}
        </div>

        {/* Footer note */}
        <footer className="mt-8 flex items-center gap-2 text-[11px] text-stone-400">
          <span className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
          Confidence reflects retrieval quality from the indexed framework knowledge base and curated
          literature — never a diagnosis.
        </footer>
      </div>
    </div>
  );
}
