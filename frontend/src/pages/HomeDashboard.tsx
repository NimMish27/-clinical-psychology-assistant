import { useEffect, useRef, useState } from "react";
import {
  fetchKbCollections,
  rebuildKnowledgeBase,
  uploadFiles,
  type KbCollectionInfo,
} from "../api/client";
import { CASES, SESSIONS } from "../lib/sampleData";
import { useNav } from "../lib/nav";
import { useToast } from "../ui/primitives";
import {
  ActivityIcon,
  BookIcon,
  CheckCircleIcon,
  ChevronRightIcon,
  ClipboardIcon,
  ClockIcon,
  FileTextIcon,
  FolderIcon,
  LibraryIcon,
  PlusIcon,
  RefreshIcon,
  SearchIcon,
  SettingsIcon,
  ShieldIcon,
  SparklesIcon,
  UploadIcon,
  UsersIcon,
} from "../ui/icons";

/* ── Small building blocks ────────────────────────────────────────── */

function Card({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-2xl border border-stone-200 bg-white shadow-[0_1px_2px_0_rgb(28_25_23/0.05)] ${className}`}
    >
      {(title || action) && (
        <header className="flex items-center justify-between gap-3 border-b border-stone-100 px-5 py-3.5">
          <div>
            <h2 className="text-sm font-semibold text-stone-900">{title}</h2>
            {subtitle && <p className="text-xs text-stone-400 mt-0.5">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

function Pill({ children, tone = "stone" }: { children: React.ReactNode; tone?: "stone" | "indigo" | "green" | "amber" | "sky" | "red" }) {
  const map: Record<string, string> = {
    stone: "bg-stone-100 text-stone-600",
    indigo: "bg-indigo-50 text-indigo-600",
    green: "bg-emerald-50 text-emerald-600",
    amber: "bg-amber-50 text-amber-600",
    sky: "bg-sky-50 text-sky-600",
    red: "bg-red-50 text-red-600",
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold ${map[tone]}`}>
      {children}
    </span>
  );
}

/* ── Main dashboard ───────────────────────────────────────────────── */

export default function HomeDashboard() {
  const toast = useToast();
  const { openSection, openCase, toCasesIndex } = useNav();

  const [collections, setCollections] = useState<KbCollectionInfo[] | null>(null);
  const [rebuilding, setRebuilding] = useState(false);
  const [kbError, setKbError] = useState<string | null>(null);

  const [intakeMode, setIntakeMode] = useState<"upload" | "paste">("upload");
  const [impression, setImpression] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [pastedText, setPastedText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    fetchKbCollections()
      .then((r) => active && setCollections(r.collections))
      .catch((e) => active && setKbError(e.message));
    return () => {
      active = false;
    };
  }, []);

  const activeCases = CASES.filter((c) => c.status === "open" || c.status === "in-progress");
  const builtCount = collections?.filter((c) => c.built).length ?? 0;
  const totalChunks = collections?.reduce((a, c) => a + (c.chunk_count ?? 0), 0) ?? 0;

  const frameworkStats = ["cbt", "act", "humanistic"].map((fw) => {
    const caseCount = CASES.filter((c) => c.framework === fw).length;
    const kb = collections?.find((c) => c.framework === fw);
    return { framework: fw, caseCount, chunks: kb?.chunk_count ?? 0, built: kb?.built ?? false };
  });
  const maxCaseCount = Math.max(1, ...frameworkStats.map((f) => f.caseCount));

  async function handleUpload(files: FileList | File[]) {
    const pdfs = Array.from(files).filter((f) => f.type === "application/pdf" || f.name.endsWith(".pdf"));
    if (pdfs.length === 0) {
      toast.push("error", "No PDF selected", "Case files must be PDF documents.");
      return;
    }
    setUploading(true);
    try {
      const res = await uploadFiles(pdfs, { collection: "user_documents" });
      const ok = res.succeeded;
      toast.push(
        ok > 0 ? "success" : "error",
        ok > 0 ? `Uploaded ${ok} case file${ok === 1 ? "" : "s"}` : "Upload failed",
        `${res.total_chunks} chunks stored in user_documents`,
      );
      setImpression("");
    } catch (err) {
      toast.push("error", "Upload failed", err instanceof Error ? err.message : "Unknown error");
    } finally {
      setUploading(false);
    }
  }

  function handlePaste() {
    if (pastedText.trim().length < 3) {
      toast.push("error", "Nothing to record", "Paste some case notes first.");
      return;
    }
    toast.push("success", "Case notes recorded", "Pasted text saved as a draft case record.");
    setPastedText("");
  }

  async function handleRebuild() {
    setRebuilding(true);
    try {
      const res = await rebuildKnowledgeBase();
      const ok = res.status === "ok";
      toast.push(
        ok ? "success" : "error",
        ok ? "Knowledge base rebuilt" : "Rebuild failed",
        ok ? `${res.children} chunks across ${res.collections.length} collections` : res.errors.join("; "),
      );
      setCollections((await fetchKbCollections()).collections);
    } catch (err) {
      toast.push("error", "Rebuild failed", err instanceof Error ? err.message : "Unknown error");
    } finally {
      setRebuilding(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-5 py-7 md:px-8">
      {/* Header */}
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-indigo-600">
            Tuesday, August 11, 2026
          </p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-stone-900">
            Good morning, Dr. Reyes
          </h1>
          <p className="mt-1 text-sm text-stone-400">
            {activeCases.length} active cases · {builtCount}/{collections?.length ?? 3} frameworks online · safety compliance 92%
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={toCasesIndex}
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-stone-200 bg-white px-3.5 text-sm font-medium text-stone-700 transition hover:bg-stone-50"
          >
            <FolderIcon className="h-4 w-4" /> All cases
          </button>
          <button
            onClick={() => openSection("evidence")}
            className="inline-flex h-9 items-center gap-2 rounded-lg bg-indigo-600 px-3.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700"
          >
            <SearchIcon className="h-4 w-4" /> Search literature
          </button>
        </div>
      </header>

      {/* Intake — Upload Case / Paste Case / Initial Impression */}
      <Card className="mt-6 overflow-hidden" >
        <div className="flex border-b border-stone-100">
          {[
            { id: "upload", label: "Upload Case", icon: <UploadIcon className="h-4 w-4" /> },
            { id: "paste", label: "Paste Case", icon: <ClipboardIcon className="h-4 w-4" /> },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setIntakeMode(t.id as never)}
              className={`inline-flex items-center gap-2 px-5 py-3 text-sm font-medium transition ${
                intakeMode === t.id
                  ? "border-b-2 border-indigo-600 text-indigo-600"
                  : "border-b-2 border-transparent text-stone-400 hover:text-stone-600"
              }`}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        <div className="grid gap-6 p-5 md:grid-cols-5">
          {/* Left: intake action */}
          <div className="md:col-span-3">
            {intakeMode === "upload" ? (
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); handleUpload(e.dataTransfer.files); }}
                onClick={() => !uploading && fileRef.current?.click()}
                className={`flex h-52 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 text-center transition ${
                  dragOver ? "border-indigo-400 bg-indigo-50" : "border-stone-200 bg-stone-50/60 hover:border-indigo-300 hover:bg-indigo-50/50"
                }`}
              >
                <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-white text-indigo-600 shadow-sm">
                  <UploadIcon className="h-6 w-6" />
                </span>
                <p className="text-sm font-semibold text-stone-700">
                  {uploading ? "Uploading…" : "Drop a case file here or click to browse"}
                </p>
                <p className="text-xs text-stone-400">PDF documents · stored privately in your document collection</p>
                <input
                  ref={fileRef}
                  type="file"
                  multiple
                  accept=".pdf,application/pdf"
                  className="hidden"
                  onChange={(e) => e.target.files && handleUpload(e.target.files)}
                />
              </div>
            ) : (
              <div className="flex h-52 flex-col">
                <textarea
                  value={pastedText}
                  onChange={(e) => setPastedText(e.target.value)}
                  placeholder={"Paste the case narrative here…\n\nPresenting concerns, history, symptoms, context — anything you want to work from."}
                  className="h-full w-full resize-none rounded-xl border border-stone-200 bg-white p-3.5 text-sm leading-relaxed text-stone-800 placeholder:text-stone-300 focus:border-indigo-400 focus:outline-none focus:ring-3 focus:ring-indigo-50"
                />
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs text-stone-400">{pastedText.length} characters</span>
                  <button
                    onClick={handlePaste}
                    className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-indigo-600 px-3 text-xs font-semibold text-white transition hover:bg-indigo-700"
                  >
                    <PlusIcon className="h-3.5 w-3.5" /> Record case
                  </button>
                </div>
              </div>
            )}

            {/* Initial Clinical Impression */}
            <div className="mt-4">
              <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-stone-500">
                <FileTextIcon className="h-3.5 w-3.5" />
                Initial clinical impression <span className="font-normal text-stone-300">(optional)</span>
              </label>
              <textarea
                value={impression}
                onChange={(e) => setImpression(e.target.value)}
                placeholder="A brief first-pass formulation — symptoms, hypothesized mechanisms, tentative direction…"
                rows={3}
                className="w-full resize-none rounded-xl border border-stone-200 bg-white p-3 text-sm leading-relaxed text-stone-800 placeholder:text-stone-300 focus:border-indigo-400 focus:outline-none focus:ring-3 focus:ring-indigo-50"
              />
            </div>
          </div>

          {/* Right: quick summary of today */}
          <div className="hidden flex-col gap-3 md:col-span-2 md:flex">
            <div className="rounded-xl bg-gradient-to-br from-indigo-600 to-indigo-700 p-4 text-white shadow-sm">
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-indigo-200">
                Today
              </p>
              <p className="mt-2 text-3xl font-bold tracking-tight">3 sessions</p>
              <p className="mt-1 text-xs text-indigo-200">Next: Jordan Mitchell · 9:30 AM</p>
              <div className="mt-4 flex items-center justify-between rounded-lg bg-white/10 px-3 py-2 text-xs">
                <span className="text-indigo-100">Sign-off compliance</span>
                <span className="font-semibold">92%</span>
              </div>
            </div>
            <div className="rounded-xl border border-stone-100 bg-stone-50 p-4 text-xs text-stone-500">
              <p className="font-semibold text-stone-600">Clinical guardrails</p>
              <p className="mt-1 leading-relaxed">
                This assistant supports — it never diagnoses. Formulations are hypotheses for your
                review, and safety decisions always remain yours.
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Recent Cases + Quick Actions */}
      <div className="mt-6 grid gap-6 lg:grid-cols-12">
        <Card
          className="lg:col-span-8"
          title="Recent Cases"
          subtitle={`${activeCases.length} active · ${CASES.length} total`}
          action={
            <button onClick={toCasesIndex} className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-600 hover:text-indigo-700">
              View all <ChevronRightIcon className="h-3.5 w-3.5" />
            </button>
          }
        >
          <ul className="divide-y divide-stone-100">
            {CASES.slice(0, 4).map((c) => {
              const done = c.safetyChecklist.filter((s) => s.done).length;
              const safetyPct = Math.round((done / c.safetyChecklist.length) * 100);
              return (
                <li key={c.id}>
                  <button
                    onClick={() => openCase(c.id, "overview")}
                    className="group flex w-full items-center gap-3.5 px-5 py-3.5 text-left transition hover:bg-stone-50/70"
                  >
                    <span
                      className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-xs font-bold text-white"
                      style={{ background: c.color }}
                    >
                      {c.initials}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate text-sm font-semibold text-stone-800">{c.name}</span>
                        <Pill tone={c.framework === "act" ? "sky" : c.framework === "humanistic" ? "green" : "indigo"}>
                          {c.framework}
                        </Pill>
                      </div>
                      <p className="mt-0.5 truncate text-xs text-stone-400">{c.presentingConcern}</p>
                    </div>
                    <div className="hidden w-36 sm:block">
                      <div className="flex justify-between text-[10px] font-medium text-stone-400">
                        <span className="capitalize">{c.stage}</span>
                        <span>{c.stageProgress}%</span>
                      </div>
                      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-stone-100">
                        <div
                          className="h-full rounded-full bg-indigo-500 transition-all"
                          style={{ width: `${c.stageProgress}%` }}
                        />
                      </div>
                    </div>
                    <div className="hidden text-right md:block">
                      <p className="text-xs font-medium text-stone-600">{c.nextAppointment}</p>
                      <p className="mt-0.5 text-[11px] text-stone-400">safety {safetyPct}%</p>
                    </div>
                    <ChevronRightIcon className="h-4 w-4 shrink-0 text-stone-300 transition group-hover:translate-x-0.5 group-hover:text-indigo-500" />
                  </button>
                </li>
              );
            })}
          </ul>
        </Card>

        {/* Quick Actions */}
        <Card className="lg:col-span-4" title="Quick Actions" subtitle="Jump to a workflow">
          <div className="grid grid-cols-2 gap-2.5 p-4">
            {[
              { label: "New case", desc: "Upload or paste", icon: <PlusIcon className="h-4 w-4" />, onClick: () => document.querySelector(".main-content")?.scrollTo({ top: 0, behavior: "smooth" }) },
              { label: "Evidence", desc: "Ask the literature", icon: <LibraryIcon className="h-4 w-4" />, onClick: () => openSection("evidence") },
              { label: "Knowledge base", desc: "Rebuild & sources", icon: <BookIcon className="h-4 w-4" />, onClick: () => openSection("knowledge") },
              { label: "Safety center", desc: "Sign-offs & flags", icon: <ShieldIcon className="h-4 w-4" />, onClick: () => openSection("safety") },
              { label: "Case index", desc: "All clients", icon: <UsersIcon className="h-4 w-4" />, onClick: toCasesIndex },
              { label: "Settings", desc: "Preferences", icon: <SettingsIcon className="h-4 w-4" />, onClick: () => openSection("settings") },
            ].map((a) => (
              <button
                key={a.label}
                onClick={a.onClick}
                className="group flex flex-col items-start gap-1.5 rounded-xl border border-stone-100 bg-stone-50/50 p-3 text-left transition hover:border-indigo-200 hover:bg-indigo-50/50"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-indigo-600 shadow-sm transition group-hover:text-indigo-700">
                  {a.icon}
                </span>
                <span className="mt-1 text-xs font-semibold text-stone-700">{a.label}</span>
                <span className="text-[11px] text-stone-400">{a.desc}</span>
              </button>
            ))}
          </div>
        </Card>
      </div>

      {/* KB Status + Framework Stats + Recent Sessions */}
      <div className="mt-6 grid gap-6 lg:grid-cols-12">
        {/* Knowledge Base Status */}
        <Card
          className="lg:col-span-5"
          title="Knowledge Base Status"
          subtitle={kbError ? undefined : `${builtCount}/${collections?.length ?? 3} frameworks indexed`}
          action={
            <button
              onClick={handleRebuild}
              disabled={rebuilding}
              className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-stone-200 bg-white px-2.5 text-[11px] font-semibold text-stone-600 transition hover:bg-stone-50 disabled:opacity-50"
            >
              <RefreshIcon className={`h-3.5 w-3.5 ${rebuilding ? "animate-spin" : ""}`} />
              {rebuilding ? "Rebuilding…" : "Rebuild"}
            </button>
          }
        >
          <div className="flex flex-col gap-2.5 p-4">
            {collections === null ? (
              <>
                <div className="h-12 animate-pulse rounded-xl bg-stone-100" />
                <div className="h-12 animate-pulse rounded-xl bg-stone-100" />
                <div className="h-12 animate-pulse rounded-xl bg-stone-100" />
              </>
            ) : kbError ? (
              <p className="px-1 py-4 text-center text-xs text-red-500">{kbError}</p>
            ) : (
              collections.map((c) => (
                <div
                  key={c.framework}
                  className="flex items-center gap-3 rounded-xl border border-stone-100 px-3.5 py-2.5"
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-xs font-bold uppercase text-indigo-600">
                    {c.framework.slice(0, 3)}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold capitalize text-stone-700">{c.framework}</p>
                    <p className="truncate text-[11px] text-stone-400">
                      {c.built ? `${c.chunk_count ?? 0} chunks · v${c.index_version}` : "Not built yet"}
                    </p>
                  </div>
                  {c.built ? (
                    <Pill tone="green">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /> Online
                    </Pill>
                  ) : (
                    <Pill tone="amber">Build</Pill>
                  )}
                </div>
              ))
            )}
            <div className="mt-1 flex items-center justify-between rounded-xl bg-stone-50 px-3.5 py-2.5 text-xs">
              <span className="text-stone-500">Total indexed chunks</span>
              <span className="font-bold text-stone-700">{totalChunks}</span>
            </div>
          </div>
        </Card>

        {/* Framework Statistics */}
        <Card className="lg:col-span-3" title="Framework Statistics" subtitle="Cases by therapeutic orientation">
          <div className="flex flex-col gap-3.5 p-4">
            {frameworkStats.map((f) => (
              <div key={f.framework}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="font-semibold capitalize text-stone-700">{f.framework}</span>
                  <span className="text-stone-400">{f.caseCount} cases</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-stone-100">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-indigo-600 transition-all duration-500"
                    style={{ width: `${(f.caseCount / maxCaseCount) * 100}%` }}
                  />
                </div>
                <p className="mt-1 text-[10px] text-stone-300">{f.chunks} chunks in KB</p>
              </div>
            ))}
            <div className="mt-1 flex items-center justify-between rounded-xl border border-stone-100 px-3 py-2 text-[11px]">
              <span className="inline-flex items-center gap-1.5 text-stone-500">
                <ActivityIcon className="h-3.5 w-3.5" /> Most used
              </span>
              <span className="font-semibold capitalize text-stone-700">
                {frameworkStats.sort((a, b) => b.caseCount - a.caseCount)[0]?.framework}
              </span>
            </div>
          </div>
        </Card>

        {/* Recent Sessions */}
        <Card
          className="lg:col-span-4"
          title="Recent Sessions"
          subtitle="Last 5 sessions"
          action={
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-stone-400">
              <ClockIcon className="h-3.5 w-3.5" /> 5 sessions this week
            </span>
          }
        >
          <ul className="divide-y divide-stone-100">
            {SESSIONS.map((s) => (
              <li key={s.id} className="flex items-start gap-3 px-5 py-3">
                <span
                  className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[11px] font-bold text-white"
                  style={{ background: s.color }}
                >
                  {s.clientInitials}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-semibold text-stone-700">{s.clientName}</span>
                    <Pill tone={s.framework === "ACT" ? "sky" : s.framework === "Humanistic" ? "green" : "indigo"}>
                      {s.framework}
                    </Pill>
                  </div>
                  <p className="mt-0.5 truncate text-xs text-stone-400">{s.notes}</p>
                </div>
                <div className="shrink-0 text-right">
                  <p className="text-xs font-semibold text-stone-600">
                    {s.date} · {s.time}
                  </p>
                  <p className="mt-0.5 text-[11px] text-stone-300">{s.durationMin} min</p>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      {/* Footer strip */}
      <footer className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-stone-200 pt-4 text-[11px] text-stone-400">
        <span className="inline-flex items-center gap-1.5">
          <SparklesIcon className="h-3.5 w-3.5 text-indigo-400" />
          Clinical Psychology Assistant · v0.1.0
        </span>
        <span className="inline-flex items-center gap-1.5">
          <CheckCircleIcon className="h-3.5 w-3.5 text-emerald-500" />
          All systems operational
        </span>
      </footer>
    </div>
  );
}
