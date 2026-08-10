import { useState, type CSSProperties, type ReactNode } from "react";
import type { FrameworkCardData, FrameworkCardId } from "../lib/frameworks";
import { BookIcon, ChevronDownIcon, ChevronRightIcon } from "./icons";

const GLYPHS: Record<FrameworkCardId, ReactNode> = {
  cbt: (
    <>
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      <path d="M9 10.5h.01M15 10.5h.01M9.5 14.5a2 2 0 0 0 3.5 0" />
    </>
  ),
  act: (
    <>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="5" />
      <circle cx="12" cy="12" r="1.1" />
    </>
  ),
  dbt: (
    <>
      <path d="M12 3v18" />
      <path d="M8 21h8" />
      <path d="m5 7 2-3 2 3" />
      <path d="m15 7 2-3 2 3" />
      <path d="M7 7h4" />
      <path d="M17 7h-4" />
    </>
  ),
  mi: (
    <>
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      <path d="M8.5 9.5h7" />
      <path d="M8.5 13h4" />
    </>
  ),
  humanistic: (
    <>
      <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
      <path d="M12 17v4" />
    </>
  ),
  "alliance-repair": (
    <>
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </>
  ),
  general: (
    <>
      <path d="m12 2 1.6 6.4L20 10l-6.4 1.6L12 18l-1.6-6.4L4 10l6.4-1.6Z" />
      <path d="m19 15 .55 2.45L22 18l-2.45.55L19 21l-.55-2.45L16 18l2.45-.55Z" />
    </>
  ),
};

export function FrameworkGlyph({ id, className }: { id: FrameworkCardId; className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {GLYPHS[id]}
    </svg>
  );
}

function confidenceTone(confidence: number): string {
  if (confidence >= 0.8) return "text-emerald-600";
  if (confidence >= 0.7) return "text-indigo-600";
  return "text-stone-600";
}

export default function FrameworkCard({
  data,
  onOpen,
}: {
  data: FrameworkCardData;
  onOpen?: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const { id, name, tagline, description, focus, whenToUse, tags, color, confidence, evidenceCount, featured } = data;
  const pct = Math.round(confidence * 100);

  const accent: CSSProperties = { "--fw": color } as CSSProperties;

  return (
    <section
      style={accent}
      onClick={onOpen}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && onOpen) {
          e.preventDefault();
          onOpen();
        }
      }}
      className="group relative flex cursor-pointer flex-col overflow-hidden rounded-2xl border border-white/70 bg-white/60 p-5 shadow-[0_1px_2px_rgba(15,23,42,0.05),0_10px_30px_-12px_rgba(15,23,42,0.12)] backdrop-blur-xl transition-all duration-300 ease-out hover:-translate-y-1 hover:border-white/90 hover:shadow-[0_2px_4px_rgba(15,23,42,0.04),0_20px_48px_-16px_rgba(15,23,42,0.24)]"
    >
      {/* coloured glow */}
      <div
        className="pointer-events-none absolute -right-10 -top-12 h-32 w-32 rounded-full opacity-[0.18] blur-2xl transition-opacity duration-500 group-hover:opacity-40"
        style={{ background: color }}
      />
      <div
        className="pointer-events-none absolute -bottom-16 -left-10 h-28 w-28 rounded-full opacity-[0.08] blur-2xl transition-opacity duration-500 group-hover:opacity-20"
        style={{ background: color }}
      />

      <div className="relative flex items-start gap-3">
        <span
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-white shadow-md transition-transform duration-300 ease-out group-hover:-rotate-3 group-hover:scale-110"
          style={{
            background: `linear-gradient(135deg, ${color}, ${color}cc)`,
            boxShadow: `0 8px 18px -8px ${color}b3`,
          }}
        >
          <FrameworkGlyph id={id} className="h-5 w-5" />
        </span>

        <div className="min-w-0 flex-1 pt-0.5">
          <div className="flex items-center gap-2">
            <h3 className="text-[15px] font-semibold tracking-tight text-stone-800">{name}</h3>
            {featured && (
              <span
                className="rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-white"
                style={{ background: color }}
              >
                Core
              </span>
            )}
          </div>
          <p className="truncate text-xs text-stone-400">{tagline}</p>
        </div>

        <button
          onClick={(e) => {
            e.stopPropagation();
            setExpanded((prev) => !prev);
          }}
          aria-expanded={expanded}
          aria-label={expanded ? "Collapse framework details" : "Expand framework details"}
          title={expanded ? "Collapse" : "Expand"}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-stone-400 transition-all duration-200 hover:bg-white/80 hover:text-stone-700 active:scale-90"
        >
          <ChevronDownIcon className={`h-4 w-4 transition-transform duration-300 ${expanded ? "rotate-180" : ""}`} />
        </button>
      </div>

      <p className="relative mt-3 text-[13px] leading-relaxed text-stone-600">{description}</p>

      {/* Primary focus */}
      <div className="relative mt-3 flex items-center gap-2 rounded-lg bg-white/60 px-3 py-2 ring-1 ring-stone-200/60 transition-colors duration-300 group-hover:bg-white/80">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: color }} />
        <span className="text-[11px] font-medium text-stone-400">Primary focus</span>
        <span className="truncate text-xs font-semibold text-stone-700">{focus}</span>
      </div>

      {/* Confidence + evidence */}
      <div className="relative mt-4 flex items-end gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between text-[11px]">
            <span className="font-medium text-stone-400">Confidence</span>
            <span className={`font-semibold tabular-nums ${confidenceTone(confidence)}`}>{pct}%</span>
          </div>
          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-stone-200/70">
            <div
              className="h-full rounded-full transition-all duration-700 ease-out"
              style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}77, ${color})` }}
            />
          </div>
        </div>
        <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-white/70 px-2.5 py-1 text-[11px] font-semibold text-stone-600 ring-1 ring-stone-200/70">
          <BookIcon className="h-3.5 w-3.5" style={{ color }} />
          {evidenceCount} sources
        </span>
      </div>

      {/* Expandable detail */}
      <div
        className={`relative grid transition-all duration-300 ease-out ${
          expanded ? "mt-3 grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        }`}
      >
        <div className="min-h-0 overflow-hidden">
          <div className="border-t border-stone-200/70 pt-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-stone-400">
              When to consider
            </div>
            <p className="mt-1 text-xs leading-relaxed text-stone-500">{whenToUse}</p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {tags.map((t) => (
                <span
                  key={t}
                  className="rounded-md bg-white/70 px-2 py-1 text-[11px] font-medium text-stone-600 ring-1 ring-stone-200/60"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Open workspace */}
      <div className="relative mt-4 flex items-center justify-between border-t border-stone-200/60 pt-3">
        <span className="text-[11px] font-medium text-stone-400">Framework workspace</span>
        <span
          className="inline-flex items-center gap-1 text-xs font-semibold transition-all duration-200"
          style={{ color }}
        >
          Open
          <ChevronRightIcon className="h-3.5 w-3.5 transition-transform duration-300 group-hover:translate-x-0.5" />
        </span>
      </div>
    </section>
  );
}
