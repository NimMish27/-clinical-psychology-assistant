import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  AlertCircleIcon,
  AlertTriangleIcon,
  CheckIcon,
  CommandIcon,
  InfoIcon,
  SearchIcon,
} from "./icons";

/* ── Card / Panel ─────────────────────────────────────────────────── */

export function Card({
  children,
  className = "",
  padded = false,
  style,
}: {
  children: ReactNode;
  className?: string;
  padded?: boolean;
  style?: React.CSSProperties;
}) {
  return (
    <div className={`card ${padded ? "card-pad" : ""} ${className}`} style={style}>
      {children}
    </div>
  );
}

export function Panel({
  title,
  actions,
  children,
  className = "",
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`panel ${className}`}>
      {title && (
        <div className="card-head">
          <h3>{title}</h3>
          <div className="spacer" />
          {actions}
        </div>
      )}
      {children}
    </div>
  );
}

/* ── Badge / Tag ──────────────────────────────────────────────────── */

export function Badge({
  tone = "neutral",
  dot = false,
  children,
}: {
  tone?: "neutral" | "primary" | "success" | "warning" | "danger" | "info";
  dot?: boolean;
  children: ReactNode;
}) {
  return (
    <span className={`badge badge-${tone}`}>
      {dot && <span className="badge-dot" />}
      {children}
    </span>
  );
}

export function Tag({ children }: { children: ReactNode }) {
  return <span className="tag">{children}</span>;
}

/* ── Buttons ──────────────────────────────────────────────────────── */

export function Button({
  children,
  variant = "default",
  size,
  disabled,
  onClick,
  type = "button",
  className = "",
  title,
}: {
  children: ReactNode;
  variant?: "default" | "primary" | "ghost" | "danger";
  size?: "sm" | "lg";
  disabled?: boolean;
  onClick?: () => void;
  type?: "button" | "submit";
  className?: string;
  title?: string;
}) {
  const v = variant === "default" ? "" : ` btn-${variant}`;
  const s = size ? ` btn-${size}` : "";
  return (
    <button
      type={type}
      className={`btn${v}${s} ${className}`}
      disabled={disabled}
      onClick={onClick}
      title={title}
    >
      {children}
    </button>
  );
}

/* ── Tabs ─────────────────────────────────────────────────────────── */

export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: Array<{ id: string; label: string; count?: number }>;
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={active === t.id}
          className={`tab ${active === t.id ? "active" : ""}`}
          onClick={() => onChange(t.id)}
        >
          {t.label}
          {t.count !== undefined && <span className="tab-count">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}

/* ── Stage bar (Understanding → Evidence → Formulation → Plan → Safety) ── */

const STAGES = ["understanding", "evidence", "formulation", "plan", "safety"] as const;
const STAGE_ORDER: Record<string, number> = {
  understanding: 0,
  evidence: 1,
  formulation: 2,
  plan: 3,
  safety: 4,
};

export function StageBar({
  current,
  locked = false,
}: {
  current: string;
  locked?: boolean;
}) {
  const labels: Record<string, string> = {
    understanding: "Understanding",
    evidence: "Evidence",
    formulation: "Formulation",
    plan: "Plan",
    safety: "Safety",
  };
  const currentIdx = STAGE_ORDER[current] ?? 0;
  return (
    <div className="stage-bar" aria-label="Case stage">
      {STAGES.map((stage, i) => {
        const done = locked ? false : i < currentIdx;
        const active = i === currentIdx;
        return (
          <div className="stage" key={stage}>
            <span
              className={`stage-dot ${done ? "done" : ""} ${active ? "active" : ""} ${active && locked ? "locked" : ""}`}
            >
              {done ? <CheckIcon /> : i + 1}
            </span>
            <span className="stage-label">{labels[stage]}</span>
            {i < STAGES.length - 1 && <span className="stage-link" />}
          </div>
        );
      })}
    </div>
  );
}

/* ── Meter ────────────────────────────────────────────────────────── */

export function Meter({
  value,
  label,
  showValue = true,
}: {
  value: number; // 0-100
  label?: string;
  showValue?: boolean;
}) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className="meter">
      {label && (
        <div className="meter-head">
          <span>{label}</span>
          {showValue && <b>{Math.round(clamped)}%</b>}
        </div>
      )}
      <div className="meter-track" role="progressbar" aria-valuenow={clamped} aria-valuemin={0} aria-valuemax={100}>
        <div className="meter-fill" style={{ width: `${clamped}%` }} />
      </div>
    </div>
  );
}

/* ── Confidence ring ──────────────────────────────────────────────── */

export function Ring({
  value,
  size = 56,
  stroke = 5,
  caption = "confidence",
}: {
  value: number; // 0-1
  size?: number;
  stroke?: number;
  caption?: string;
}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value));
  const offset = c * (1 - pct);
  return (
    <div className="ring" style={{ width: size, height: size }} role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={1}>
      <svg width={size} height={size}>
        <circle className="ring-track" cx={size / 2} cy={size / 2} r={r} strokeWidth={stroke} fill="none" />
        <circle
          className="ring-fill"
          cx={size / 2}
          cy={size / 2}
          r={r}
          strokeWidth={stroke}
          fill="none"
          strokeDasharray={c}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="ring-label">
        <span className="ring-value">{Math.round(pct * 100)}</span>
        <span className="ring-caption">{caption}</span>
      </div>
    </div>
  );
}

/* ── Stepper ──────────────────────────────────────────────────────── */

export function Stepper({
  steps,
  current,
}: {
  steps: Array<{ id: string; title: string; desc?: string }>;
  current: string;
}) {
  const currentIdx = steps.findIndex((s) => s.id === current);
  return (
    <div className="stepper">
      {steps.map((step, i) => {
        const done = i < currentIdx;
        const active = i === currentIdx;
        return (
          <div key={step.id} className={`step ${done ? "done" : ""} ${active ? "active" : ""}`}>
            <span className="step-dot">{done ? <CheckIcon /> : i + 1}</span>
            <div className="step-body">
              <div className="step-title">{step.title}</div>
              {step.desc && <div className="step-desc">{step.desc}</div>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Progress segments ────────────────────────────────────────────── */

export function ProgressSegments({
  value,
  segments = 5,
}: {
  value: number; // 0-100
  segments?: number;
}) {
  const filled = Math.round((value / 100) * segments);
  return (
    <div className="progress-stage" role="progressbar" aria-valuenow={value} aria-valuemin={0} aria-valuemax={100}>
      {Array.from({ length: segments }).map((_, i) => (
        <div key={i} className={`progress-stage-seg ${i < filled ? "filled" : ""}`} />
      ))}
    </div>
  );
}

/* ── States ───────────────────────────────────────────────────────── */

export function Skeleton({ width, height = 14 }: { width: number | string; height?: number }) {
  return <div className="skeleton" style={{ width, height }} />;
}

export function EmptyState({
  icon,
  title,
  children,
}: {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty-state">
      {icon}
      <h3>{title}</h3>
      {children && <p>{children}</p>}
    </div>
  );
}

export function ErrorState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="error-state">
      <AlertCircleIcon />
      <div>
        <div className="error-title">{title}</div>
        {detail && <p>{detail}</p>}
      </div>
    </div>
  );
}

/* ── Alert ────────────────────────────────────────────────────────── */

export function Alert({
  tone,
  title,
  children,
}: {
  tone: "warning" | "danger" | "info";
  title: string;
  children?: ReactNode;
}) {
  const Icon = tone === "warning" ? AlertTriangleIcon : tone === "danger" ? AlertCircleIcon : InfoIcon;
  return (
    <div className={`alert alert-${tone}`}>
      <Icon />
      <div>
        <div className="alert-title">{title}</div>
        {children && <div className="alert-desc">{children}</div>}
      </div>
    </div>
  );
}

/* ── Switch ───────────────────────────────────────────────────────── */

export function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="switch">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        aria-label={label}
      />
      <span className="track" />
      <span className="thumb" />
    </label>
  );
}

/* ── Toast system ─────────────────────────────────────────────────── */

type ToastTone = "success" | "error" | "info";
interface ToastData {
  id: number;
  tone: ToastTone;
  title: string;
  desc?: string;
}

const ToastContext = createContext<{ push: (tone: ToastTone, title: string, desc?: string) => void }>({
  push: () => {},
});

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastData[]>([]);
  const idRef = useRef(0);

  const push = useCallback((tone: ToastTone, title: string, desc?: string) => {
    const id = ++idRef.current;
    setToasts((t) => [...t, { id, tone, title, desc }]);
    window.setTimeout(() => {
      setToasts((t) => t.filter((x) => x.id !== id));
    }, 4200);
  }, []);

  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      <div className="toast-viewport" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.tone}`}>
            <div>
              <div className="toast-title">{t.title}</div>
              {t.desc && <div className="toast-desc">{t.desc}</div>}
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

/* ── Command palette (⌘K) ─────────────────────────────────────────── */

export interface PaletteAction {
  id: string;
  label: string;
  hint?: string;
  keywords?: string;
  icon?: ReactNode;
  action: () => void;
}

export function useCommandPalette() {
  return useContext(CommandContext);
}

const CommandContext = createContext<{ open: (items: PaletteAction[]) => void }>({
  open: () => {},
});

export function CommandPaletteProvider({
  children,
  onDefaultOpen,
}: {
  children: ReactNode;
  onDefaultOpen?: () => PaletteAction[];
}) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<PaletteAction[]>([]);
  const [query, setQuery] = useState("");
  const [hl, setHl] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const openPalette = useCallback((next: PaletteAction[]) => {
    setItems(next);
    setQuery("");
    setHl(0);
    setOpen(true);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (open) {
          setOpen(false);
        } else {
          openPalette(onDefaultOpen ? onDefaultOpen() : []);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, openPalette, onDefaultOpen]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  if (!open) return <CommandContext.Provider value={{ open: openPalette }}>{children}</CommandContext.Provider>;

  const q = query.trim().toLowerCase();
  const filtered = items.filter(
    (i) =>
      !q ||
      i.label.toLowerCase().includes(q) ||
      (i.keywords ?? "").toLowerCase().includes(q),
  );

  return (
    <CommandContext.Provider value={{ open: openPalette }}>
      {children}
      <div
        className="palette-overlay"
        onClick={() => setOpen(false)}
        onKeyDown={(e) => {
          if (e.key === "Escape") setOpen(false);
        }}
      >
        <div className="palette" onClick={(e) => e.stopPropagation()}>
          <div className="palette-input-row">
            <SearchIcon />
            <input
              ref={inputRef}
              className="palette-input"
              placeholder="Search or jump to…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setHl(0);
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setHl((h) => Math.min(h + 1, filtered.length - 1));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setHl((h) => Math.max(h - 1, 0));
                } else if (e.key === "Enter" && filtered[hl]) {
                  filtered[hl].action();
                  setOpen(false);
                } else if (e.key === "Escape") {
                  setOpen(false);
                }
              }}
            />
          </div>
          <div className="palette-list">
            {filtered.length === 0 && <div className="palette-empty">No matches for “{query}”</div>}
            {filtered.map((item, i) => (
              <button
                key={item.id}
                className={`palette-item ${i === hl ? "hl" : ""}`}
                onMouseEnter={() => setHl(i)}
                onClick={() => {
                  item.action();
                  setOpen(false);
                }}
              >
                {item.icon}
                <span className="palette-label">{item.label}</span>
                {item.hint && <span className="palette-hint">{item.hint}</span>}
              </button>
            ))}
          </div>
          <div className="palette-footer">
            <span>
              <span className="kbd">↑</span> <span className="kbd">↓</span> navigate
            </span>
            <span>
              <span className="kbd">↵</span> select
            </span>
            <span>
              <span className="kbd">esc</span> close
            </span>
          </div>
        </div>
      </div>
    </CommandContext.Provider>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <span className="kbd">{children}</span>;
}

export function CommandHint() {
  return (
    <span className="palette-hint">
      <CommandIcon style={{ width: 13, height: 13, verticalAlign: "middle" }} />{" "}
      <span className="kbd">⌘K</span>
    </span>
  );
}
