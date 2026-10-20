import type { ReactNode } from "react";
import { useApp } from "../store";
import type { Level, ModelCheck, ModelStatus } from "../types";

export const LEVELS: Level[] = ["LOW", "MODERATE", "HIGH", "SEVERE"];
export function levelColor(l: string, colors?: Record<string, string>) {
  return colors?.[l] ?? { LOW: "#2fbf71", MODERATE: "#f2c94c", HIGH: "#f2994a", SEVERE: "#eb5757" }[l] ?? "#8892a6";
}

export function Card({ title, subtitle, right, children, className = "" }: { title?: string; subtitle?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card fade-in ${className}`}>
      {(title || right) && (
        <header className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2 border-b border-ink-600/60 px-4 py-3">
          <div className="min-w-0"><h3 className="text-sm font-semibold text-slate-100">{title}</h3>{subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}</div>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "bad" | "neutral" }) {
  const c = tone === "good" ? "text-accent" : tone === "bad" ? "text-lv-high" : "text-slate-50";
  return (
    <div className="card fade-in p-4">
      <div className="label">{label}</div>
      <div className={`mt-1.5 text-2xl font-semibold tabular-nums ${c}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-400">{sub}</div>}
    </div>
  );
}

export function LevelBadge({ level }: { level?: string }) {
  const { config } = useApp();
  if (!level) return <span className="text-slate-500">—</span>;
  const col = levelColor(level, config?.congestion.colors);
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium" style={{ borderColor: col + "66", color: col, background: col + "14" }}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: col }} />{level.charAt(0) + level.slice(1).toLowerCase()}
    </span>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "good" | "warn" | "bad" | "neutral" | "info" }) {
  const t = { good: "border-accent/40 text-accent bg-accent/10", warn: "border-lv-moderate/40 text-lv-moderate bg-lv-moderate/10",
    bad: "border-lv-severe/40 text-lv-severe bg-lv-severe/10", neutral: "border-ink-500 text-slate-300 bg-ink-700/50", info: "border-accent-blue/40 text-accent-blue bg-accent-blue/10" }[tone];
  return <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${t}`}>{children}</span>;
}

/** Real model readiness from the backend: Ready only after weights loaded and a validation prediction passed. */
export function StatusBadge({ status, checks }: { status: ModelStatus; checks?: ModelCheck[] }) {
  const tone = status === "ready" ? "good" : status === "error" ? "bad" : "warn";
  const text = status === "ready" ? "Trained · Ready" : status === "error" ? "Error" : "Untrained";
  const tip = checks?.length ? checks.map((c) => `${c.ok ? "✓" : "✗"} ${c.name}${c.detail ? ": " + c.detail : ""}`).join("\n") : undefined;
  return <span title={tip}><Badge tone={tone}>{text}</Badge></span>;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-ink-500 border-t-accent" />{label ?? "Loading…"}
    </div>
  );
}

export function ErrorBox({ title = "Something went wrong", children }: { title?: string; children: ReactNode }) {
  return (
    <div className="rounded-lg border border-lv-severe/40 bg-lv-severe/10 px-4 py-3 text-sm" role="alert">
      <div className="font-semibold text-lv-severe">{title}</div><div className="mt-1 text-slate-300">{children}</div>
    </div>
  );
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  const c = tone === "warn" ? "border-lv-moderate/30 bg-lv-moderate/5 text-slate-300" : "border-accent-blue/30 bg-accent-blue/5 text-slate-300";
  return <div className={`rounded-lg border px-3 py-2 text-xs leading-relaxed ${c}`}>{children}</div>;
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-ink-500 px-6 py-10 text-center">
      <div className="text-sm font-medium text-slate-200">{title}</div>
      {children && <div className="mt-1 max-w-md text-xs text-slate-400">{children}</div>}
    </div>
  );
}

export function Legend() {
  const { config } = useApp();
  const t = config?.congestion.thresholds;
  const rng: Record<string, string> = t ? { LOW: `CI < ${t.LOW}`, MODERATE: `${t.LOW}–${t.MODERATE}`, HIGH: `${t.MODERATE}–${t.HIGH}`, SEVERE: `≥ ${t.HIGH}` } : {};
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-300" aria-label="Congestion legend">
      {LEVELS.map((l) => (
        <span key={l} className="inline-flex items-center gap-1.5">
          <span className="h-2 w-5 rounded-sm" style={{ background: levelColor(l, config?.congestion.colors) }} />
          {l.charAt(0) + l.slice(1).toLowerCase()} <span className="text-slate-500">({rng[l]})</span>
        </span>
      ))}
    </div>
  );
}

export const chartTheme = { grid: "#22304d", axis: "#7d8aa5", tooltipBg: "#111a2e", tooltipBorder: "#31425f" };
export const fmt = (n: number | null | undefined, d = 2) => (n === null || n === undefined || Number.isNaN(n) ? "n/a" : n.toFixed(d));
export const fmtMin = (n: number | null | undefined) => (n === null || n === undefined ? "n/a" : `${n.toFixed(1)} min`);
export const clock = (s?: string) => (s ? s.slice(11, 16) : "n/a");
export const dayTime = (s?: string) => (s ? s.slice(5, 16).replace("T", " ").replace(" ", " · ") : "n/a");
