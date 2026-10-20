import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend as RLegend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Badge, Card, Notice, Spinner, StatusBadge, chartTheme, fmt } from "../components/ui";
import { useApp } from "../store";

export default function Models() {
  const { metrics } = useApp();
  const [ds, setDs] = useState<any>();
  useEffect(() => { api.dataset().then(setDs).catch(() => setDs({ error: true })); }, []);
  if (!metrics) return <Spinner />;
  // "ready" = weights loaded AND a held-out validation prediction passed (decided by the backend, src/model_status.py)
  const trained = metrics.registry.filter((m) => m.status === "ready" && m.metrics);
  const bestTest = trained.length ? trained.reduce((a, b) => (a.metrics!.test.MAE! < b.metrics!.test.MAE! ? a : b)) : undefined;
  const chart = trained.map((m) => ({ name: m.label, MAE: Number(m.metrics!.test.MAE!.toFixed(2)), RMSE: Number(m.metrics!.test.RMSE!.toFixed(2)) }));
  const missing = metrics.registry.filter((m) => !(m.status === "ready" && m.metrics));
  const tip = { contentStyle: { background: chartTheme.tooltipBg, border: `1px solid ${chartTheme.tooltipBorder}`, borderRadius: 8, fontSize: 12 }, cursor: { fill: "#ffffff08" } };
  return (
    <div className="space-y-4">
      {bestTest && <Notice tone="warn"><b>Honest finding:</b> on the held-out test days, <b>{bestTest.label}</b> has the lowest MAE ({fmt(bestTest.metrics!.test.MAE)}). {bestTest.role === "baseline" ? "A simple historical-average baseline beats the learned models that are ready: the hourly pattern repeats strongly and only 12 training days exist. " : ""}{missing.length > 0 ? `Not available: ${missing.map((m) => `${m.label} (${m.status})`).join(", ")}. No claim is made about them.` : ""}</Notice>}
      <Card title="Model comparison" subtitle={`Held-out evaluation · target: ${metrics.target ?? "hourly probe count"} · model for routing is chosen by ${metrics.default_model_rule}`}>
        <div className="overflow-x-auto"><table className="w-full min-w-[900px]">
          <thead><tr><th className="th">Model</th><th className="th">Role</th><th className="th">Status</th><th className="th text-right">Val MAE</th><th className="th text-right">Test MAE</th><th className="th text-right">Test RMSE</th><th className="th text-right">Test MAPE %</th><th className="th text-right">Test R²</th><th className="th text-right">Level agreement</th><th className="th text-right">Fit (s)</th></tr></thead>
          <tbody>{trained.map((m) => { const t = m.metrics!.test; return (
            <tr key={m.key} className="border-t border-ink-600">
              <td className="td font-medium text-slate-100">{m.label}{m.key === metrics.default_model && <span className="ml-2"><Badge tone="good">used for routing</Badge></span>}</td>
              <td className="td"><Badge tone={m.role === "primary" ? "info" : "neutral"}>{m.role}</Badge></td><td className="td"><StatusBadge status={m.status} checks={m.checks} /></td>
              <td className="td text-right tabular-nums">{fmt(m.metrics!.val.MAE)}</td><td className="td text-right tabular-nums">{fmt(t.MAE)}</td><td className="td text-right tabular-nums">{fmt(t.RMSE)}</td>
              <td className="td text-right tabular-nums" title={`computed on ${fmt((t.mape_coverage ?? 0) * 100, 0)}% of rows (non-zero targets)`}>{fmt(t.MAPE, 1)}</td>
              <td className="td text-right tabular-nums">{fmt(t.R2, 3)}</td><td className="td text-right tabular-nums">{t.congestion_level_agreement !== undefined ? `${fmt(t.congestion_level_agreement * 100, 1)}%` : "n/a"}</td>
              <td className="td text-right tabular-nums">{fmt(m.metrics!.fit_seconds, 0)}</td></tr>); })}
            {missing.map((m) => (<tr key={m.key} className="border-t border-ink-600 opacity-80"><td className="td font-medium text-slate-300">{m.label}</td><td className="td"><Badge tone={m.role === "primary" ? "info" : "neutral"}>{m.role}</Badge></td>
              <td className="td"><StatusBadge status={m.status} checks={m.checks} /></td><td className={`td text-xs ${m.status === "error" ? "text-lv-severe" : "text-slate-400"}`} colSpan={7}>{m.status_detail}</td></tr>))}</tbody></table></div>
        <p className="mt-3 text-[11px] text-slate-500">Status is decided by the backend: “Trained · Ready” only after the weights loaded and a held-out validation prediction passed (hover the badge for the individual checks); “Untrained” = no weights exist; “Error” = weights exist but failed to load or validate. MAPE excludes zero-valued targets (undefined there); its coverage is shown on hover. “Level agreement” = share of segment-hours where predicted and observed counts map to the same congestion level. Generated {metrics.generated_at.slice(0, 19).replace("T", " ")} UTC.</p>
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Test error by model" subtitle="Lower is better (probes per hour)">
          <ResponsiveContainer width="100%" height={280}><BarChart data={chart}><CartesianGrid stroke={chartTheme.grid} vertical={false} /><XAxis dataKey="name" stroke={chartTheme.axis} fontSize={11} interval={0} /><YAxis stroke={chartTheme.axis} fontSize={11} width={40} />
            <Tooltip {...tip} /><RLegend wrapperStyle={{ fontSize: 12 }} /><Bar dataKey="MAE" fill="#2dd4bf" radius={[4, 4, 0, 0]} isAnimationActive={false} /><Bar dataKey="RMSE" fill="#4f8df7" radius={[4, 4, 0, 0]} isAnimationActive={false} /></BarChart></ResponsiveContainer>
        </Card>
        <Card title="Dataset & split">
          {!ds ? <Spinner /> : ds.error ? <p className="text-sm text-slate-400">Dataset status unavailable.</p> : (
            <div className="space-y-3 text-sm">
              <div><div className="font-medium text-slate-100">{ds.name}</div><div className="text-xs text-slate-400">{ds.period_start} → {ds.period_end} · {ds.days} days · {ds.records.toLocaleString()} records · {ds.segments.toLocaleString()} segments</div></div>
              <table className="w-full text-xs"><tbody>{Object.entries(ds.splits).map(([k, v]: any) => (<tr key={k} className="border-t border-ink-600"><td className="py-1.5 capitalize text-slate-300">{k}</td><td className="text-slate-400">{v.start.slice(0, 10)} → {v.end.slice(0, 10)}</td><td className="text-right tabular-nums text-slate-400">{v.hours} h</td></tr>))}</tbody></table>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400">
                {Object.entries(ds.quality).map(([k, v]: any) => <div key={k} className="flex justify-between border-b border-ink-700 py-0.5"><span>{k.replaceAll("_", " ")}</span><span className="tabular-nums text-slate-200">{String(v)}</span></div>)}</div>
              <p className="text-[11px] text-slate-500">Tree models trained on a random {metrics.train_subsample_rows.toLocaleString()}-row sample of the training period; all test rows are used for evaluation.</p>
            </div>)}
        </Card>
      </div>
    </div>
  );
}
