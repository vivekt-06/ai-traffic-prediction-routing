import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import Controls from "../components/Controls";
import type { PickMode } from "../components/MapView";
import { Badge, Card, Empty, ErrorBox, LevelBadge, Notice, Spinner, chartTheme, clock, fmt } from "../components/ui";
import { useApp } from "../store";

export default function Planner() {
  const a = useApp(); const p = a.planner;
  const [pick, setPick] = useState<PickMode>(null);
  const rec = p?.recommendation;
  const rows = p?.scenarios ?? [];
  const now = rows.find((s) => s.offset_min === 0 && s.available);
  const chart = rows.filter((s) => s.available).map((s) => ({ label: s.label, duration: Number(s.duration_min!.toFixed(1)), best: s.offset_min === rec?.best_offset_min }));
  return (
    <div className="space-y-4">
      <Card title="Trip" subtitle="“Now” is the departure time you select (the app replays historical data, there is no live clock)">
        <Controls pick={pick} setPick={setPick} compact />{a.runError && <div className="mt-3"><ErrorBox>{a.runError}</ErrorBox></div>}
      </Card>
      {a.running && <Spinner label="Recomputing the route for each departure time…" />}
      {!p && !a.running && <Empty title="No departure analysis yet">Press compute. Each scenario re-runs the TDSP search with its own departure time.</Empty>}
      {p && (<>
        <Card title="Recommendation" right={<Badge tone="info">{a.metrics?.registry.find((m) => m.key === p.model)?.label}</Badge>}>
          <div className="flex flex-wrap items-center gap-4">
            <div className="rounded-xl border border-accent/40 bg-accent/10 px-5 py-3 text-center">
              <div className="label">Best departure</div>
              <div className="text-3xl font-semibold text-accent">{rec?.best_offset_min === null || rec?.best_offset_min === undefined ? "n/a" : rec.best_offset_min === 0 ? "Now" : `+${rec.best_offset_min} min`}</div>
            </div>
            <p className="max-w-2xl flex-1 text-sm leading-relaxed text-slate-300">{rec?.explanation}</p>
          </div>
        </Card>
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Predicted travel time by departure time" subtitle="TDSP route recomputed for each departure">
            {chart.length ? (<ResponsiveContainer width="100%" height={280}><BarChart data={chart} margin={{ top: 22 }}>
              <CartesianGrid stroke={chartTheme.grid} vertical={false} /><XAxis dataKey="label" stroke={chartTheme.axis} fontSize={12} />
              <YAxis stroke={chartTheme.axis} fontSize={11} unit=" min" width={60} domain={[0, "auto"]} />
              <Tooltip contentStyle={{ background: chartTheme.tooltipBg, border: `1px solid ${chartTheme.tooltipBorder}`, borderRadius: 8, fontSize: 12 }} cursor={{ fill: "#ffffff08" }} formatter={(v: any) => [`${v} min`, "Travel time"]} />
              <Bar dataKey="duration" radius={[6, 6, 0, 0]} isAnimationActive={false}>{chart.map((c, i) => <Cell key={i} fill={c.best ? "#2dd4bf" : "#4f8df7"} />)}
                <LabelList dataKey="duration" position="top" fill="#cbd5e1" fontSize={12} /></Bar></BarChart></ResponsiveContainer>) : <Empty title="No scenario could be evaluated" />}
            <p className="mt-2 text-[11px] text-slate-500">Highlighted bar = shortest predicted trip.</p>
          </Card>
          <Card title="Scenario comparison">
            <div className="overflow-x-auto"><table className="w-full"><thead><tr><th className="th">Depart</th><th className="th">Travel</th><th className="th">Δ vs now</th><th className="th">Arrive</th><th className="th">Congestion</th></tr></thead>
              <tbody>{rows.map((s) => (
                <tr key={s.offset_min} className={`border-t border-ink-600 ${s.offset_min === rec?.best_offset_min ? "bg-accent/5" : ""}`}>
                  <td className="td">{s.label}<div className="text-[11px] text-slate-500">{clock(s.departure)}</div></td>
                  {s.available ? (<>
                    <td className="td tabular-nums">{fmt(s.duration_min, 1)} min</td>
                    <td className="td tabular-nums">{now && s.offset_min !== 0 ? `${s.duration_min! - now.duration_min! > 0 ? "+" : ""}${fmt(s.duration_min! - now.duration_min!, 1)} min` : "—"}</td>
                    <td className="td tabular-nums">{clock(s.arrival)}</td><td className="td"><LevelBadge level={s.level} /></td></>)
                    : <td className="td text-xs text-slate-400" colSpan={4}>Forecast unavailable: {s.reason}</td>}
                </tr>))}</tbody></table></div>
          </Card>
        </div>
        <Notice>{p.note} The best option minimises <b>driving time</b>; arrival times are shown so the cost of waiting is visible.</Notice>
      </>)}
    </div>
  );
}
