import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ComposedChart, Legend as RLegend, Line, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Badge, Card, Empty, ErrorBox, LevelBadge, Notice, Spinner, chartTheme, fmt } from "../components/ui";
import { useApp } from "../store";

export default function Forecast() {
  const a = useApp();
  const [q, setQ] = useState("Ring Road");
  const [hits, setHits] = useState<any[]>([]);
  const [seg, setSeg] = useState<any>();
  const [pred, setPred] = useState<any>();
  const [series, setSeries] = useState<any>();
  const [err, setErr] = useState<string>();
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (q.trim().length < 2) { setHits([]); return; }
    const h = window.setTimeout(() => api.search(q.trim()).then((r) => { setHits(r); if (!seg && r[0]) setSeg(r[0]); }).catch((e) => setErr(e.message)), 300);
    return () => window.clearTimeout(h);
  }, [q]);

  useEffect(() => {
    if (!seg || !a.model) return;
    setBusy(true); setErr(undefined);
    Promise.all([api.predict(seg.segment_id, a.departure, a.model), api.series(seg.segment_id, a.model)])
      .then(([p, s]) => { setPred(p); setSeries(s); }).catch((e) => { setErr(e.message); setPred(undefined); setSeries(undefined); }).finally(() => setBusy(false));
  }, [seg, a.departure, a.model]);

  const data = useMemo(() => series?.timestamps.map((t: string, i: number) => ({ i, t, actual: series.actual_probe[i], predicted: series.predicted_probe[i],
    speed: series.predicted_speed_kmh[i], ci: series.predicted_ci[i], obsCi: series.observed_ci[i] })) ?? [], [series]);
  const bounds = useMemo(() => {
    if (!series) return null; const s = series.split as string[];
    const f = (k: string) => [s.indexOf(k), s.lastIndexOf(k)]; return { val: f("val"), test: f("test") };
  }, [series]);
  const tick = (i: number) => series?.timestamps[i]?.slice(5, 13).replace(" ", " ") ?? "";
  const tip = { contentStyle: { background: chartTheme.tooltipBg, border: `1px solid ${chartTheme.tooltipBorder}`, borderRadius: 8, fontSize: 12 }, labelFormatter: (i: any) => series?.timestamps[i] };

  const Chart = ({ keys, unit, h = 260 }: { keys: { k: string; name: string; color: string; dash?: string }[]; unit: string; h?: number }) => (
    <ResponsiveContainer width="100%" height={h}>
      <ComposedChart data={data} margin={{ left: 0, right: 12, top: 8 }}>
        <CartesianGrid stroke={chartTheme.grid} vertical={false} />
        {bounds && <ReferenceArea x1={bounds.val[0]} x2={bounds.val[1]} fill="#f2c94c" fillOpacity={0.05} label={{ value: "validation", fill: "#a8923a", fontSize: 10, position: "insideTop" }} />}
        {bounds && <ReferenceArea x1={bounds.test[0]} x2={bounds.test[1]} fill="#2dd4bf" fillOpacity={0.07} label={{ value: "test (held-out)", fill: "#2dd4bf", fontSize: 10, position: "insideTop" }} />}
        <XAxis dataKey="i" type="number" domain={[0, "dataMax"]} tickFormatter={tick} stroke={chartTheme.axis} fontSize={11} tickCount={9} />
        <YAxis stroke={chartTheme.axis} fontSize={11} width={48} label={{ value: unit, angle: -90, position: "insideLeft", fill: chartTheme.axis, fontSize: 11 }} />
        <Tooltip {...tip} /><RLegend wrapperStyle={{ fontSize: 12 }} />
        {keys.map((x) => <Line key={x.k} dataKey={x.k} name={x.name} stroke={x.color} strokeDasharray={x.dash} dot={false} strokeWidth={1.6} isAnimationActive={false} />)}
      </ComposedChart>
    </ResponsiveContainer>
  );

  return (
    <div className="space-y-4">
      <Notice tone="warn"><b>Resolution:</b> the dataset is hourly, so forecasts are per hour (the hour containing the chosen departure time and the next three). 15- and 30-minute forecasts are not supported by this data and are not shown. “Speed” is derived from predicted probe counts (see Methodology).</Notice>
      <div className="grid gap-4 xl:grid-cols-[340px_1fr]">
        <Card title="Road segment" subtitle="Search by street name; busiest segments first">
          <input className="input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. Ring Road, Mathura Road…" aria-label="Search street" />
          <ul className="mt-3 max-h-80 space-y-1 overflow-auto pr-1">
            {hits.map((h) => (
              <li key={h.segment_id}><button onClick={() => setSeg(h)} className={`w-full rounded-lg border px-3 py-2 text-left text-sm transition hover:bg-ink-700 ${seg?.segment_id === h.segment_id ? "border-accent bg-ink-700" : "border-ink-600"}`}>
                <div className="font-medium text-slate-100">{h.street}</div><div className="text-xs text-slate-400">#{h.segment_id} · {h.length_m} m · road class {h.frc} · limit {h.speed_limit} km/h</div></button></li>))}
            {q.trim().length >= 2 && !hits.length && <li className="text-xs text-slate-400">No segments match.</li>}
          </ul>
        </Card>
        <div className="space-y-4">
          {err && <ErrorBox>{err}</ErrorBox>}
          {busy && !pred && <Spinner label="Running model…" />}
          {pred && (
            <Card title={`${pred.street} · segment #${pred.segment_id}`} subtitle={`Forecast by ${pred.model_label} for the hour containing ${a.departure.replace("T", " ")}`} right={<Badge tone="info">{pred.model_label}</Badge>}>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {pred.hourly.map((h: any) => h.available ? (
                  <div key={h.offset_hours} className="rounded-lg border border-ink-500 p-3">
                    <div className="flex items-center justify-between"><span className="label">{h.offset_hours === 0 ? "This hour" : `+${h.offset_hours} h`} · {h.hour_start.slice(11, 16)}</span><LevelBadge level={h.congestion_level} /></div>
                    <div className="mt-2 text-2xl font-semibold tabular-nums">{h.estimated_speed_kmh}<span className="ml-1 text-xs font-normal text-slate-400">km/h est.</span></div>
                    <div className="text-xs text-slate-400">CI {h.congestion_index} · free-flow {pred.free_flow_kmh} km/h</div>
                    <div className="mt-2 border-t border-ink-600 pt-2 text-xs text-slate-300">Predicted {h.predicted_probe_count} probes/h<br />Observed {h.actual_probe_count} · <span className="text-slate-400">observed level </span><LevelBadge level={h.observed_congestion_level} /></div>
                    <div className="mt-1 text-[10px] text-slate-500">{h.split === "test" ? "held-out day" : `${h.split} period (in-sample)`}</div>
                  </div>) : <div key={h.offset_hours} className="rounded-lg border border-dashed border-ink-500 p-3 text-xs text-slate-400">Not available: {h.reason}</div>)}
              </div>
            </Card>)}
          {series && (<>
            <Card title="Actual vs predicted probe count" subtitle="Whole dataset window. Shaded regions are the chronological validation and test splits."
              right={<span className="text-xs text-slate-400">Test-day error for this segment: MAE {fmt(series.test_metrics_this_segment.MAE, 1)} · MAPE {fmt(series.test_metrics_this_segment.MAPE, 1)}% · R² {fmt(series.test_metrics_this_segment.R2, 2)}</span>}>
              <Chart unit="probes/h" keys={[{ k: "actual", name: "Observed", color: "#94a3b8" }, { k: "predicted", name: `Predicted (${series.model})`, color: "#2dd4bf" }]} />
            </Card>
            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="Predicted speed (derived)"><Chart h={220} unit="km/h" keys={[{ k: "speed", name: "Predicted speed", color: "#4f8df7" }]} /></Card>
              <Card title="Congestion Index"><Chart h={220} unit="CI" keys={[{ k: "ci", name: "Predicted CI", color: "#f2994a" }, { k: "obsCi", name: "CI from observed counts", color: "#94a3b8", dash: "4 3" }]} /></Card>
            </div></>)}
          {!pred && !busy && !err && <Empty title="Select a road segment">Pick a segment to see its model forecast and observed values.</Empty>}
        </div>
      </div>
    </div>
  );
}
