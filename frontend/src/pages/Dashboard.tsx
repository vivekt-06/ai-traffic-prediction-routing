import { useEffect, useState } from "react";
import { api } from "../api";
import Controls from "../components/Controls";
import MapSourceToggle from "../components/MapSourceToggle";
import MapView, { type PickMode } from "../components/MapView";
import { Badge, Card, Empty, ErrorBox, Legend, LevelBadge, Notice, Spinner, Stat, clock, dayTime, fmt, fmtMin } from "../components/ui";
import { useApp } from "../store";

export default function Dashboard({ goto }: { goto: (p: string) => void }) {
  const a = useApp();
  const [pick, setPick] = useState<PickMode>(null);
  const [ds, setDs] = useState<any>();
  const onPick = (lat: number, lon: number) => {
    const p = { lat, lon }; if (pick === "origin") a.setOrigin(p); else if (pick === "destination") a.setDestination(p); setPick(null);
  };
  const m = a.metrics?.registry.find((r) => r.key === a.model);
  useEffect(() => { api.dataset().then(setDs).catch(() => setDs({ error: true })); }, []);
  const r = a.route, p = a.planner, best = p?.scenarios.find((s) => s.offset_min === p.recommendation.best_offset_min);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Card title="Dataset" right={<Badge tone="warn">Historical replay</Badge>}>
          {ds && !ds.error ? (<div className="space-y-1 text-sm">
            <div className="font-medium text-slate-100">{ds.name}</div>
            <div className="text-xs text-slate-400">{ds.period_start.slice(0, 10)} → {ds.period_end.slice(0, 10)} · {ds.resolution_minutes}-min resolution</div>
            <div className="text-xs text-slate-400">{ds.records.toLocaleString()} records · {ds.segments.toLocaleString()} segments</div>
            <div className="text-xs text-slate-500">Target: {ds.target_kind.replace("_", " ")} (no measured speeds)</div>
          </div>) : ds?.error ? <span className="text-xs text-slate-400">Unavailable</span> : <Spinner />}
        </Card>
        <Card title="Forecast model" right={m && <Badge tone={m.role === "primary" ? "info" : "neutral"}>{m.role}</Badge>}>
          {m?.metrics ? (<div className="space-y-1 text-sm">
            <div className="font-medium text-slate-100">{m.label}</div>
            <div className="text-xs text-slate-400">Test MAE {fmt(m.metrics.test.MAE)} · RMSE {fmt(m.metrics.test.RMSE)} probes/h</div>
            <div className="text-xs text-slate-400">R² {fmt(m.metrics.test.R2, 3)} · trained {m.metrics.trained_at.slice(0, 10)}</div>
            <button className="text-xs text-accent hover:underline" onClick={() => goto("models")}>Compare all models →</button>
          </div>) : <Spinner />}
        </Card>
        <Card title={a.mapSource === "predicted" ? "Predicted network congestion" : "Observed network congestion"} subtitle={a.mapHour.replace("T", " ")}>
          {a.summary ? (<div className="space-y-1"><div className="text-2xl font-semibold tabular-nums">{fmt(a.summary.mean_ci, 2)}<span className="ml-1 text-xs font-normal text-slate-400">mean CI</span></div>
            <div className="text-xs text-slate-400">{fmt(a.summary.share_high_or_severe_pct, 1)}% of {a.summary.n_segments.toLocaleString()} segments High or Severe</div></div>) : <Spinner />}
        </Card>
        <Card title="Data provenance">
          <p className="text-xs leading-relaxed text-slate-400">Forecasts are produced by the saved model from <b className="text-slate-200">observed probe counts ≥ 24 h earlier</b>. No live traffic feed is connected; speeds are a derived estimate.</p>
        </Card>
      </div>

      <Card title="Trip setup" subtitle="Choose a trip inside the dataset's coverage area and time window">
        <Controls pick={pick} setPick={setPick} compact />
        {a.runError && <div className="mt-3"><ErrorBox title="Could not compute the trip">{a.runError}</ErrorBox></div>}
      </Card>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Recommended travel time" value={r ? fmtMin(r.tdsp.duration_min) : "—"} sub={r ? `Arrive ${clock(r.tdsp.arrival)} · ${fmt(r.tdsp.distance_km, 1)} km (TDSP)` : "Compute a trip"} />
        <Stat label="Time saved vs shortest route" value={r ? fmtMin(r.difference.time_saved_min) : "—"} tone={r && r.difference.time_saved_min > 0.05 ? "good" : "neutral"}
          sub={r ? (r.difference.same_path ? "Both algorithms choose the same road path" : `${r.difference.extra_distance_km >= 0 ? "+" : ""}${fmt(r.difference.extra_distance_km, 2)} km vs shortest`) : undefined} />
        <Stat label="Best departure time" value={best ? (best.offset_min === 0 ? "Now" : `+${best.offset_min} min`) : p ? "n/a" : "—"} tone="good"
          sub={p ? (best ? `Leave ${clock(best.departure)}${p.recommendation.negligible ? " (difference negligible)" : ""}` : "No forecast available") : undefined} />
        <Stat label="Route congestion exposure" value={r ? <LevelBadge level={r.tdsp.congestion_exposure.mean_level} /> : "—"}
          sub={r ? `CI ${fmt(r.tdsp.congestion_exposure.mean_ci, 2)} · ${fmt(r.tdsp.congestion_exposure.minutes_in_high_or_severe, 1)} min in High/Severe` : undefined} />
      </div>

      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card title={a.mapSource === "predicted" ? "Predicted congestion map" : "Observed congestion map (historical)"}
          subtitle={a.mapSource === "predicted" ? `Model forecast for ${a.mapHour.replace("T", " ")} · segments coloured by Congestion Index` : `Observed historical probe counts of ${a.mapHour.replace("T", " ")} (dataset replay) · coloured by the same Congestion Index heuristic`}
          right={<div className="flex flex-wrap items-center justify-end gap-x-4 gap-y-2"><MapSourceToggle /><Legend /></div>}>
          <MapView height={600} pick={pick} onPick={onPick} />
          {pick && <p className="mt-2 text-xs text-accent">Click the map to set the {pick}.</p>}
          <p className="mt-2 text-[11px] text-slate-500">Dashed blue = shortest distance · outlined teal = congestion-aware TDSP route (coloured by predicted congestion). Hover or click a road for details. Zoom in for minor roads. OpenStreetMap is only the basemap; no live traffic source is connected.</p>
        </Card>
        <Card title="Route comparison">
          {a.running ? <Spinner label="Running TDSP…" /> : r ? (
            <div className="space-y-3 text-sm">
              {[r.tdsp, r.shortest].map((x, i) => (
                <div key={i} className={`rounded-lg border p-3 ${i === 0 ? "border-accent/40" : "border-ink-500"}`}>
                  <div className="mb-1 flex items-center justify-between"><b>{x.label}</b><LevelBadge level={x.congestion_exposure.mean_level} /></div>
                  <div className="grid grid-cols-3 gap-2 text-xs text-slate-400">
                    <div><div className="label">Time</div><div className="text-sm text-slate-100">{fmtMin(x.duration_min)}</div></div>
                    <div><div className="label">Distance</div><div className="text-sm text-slate-100">{fmt(x.distance_km, 1)} km</div></div>
                    <div><div className="label">Arrive</div><div className="text-sm text-slate-100">{clock(x.arrival)}</div></div>
                  </div>
                </div>))}
              <Notice>Model: <b>{r.model_label}</b> · departure hour is in the <b>{r.split_of_departure_hour}</b> period{r.split_of_departure_hour !== "test" ? " (in-sample for training; use test days for held-out evidence)" : " (held-out)"}.</Notice>
              <button className="btn-ghost w-full" onClick={() => goto("routes")}>Open route details</button>
            </div>
          ) : <Empty title="No route yet">Press “Compute routes & departure plan”. Results are calculated by the backend from the saved model forecasts. {p && dayTime(p.scenarios[0].departure)}</Empty>}
        </Card>
      </div>
    </div>
  );
}
