import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend as RLegend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import Controls from "../components/Controls";
import MapSourceToggle from "../components/MapSourceToggle";
import MapView, { type PickMode } from "../components/MapView";
import { Card, Empty, ErrorBox, Legend, LevelBadge, Notice, Spinner, Stat, chartTheme, clock, fmt, fmtMin } from "../components/ui";
import { useApp } from "../store";
import type { RouteSide } from "../types";

export default function Routes() {
  const a = useApp(); const r = a.route;
  const [pick, setPick] = useState<PickMode>(null);
  const [mode, setMode] = useState<"both" | "tdsp" | "shortest">("both");
  const onPick = (lat: number, lon: number) => { const p = { lat, lon }; if (pick === "origin") a.setOrigin(p); else a.setDestination(p); setPick(null); };
  const levels = ["LOW", "MODERATE", "HIGH", "SEVERE"] as const;
  const exposure = r ? levels.map((l) => ({ level: l, TDSP: r.tdsp.congestion_exposure.km_by_level[l], Shortest: r.shortest.congestion_exposure.km_by_level[l] })) : [];

  const Side = ({ x, accent }: { x: RouteSide; accent?: boolean }) => (
    <Card title={x.label} right={<LevelBadge level={x.congestion_exposure.mean_level} />} className={accent ? "border-accent/40" : ""}>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
        {[["Travel time", fmtMin(x.duration_min)], ["Distance", `${fmt(x.distance_km, 2)} km`], ["Depart → arrive", `${clock(x.departure)} → ${clock(x.arrival)}`],
          ["Delay vs free-flow", fmtMin(x.delay_min)], ["Mean congestion index", fmt(x.congestion_exposure.mean_ci, 3)],
          ["Time in High/Severe", fmtMin(x.congestion_exposure.minutes_in_high_or_severe)], ["Distance in High/Severe", `${fmt(x.congestion_exposure.pct_distance_high_or_severe, 1)}%`], ["Road segments", x.n_segments]].map(([k, v]) => (
          <div key={k as string}><dt className="label">{k}</dt><dd className="mt-0.5 font-medium text-slate-100 tabular-nums">{v}</dd></div>))}
      </dl>
      <h4 className="label mt-4">Most congested roads on this route</h4>
      {x.top_congested_segments.length ? (
        <table className="mt-1 w-full"><tbody>{x.top_congested_segments.map((s) => (
          <tr key={s.segment_id} className="border-t border-ink-600"><td className="py-1.5 pr-2 text-sm text-slate-200">{s.street}<div className="text-[11px] text-slate-500">{s.length_m} m · entered {clock(s.entered_at)}</div></td>
            <td className="py-1.5 text-right text-xs tabular-nums text-slate-300">CI {s.congestion_index}<br />{s.speed_kmh} km/h</td><td className="py-1.5 pl-2 text-right"><LevelBadge level={s.level} /></td></tr>))}</tbody></table>
      ) : <p className="mt-1 text-xs text-slate-400">No segment above Low congestion on this route.</p>}
    </Card>);

  return (
    <div className="space-y-4">
      <Card title="Trip" subtitle="Same controls as the dashboard"><Controls pick={pick} setPick={setPick} compact />{a.runError && <div className="mt-3"><ErrorBox>{a.runError}</ErrorBox></div>}</Card>
      {a.running && <Spinner label="Running time-dependent shortest path…" />}
      {!r && !a.running && <Empty title="No route computed yet">Choose a trip and press compute. Origin/destination must lie within the dataset's coverage (Delhi core).</Empty>}
      {r && (<>
        <div className="grid gap-3 sm:grid-cols-3">
          <Stat label="Time saved by TDSP" value={fmtMin(r.difference.time_saved_min)} tone={r.difference.time_saved_min > 0.05 ? "good" : "neutral"} sub={r.difference.same_path ? "Same road path chosen by both" : "Different road paths"} />
          <Stat label="Distance difference" value={`${r.difference.extra_distance_km >= 0 ? "+" : ""}${fmt(r.difference.extra_distance_km, 2)} km`} sub="TDSP minus shortest-distance" />
          <Stat label="Forecast model" value={r.model_label} sub={`departure hour in ${r.split_of_departure_hour} period`} />
        </div>
        <Card title="Map" subtitle="Hover or click roads for details · the TDSP route is coloured by predicted congestion at the time each road is entered; background roads follow the Predicted/Observed switch" right={
          <div className="flex flex-wrap items-center justify-end gap-3"><MapSourceToggle /><div className="flex gap-1" role="group" aria-label="Route overlay">{(["both", "tdsp", "shortest"] as const).map((m) => (
            <button key={m} onClick={() => setMode(m)} aria-pressed={mode === m} className={`rounded-md px-2.5 py-1 text-xs ${mode === m ? "bg-accent text-ink-950" : "bg-ink-700 text-slate-300 hover:bg-ink-600"}`}>{m === "both" ? "Both" : m === "tdsp" ? "TDSP" : "Shortest"}</button>))}</div></div>}>
          <MapView height={520} routeMode={mode} pick={pick} onPick={onPick} />
          <div className="mt-2"><Legend /></div>
        </Card>
        <div className="grid gap-4 lg:grid-cols-2"><Side x={r.tdsp} accent /><Side x={r.shortest} /></div>
        <Card title="Congestion exposure by distance" subtitle="Kilometres driven at each predicted congestion level (level at the time each segment is entered)">
          <ResponsiveContainer width="100%" height={220}><BarChart data={exposure}><CartesianGrid stroke={chartTheme.grid} vertical={false} />
            <XAxis dataKey="level" stroke={chartTheme.axis} fontSize={12} /><YAxis stroke={chartTheme.axis} fontSize={11} unit=" km" width={56} />
            <Tooltip contentStyle={{ background: chartTheme.tooltipBg, border: `1px solid ${chartTheme.tooltipBorder}`, borderRadius: 8, fontSize: 12 }} cursor={{ fill: "#ffffff08" }} /><RLegend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="TDSP" fill="#2dd4bf" radius={[4, 4, 0, 0]} isAnimationActive={false} /><Bar dataKey="Shortest" fill="#4f8df7" radius={[4, 4, 0, 0]} isAnimationActive={false} /></BarChart></ResponsiveContainer>
        </Card>
        {r.events.length > 0 && <Notice tone="warn"><b>Events overlapping this trip:</b> {r.events.map((e) => `${e.event_name} (${e.overlap_type})`).join("; ")}. Event context is shown for awareness and does not alter the predicted speeds.</Notice>}
        <Notice>{r.notes.join(" ")} Origin snapped {r.snapped.origin.snap_m} m, destination {r.snapped.destination.snap_m} m onto the road network.</Notice>
      </>)}
    </div>
  );
}
