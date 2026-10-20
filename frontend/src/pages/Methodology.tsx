import { useEffect, useState } from "react";
import { Badge, Card, Notice } from "../components/ui";
import { api } from "../api";
import { useApp } from "../store";

const Box = ({ t, s, tone = "" }: { t: string; s: string; tone?: string }) => (
  <div className={`flex-1 rounded-lg border px-3 py-2.5 text-center ${tone || "border-ink-500 bg-ink-900"}`}><div className="text-sm font-medium text-slate-100">{t}</div><div className="mt-0.5 text-[11px] text-slate-400">{s}</div></div>);
const Arrow = () => <div className="hidden self-center text-slate-500 lg:block">→</div>;

export default function Methodology() {
  const { config, metrics } = useApp();
  const deepNotReady = (metrics?.registry ?? []).filter((m) => ["lstm", "gru", "st_gcn_gru"].includes(m.key) && m.status !== "ready");
  const [ds, setDs] = useState<any>();
  useEffect(() => { api.dataset().then(setDs).catch(() => {}); }, []);
  const c = config?.congestion;
  return (
    <div className="space-y-4">
      <Card title="System architecture">
        <div className="space-y-3">
          <div className="flex flex-col gap-2 lg:flex-row"><Box t="Probe-count dataset" s="hourly, 24,938 Delhi road segments" /><Arrow /><Box t="Preprocessing" s="validation · chronological split · lag/rolling features" /><Arrow /><Box t="Model training" s="HA · LR · RF · XGBoost (CPU) · LSTM/GRU/ST-GCN-GRU (GPU script)" /><Arrow /><Box t="Saved artifacts" s="models/trained + segment stats" tone="border-accent/40 bg-accent/5" /></div>
          <div className="flex flex-col gap-2 lg:flex-row"><Box t="FastAPI backend" s="loads artifacts, never retrains" tone="border-accent/40 bg-accent/5" /><Arrow /><Box t="Forecast → Congestion Index" s="probe → speed → CI → level" /><Arrow /><Box t="TDSP router" s="arrival-time-dependent edge cost" /><Arrow /><Box t="Departure planner" s="re-route at +0/15/30/45 min" /></div>
          <div className="flex flex-col gap-2 lg:flex-row"><Box t="SQLite" s="events · weather cache · query log" /><Arrow /><Box t="Open-Meteo" s="weather context (no key)" /><Arrow /><Box t="React + Leaflet UI" s="OpenStreetMap tiles, Recharts" /></div>
        </div>
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Data pipeline">
          <ol className="list-decimal space-y-1.5 pl-5 text-sm text-slate-300">
            <li>Load segments + hourly probe counts; map columns through <code>configs/dataset.json</code>.</li>
            <li>Validate: duplicates, missing, negatives, implausible values, speed-limit range (statistics in Models page).</li>
            <li>Chronological split: train / validation / test by date, never shuffled across time.</li>
            <li>Features: hour, weekday, weekend, peak flag, road attributes, lag-24 h, rolling mean of hours t-24…t-26, per-segment <b>train-only</b> mean.</li>
            <li>Identical feature code at training and inference (<code>FeatureBuilder</code>).</li>
          </ol>
          <p className="mt-3 text-xs text-slate-500">Every input is at least 24 h older than its target, so forecasts need no live feed and do not leak the target.</p>
        </Card>
        <Card title="Models">
          <ul className="space-y-1.5 text-sm text-slate-300">{config && null}
            <li><b>Historical Average</b>: per-segment mean for each hour of day.</li><li><b>Linear Regression</b>: OLS baseline.</li>
            <li><b>Random Forest</b> (primary) and <b>XGBoost</b> (extension).</li>
            <li><b>LSTM</b> (primary), <b>GRU</b>, <b>ST-GCN-GRU</b>: sequence models over the 24 h window ending 24 h before the target; ST-GCN-GRU applies a 2-hop graph convolution over road connectivity at each step, then a GRU. Provided as <code>scripts/train_deep.py</code> for GPU; <Badge tone="warn">not yet trained or evaluated</Badge></li>
          </ul>
        </Card>
        <Card title="Congestion Index">
          <pre className="overflow-x-auto rounded-lg bg-ink-900 p-3 text-xs leading-relaxed text-teal-200">{`intensity = clip(predicted_probes / saturation_ref, 0, ${c?.intensity_cap ?? "…"})
speed     = max(free_flow × (1 − ${c?.k_speed_loss ?? "…"} × intensity), ${c?.min_speed_kmh ?? "…"} km/h)
CI        = 1 − speed / free_flow          (0 … 1)
Low < ${c?.thresholds.LOW} ≤ Moderate < ${c?.thresholds.MODERATE} ≤ High < ${c?.thresholds.HIGH} ≤ Severe`}</pre>
          <p className="mt-2 text-xs text-slate-400">free_flow = posted speed limit; saturation_ref = the segment's own P{Math.round((c?.saturation_quantile ?? 0.9) * 100)} probe count over the training period. All constants live in <code>configs/congestion.json</code>.</p>
          <Notice tone="warn"><b>Uncalibrated heuristic:</b> the dataset has no measured speeds, so k and the thresholds could not be fitted. Treat speeds, minutes and CI levels as <i>relative</i> indicators, not measured values.</Notice>
        </Card>
        <Card title="Time-Dependent Shortest Path (TDSP)">
          <p className="text-sm leading-relaxed text-slate-300">Edge cost is not a fixed weight. A vehicle that enters an edge at time <i>t</i> travels at the forecast speed of the hour it is in, rolling into the next hour's speed if the edge crosses an hour boundary. Dijkstra settles junctions in order of <b>earliest arrival time</b>, which is exactly optimal because this cost function is FIFO. The shortest-distance route is then driven through the same forecast so both are compared fairly.</p>
          <p className="mt-2 text-xs text-slate-500">Verified in tests against brute-force enumeration and a hand-computed hour-boundary case.</p>
        </Card>
      </div>
      <Card title="Known limitations">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-300">
          {(ds?.limitations ?? []).map((l: string) => <li key={l}>{l}</li>)}
          <li>Not live: forecasts replay the dataset window; OpenStreetMap is a map source, not a traffic feed.</li>
          <li>The road graph is built from the dataset's segments (not downloaded with OSMnx) because observations are keyed to those segments.</li>
          <li>Event context is a curated list with assumed impact radii; weather does not influence the deployed models.</li>
          {deepNotReady.length > 0 && <li>Not ready in this deployment: {deepNotReady.map((m) => `${m.label} (${m.status})`).join(", ")}. The architectures are implemented in <code>scripts/train_deep.py</code>; see the README for the training command. Their status above is read from the backend, not assumed.</li>}
        </ul>
      </Card>
    </div>
  );
}
