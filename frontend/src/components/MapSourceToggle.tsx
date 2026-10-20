import { useApp } from "../store";
import type { MapSource } from "../types";

const OPTIONS: [MapSource, string, string][] = [
  ["predicted", "Predicted", "Congestion forecast by the selected model for the chosen hour"],
  ["observed", "Observed", "Congestion derived from the observed historical probe counts of that hour (dataset replay, not live)"],
];

/** Switches the map between model-predicted congestion and observed historical congestion. */
export default function MapSourceToggle() {
  const { mapSource, setMapSource } = useApp();
  return (
    <div className="flex gap-1" role="group" aria-label="Map data source">
      {OPTIONS.map(([k, label, tip]) => (
        <button key={k} type="button" title={tip} onClick={() => setMapSource(k)} aria-pressed={mapSource === k}
          className={`rounded-md px-2.5 py-1 text-xs ${mapSource === k ? "bg-accent text-ink-950" : "bg-ink-700 text-slate-300 hover:bg-ink-600"}`}>{label}</button>
      ))}
    </div>
  );
}
