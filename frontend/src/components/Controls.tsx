import { useApp } from "../store";
import type { Place } from "../types";
import type { PickMode } from "./MapView";

const CUSTOM = "__custom__";

function PlaceSelect({ label, value, onChange, pickActive, onPickToggle }: {
  label: string; value?: Place; onChange: (p: Place) => void; pickActive: boolean; onPickToggle: () => void }) {
  const { config } = useApp();
  const isLandmark = value?.name && config?.landmarks.some((l) => l.name === value.name);
  return (
    <div>
      <label className="label">{label}</label>
      <div className="mt-1 flex gap-2">
        <select className="input" aria-label={label} value={isLandmark ? value!.name : CUSTOM}
          onChange={(e) => { const l = config?.landmarks.find((x) => x.name === e.target.value); if (l) onChange(l); }}>
          {!isLandmark && <option value={CUSTOM}>{value ? `Map point (${value.lat.toFixed(4)}, ${value.lon.toFixed(4)})` : "Select…"}</option>}
          {config?.landmarks.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
        </select>
        <button type="button" className={pickActive ? "btn-primary" : "btn-ghost"} onClick={onPickToggle} aria-pressed={pickActive}
          title="Then click anywhere on the map to set this point">{pickActive ? "Click map…" : "Pick"}</button>
      </div>
    </div>
  );
}

export default function Controls({ pick, setPick, compact }: { pick: PickMode; setPick: (p: PickMode) => void; compact?: boolean }) {
  const a = useApp();
  const tw = a.config?.time_window;
  const servable = a.metrics?.registry.filter((m) => m.servable) ?? [];
  return (
    <div className={`grid gap-3 ${compact ? "sm:grid-cols-2 xl:grid-cols-3" : ""}`}>
      <PlaceSelect label="Origin" value={a.origin} onChange={(p) => { a.setOrigin(p); setPick(null); }} pickActive={pick === "origin"} onPickToggle={() => setPick(pick === "origin" ? null : "origin")} />
      <PlaceSelect label="Destination" value={a.destination} onChange={(p) => { a.setDestination(p); setPick(null); }} pickActive={pick === "destination"} onPickToggle={() => setPick(pick === "destination" ? null : "destination")} />
      <div>
        <label className="label" htmlFor="dep">Departure time (dataset replay clock)</label>
        <input id="dep" type="datetime-local" className="input mt-1" value={a.departure} min={tw?.start.slice(0, 16).replace(" ", "T")} max={tw?.end.slice(0, 16).replace(" ", "T")}
          onChange={(e) => e.target.value && a.setDeparture(e.target.value)} />
      </div>
      <div>
        <label className="label" htmlFor="model">Forecast model</label>
        <select id="model" className="input mt-1" value={a.model} onChange={(e) => a.setModel(e.target.value)}>
          {servable.map((m) => <option key={m.key} value={m.key}>{m.label}{m.key === a.metrics?.default_model ? " (best on validation)" : ""}</option>)}
        </select>
      </div>
      <div>
        <label className="label" htmlFor="off">Map forecast hour</label>
        <select id="off" className="input mt-1" value={a.mapOffsetH} onChange={(e) => a.setMapOffsetH(Number(e.target.value))}>
          {[0, 1, 2, 3, 4, 5, 6].map((h) => <option key={h} value={h}>{h === 0 ? "Departure hour" : `+${h} h after departure`}</option>)}
        </select>
      </div>
      <div className="flex items-end">
        <button className="btn-primary w-full" onClick={a.run} disabled={a.running || !a.origin || !a.destination}>
          {a.running ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-ink-900/30 border-t-ink-900" />Computing…</> : "Compute routes & departure plan"}
        </button>
      </div>
    </div>
  );
}
