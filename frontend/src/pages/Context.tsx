import { useEffect, useState } from "react";
import { CartesianGrid, ComposedChart, Bar, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Badge, Card, ErrorBox, Notice, Spinner, chartTheme } from "../components/ui";
import { useApp } from "../store";

export default function Context() {
  const a = useApp();
  const [day, setDay] = useState(a.departure.slice(0, 10));
  const [w, setW] = useState<any>(); const [ev, setEv] = useState<any>(); const [busy, setBusy] = useState(false);
  useEffect(() => { api.events().then(setEv).catch(() => setEv({ error: true })); }, []);
  useEffect(() => { setBusy(true); api.weather(day, day).then(setW).catch((e) => setW({ available: false, error: e.message })).finally(() => setBusy(false)); }, [day]);
  const dayEvents = ev?.events?.filter((e: any) => e.start_time.slice(0, 10) <= day && e.end_time.slice(0, 10) >= day) ?? [];
  const weatherModels = a.metrics?.registry.filter((m) => m.key.endsWith("_weather")) ?? [];
  return (
    <div className="space-y-4">
      <Notice tone="warn"><b>How context is used:</b> weather and events are shown as <b>context only</b>. The deployed models were not trained with weather features, so predictions are <b>not</b> adjusted for weather, and event overlap does not change speeds. Weather-aware variants exist as separate models and are only usable after training ({weatherModels.every((m) => !m.metrics) ? "currently not trained" : "see Models"}).</Notice>
      <Card title="Weather (Open-Meteo)" subtitle="Free, key-less API · Delhi city centre" right={<input type="date" className="input !w-auto" value={day} min={a.config?.time_window.start.slice(0, 10)} max={a.config?.time_window.end.slice(0, 10)} onChange={(e) => e.target.value && setDay(e.target.value)} aria-label="Weather date" />}>
        {busy && <Spinner label="Fetching weather…" />}
        {!busy && w && !w.available && <ErrorBox title="Weather unavailable">{w.error}</ErrorBox>}
        {!busy && w?.available && (<>
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-400"><Badge tone="info">{w.source}</Badge><span>fetched {w.fetched_at?.slice(0, 19).replace("T", " ")} UTC · cache {w.cache}</span>{w.warning && <Badge tone="warn">{w.warning}</Badge>}</div>
          <ResponsiveContainer width="100%" height={260}><ComposedChart data={w.hourly.map((h: any) => ({ ...h, hr: h.time.slice(11, 16) }))}>
            <CartesianGrid stroke={chartTheme.grid} vertical={false} /><XAxis dataKey="hr" stroke={chartTheme.axis} fontSize={11} /><YAxis yAxisId="t" stroke={chartTheme.axis} fontSize={11} width={40} unit="°" />
            <YAxis yAxisId="p" orientation="right" stroke={chartTheme.axis} fontSize={11} width={44} unit="mm" />
            <Tooltip contentStyle={{ background: chartTheme.tooltipBg, border: `1px solid ${chartTheme.tooltipBorder}`, borderRadius: 8, fontSize: 12 }} />
            <Bar yAxisId="p" dataKey="precipitation" name="Precipitation (mm)" fill="#4f8df7" isAnimationActive={false} /><Line yAxisId="t" dataKey="temperature_2m" name="Temperature (°C)" stroke="#f2994a" dot={false} isAnimationActive={false} /></ComposedChart></ResponsiveContainer>
          <div className="mt-3 max-h-56 overflow-auto"><table className="w-full"><thead><tr><th className="th">Time</th><th className="th">Temp °C</th><th className="th">Humidity %</th><th className="th">Rain mm</th><th className="th">Wind km/h</th></tr></thead>
            <tbody>{w.hourly.map((h: any) => <tr key={h.time} className="border-t border-ink-600"><td className="td">{h.time.slice(11, 16)}</td><td className="td tabular-nums">{h.temperature_2m ?? "n/a"}</td><td className="td tabular-nums">{h.relative_humidity_2m ?? "n/a"}</td><td className="td tabular-nums">{h.precipitation ?? "n/a"}</td><td className="td tabular-nums">{h.wind_speed_10m ?? "n/a"}</td></tr>)}</tbody></table></div></>)}
      </Card>
      <Card title="Curated events" subtitle="Manually maintained list (data/contextual/events.csv). No automatic event detection.">
        {!ev ? <Spinner /> : ev.error ? <ErrorBox>Could not load events.</ErrorBox> : (<div className="space-y-2">
          {ev.events.map((e: any) => { const active = dayEvents.includes(e); return (
            <div key={e.event_id} className={`rounded-lg border p-3 text-sm ${active ? "border-accent/50 bg-accent/5" : "border-ink-600"}`}>
              <div className="flex flex-wrap items-center gap-2"><b className="text-slate-100">{e.event_name}</b><Badge>{e.event_type.replace("_", " ")}</Badge>{active && <Badge tone="good">on selected day</Badge>}<Badge tone="warn">{e.verification_status.replaceAll("_", " ")}</Badge></div>
              <div className="mt-1 text-xs text-slate-400">{e.location} · {e.start_time} → {e.end_time} · {e.citywide ? "citywide" : `impact radius ${e.impact_radius_m} m (assumed)`}</div>
              <div className="mt-1 text-xs text-slate-500">{e.note} Source: {e.source}</div></div>); })}
          {a.route && <Notice>Events overlapping the current route: {a.route.events.length ? a.route.events.map((e) => e.event_name).join(", ") : "none"}.</Notice>}
        </div>)}
      </Card>
    </div>
  );
}
