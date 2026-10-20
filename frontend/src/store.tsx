import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "./api";
import type { Config, DepartureResult, MapSource, MapSummary, MetricsDoc, Place, RouteResult } from "./types";

interface Ctx {
  config?: Config; metrics?: MetricsDoc; bootError?: string;
  origin?: Place; destination?: Place; setOrigin: (p?: Place) => void; setDestination: (p?: Place) => void;
  departure: string; setDeparture: (s: string) => void; model: string; setModel: (s: string) => void;
  mapOffsetH: number; setMapOffsetH: (n: number) => void;
  mapSource: MapSource; setMapSource: (s: MapSource) => void;
  route?: RouteResult; planner?: DepartureResult; running: boolean; runError?: string; run: () => Promise<void>;
  summary?: MapSummary; setSummary: (s?: MapSummary) => void;
  mapHour: string;
}
const C = createContext<Ctx>(null as unknown as Ctx);
export const useApp = () => useContext(C);

const pad = (n: number) => String(n).padStart(2, "0");
export const toLocal = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
export const fmtTime = (s: string) => s.replace("T", " ").slice(0, 16);

export function AppProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<Config>();
  const [metrics, setMetrics] = useState<MetricsDoc>();
  const [bootError, setBootError] = useState<string>();
  const [origin, setOrigin] = useState<Place>();
  const [destination, setDestination] = useState<Place>();
  const [departure, setDeparture] = useState("2024-08-28T08:30");
  const [model, setModel] = useState("");
  const [mapOffsetH, setMapOffsetH] = useState(0);
  const [mapSource, setMapSource] = useState<MapSource>("predicted");
  const [route, setRoute] = useState<RouteResult>();
  const [planner, setPlanner] = useState<DepartureResult>();
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string>();
  const [summary, setSummary] = useState<MapSummary>();

  useEffect(() => {
    Promise.all([api.config(), api.metrics()]).then(([c, m]) => {
      setConfig(c); setMetrics(m); setModel(m.default_model);
      const lm = c.landmarks;
      const a = lm.find((l) => l.name === "Connaught Place") ?? lm[0], b = lm.find((l) => l.name === "Saket") ?? lm[1];
      setOrigin(a); setDestination(b);
      setDeparture(c.splits.test[0].slice(0, 10) + "T08:30");
    }).catch((e) => setBootError(e.message));
  }, []);

  const run = useCallback(async () => {
    if (!origin || !destination) { setRunError("Choose an origin and a destination first."); return; }
    setRunning(true); setRunError(undefined);
    try {
      const [r, p] = await Promise.all([api.route(origin, destination, departure, model), api.departure(origin, destination, departure, model)]);
      setRoute(r); setPlanner(p);
    } catch (e) { setRunError((e as Error).message); setRoute(undefined); setPlanner(undefined); }
    finally { setRunning(false); }
  }, [origin, destination, departure, model]);

  const mapHour = useMemo(() => {
    const d = new Date(departure); d.setMinutes(0, 0, 0); d.setHours(d.getHours() + mapOffsetH);
    return toLocal(d);
  }, [departure, mapOffsetH]);

  const v: Ctx = { config, metrics, bootError, origin, destination, setOrigin, setDestination, departure, setDeparture, model, setModel,
    mapOffsetH, setMapOffsetH, mapSource, setMapSource, route, planner, running, runError, run, summary, setSummary, mapHour };
  return <C.Provider value={v}>{children}</C.Provider>;
}
