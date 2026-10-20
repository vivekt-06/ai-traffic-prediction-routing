import type { Config, DepartureResult, MapLayerResponse, MapSource, MetricsDoc, Place, RouteResult } from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number, public kind?: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Cannot reach the backend. Start it with: uvicorn api.main:app --port 8000", 0, "network");
  }
  const text = await res.text();
  let body: any = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* non-JSON error page */ }
  if (!res.ok) throw new ApiError(body?.detail ?? `Request failed (${res.status})`, res.status, body?.kind);
  return body as T;
}

// Static endpoints are fetched once per page load (no duplicate requests).
const once = new Map<string, Promise<unknown>>();
function cached<T>(key: string, fn: () => Promise<T>): Promise<T> {
  if (!once.has(key)) once.set(key, fn().catch((e) => { once.delete(key); throw e; }));
  return once.get(key) as Promise<T>;
}

const pt = (p: Place) => ({ lat: p.lat, lon: p.lon });
const post = (path: string, body: unknown, signal?: AbortSignal) => ({ path, init: { method: "POST", body: JSON.stringify(body), signal } });

export const api = {
  health: () => request<any>("/api/health"),
  config: () => cached("config", () => request<Config>("/api/config")),
  dataset: () => cached("dataset", () => request<any>("/api/dataset/status")),
  metrics: () => cached("metrics", () => request<MetricsDoc>("/api/model-metrics")),
  events: () => cached("events", () => request<{ events: any[]; note: string }>("/api/events")),
  weather: (start: string, end: string) => request<any>(`/api/weather?start=${start}&end=${end}`),
  search: (q: string) => request<any[]>(`/api/segments/search?q=${encodeURIComponent(q)}`),
  predict: (segment_id: number, timestamp: string, model: string) =>
    request<any>("/api/predict", { method: "POST", body: JSON.stringify({ segment_id, timestamp, model: model || null }) }),
  series: (segment_id: number, model: string) => request<any>(`/api/segment-series?segment_id=${segment_id}&model=${model}`),
  route: (o: Place, d: Place, departure: string, model: string) => {
    const r = post("/api/route", { origin: pt(o), destination: pt(d), departure, model: model || null });
    return request<RouteResult>(r.path, r.init);
  },
  departure: (o: Place, d: Place, departure: string, model: string) => {
    const r = post("/api/departure-analysis", { origin: pt(o), destination: pt(d), departure, model: model || null });
    return request<DepartureResult>(r.path, r.init);
  },
  mapSegments: (b: { west: number; south: number; east: number; north: number }, timestamp: string, model: string, maxFrc: number, source: MapSource, signal: AbortSignal) =>
    request<MapLayerResponse>(
      `/api/map/segments?west=${b.west}&south=${b.south}&east=${b.east}&north=${b.north}&timestamp=${encodeURIComponent(timestamp)}&model=${encodeURIComponent(model)}&source=${source}&max_frc=${maxFrc}`,
      { signal }),
};
