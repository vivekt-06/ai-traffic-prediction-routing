import L from "leaflet";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useApp } from "../store";
import type { MapLayerResponse, MapSegment } from "../types";
import { levelColor } from "./ui";

// Basemap = OpenStreetMap standard tiles (no API key). The provider is configurable (see frontend/.env.example):
//   VITE_TILE_URL          e.g. https://tile.openstreetmap.org/{z}/{x}/{y}.png  (default)
//   VITE_TILE_ATTRIBUTION  HTML attribution string required by the chosen provider (default: OSM contributors, from /api/config)
// OSM tiles are a basemap only - they carry no traffic or routing data. Respect the tile usage policy for anything beyond a demo.
const TILE_URL = (import.meta.env.VITE_TILE_URL as string | undefined) || "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const TILE_ATTRIBUTION = import.meta.env.VITE_TILE_ATTRIBUTION as string | undefined;
const maxFrcFor = (z: number) => (z <= 11 ? 2 : z === 12 ? 3 : z === 13 ? 4 : z === 14 ? 5 : 6);

const ESC: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (v: unknown) => String(v).replace(/[&<>"']/g, (c) => ESC[c]);

export type PickMode = "origin" | "destination" | null;

function segmentPopup(s: MapSegment, r: MapLayerResponse) {
  const kind = r.source === "predicted" ? "Predicted probe count" : "Observed probe count (historical)";
  const src = r.source === "predicted" ? `Forecast · ${esc(r.model ?? "")} · ${esc(r.split)} period` : `Observed historical data · ${esc(r.split)} period`;
  return `<div style="font-size:12px;line-height:1.45"><b>${esc(s.street)}</b><br/>` +
    `<span style="color:${esc(levelColor(s.level))}">&#9679;</span> ${esc(s.level.charAt(0) + s.level.slice(1).toLowerCase())} · CI ${esc(s.ci)}<br/>` +
    `${esc(s.speed)} km/h <span style="opacity:.65">(heuristic estimate)</span><br/>${kind}: ${esc(s.count)} /h<br/>` +
    `<span style="opacity:.65">${src}<br/>${esc(r.hour)} · segment #${esc(s.id)}</span></div>`;
}

export default function MapView({ height = 560, showCongestion = true, routeMode = "both", pick, onPick }: {
  height?: number; showCongestion?: boolean; routeMode?: "both" | "tdsp" | "shortest" | "none"; pick?: PickMode; onPick?: (lat: number, lon: number) => void;
}) {
  const { config, metrics, mapHour, model, mapSource, origin, destination, route, setSummary } = useApp();
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map>();
  const congLayer = useRef(L.layerGroup());
  const routeLayer = useRef(L.layerGroup());
  const pickRef = useRef<{ pick?: PickMode; onPick?: (a: number, b: number) => void }>({});
  const lastKey = useRef(""); const abort = useRef<AbortController>(); const timer = useRef<number>();
  const [tileFail, setTileFail] = useState(false);
  const [layerError, setLayerError] = useState<string>();
  pickRef.current = { pick, onPick };
  const colors = config?.congestion.colors;

  useEffect(() => {
    const node = el.current;
    if (!node || map.current || !config) return;
    const m = L.map(node, { center: config.map.center, zoom: 12, preferCanvas: true, zoomControl: true });
    const tiles = L.tileLayer(TILE_URL, { maxZoom: 19, attribution: TILE_ATTRIBUTION || config.map.attribution }).addTo(m);
    tiles.on("tileerror", () => setTileFail(true));
    tiles.on("tileload", () => setTileFail(false));
    L.control.scale({ imperial: false }).addTo(m);
    congLayer.current.addTo(m); routeLayer.current.addTo(m);
    m.on("click", (e: L.LeafletMouseEvent) => { const p = pickRef.current; if (p.pick && p.onPick) p.onPick(e.latlng.lat, e.latlng.lng); });
    const ro = new ResizeObserver(() => m.invalidateSize());   // responsive: re-measure whenever the container changes size
    ro.observe(node);
    map.current = m;
    return () => { ro.disconnect(); m.remove(); map.current = undefined; };
  }, [config]);

  useEffect(() => { const c = map.current?.getContainer(); if (c) c.style.cursor = pick ? "crosshair" : ""; }, [pick]);

  const load = useRef<() => void>();
  load.current = () => {
    const m = map.current; if (!m || !showCongestion || !model) { congLayer.current.clearLayers(); return; }
    const b = m.getBounds(), z = m.getZoom();
    const key = [b.getWest().toFixed(3), b.getSouth().toFixed(3), b.getEast().toFixed(3), b.getNorth().toFixed(3), mapHour, model, mapSource, maxFrcFor(z)].join("|");
    if (key === lastKey.current) return;                       // identical request already drawn
    lastKey.current = key; abort.current?.abort(); abort.current = new AbortController();
    api.mapSegments({ west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() }, mapHour, model, maxFrcFor(z), mapSource, abort.current.signal)
      .then((r) => {
        setLayerError(undefined);
        congLayer.current.clearLayers();
        r.segments.forEach((s) => {
          const pl = L.polyline(s.points, { color: levelColor(s.level, colors), weight: z >= 15 ? 5 : 3.5, opacity: 0.85 });
          pl.bindTooltip(`<b>${esc(s.street)}</b><br/>${esc(s.level.toLowerCase())} · CI ${esc(s.ci)} · ${esc(s.speed)} km/h`, { sticky: true });
          pl.on("click", (e: L.LeafletMouseEvent) => {           // popups are disabled while picking an origin/destination
            if (!pickRef.current.pick) L.popup({ maxWidth: 260 }).setLatLng(e.latlng).setContent(segmentPopup(s, r)).openOn(m);
          });
          pl.addTo(congLayer.current);
        });
        setSummary(r.citywide_summary);
      })
      .catch((e) => { if (e.name !== "AbortError") { lastKey.current = ""; setLayerError(e.message); console.error("map layer failed:", e.message); } });
  };

  useEffect(() => {
    const m = map.current; if (!m) return;
    const h = () => { window.clearTimeout(timer.current); timer.current = window.setTimeout(() => load.current?.(), 300); };
    m.on("moveend", h); h();
    return () => { m.off("moveend", h); window.clearTimeout(timer.current); };
  }, [config, mapHour, model, mapSource, showCongestion]);

  useEffect(() => {
    const g = routeLayer.current; g.clearLayers(); const m = map.current; if (!m) return;
    const pin = (p: { lat: number; lon: number; name?: string }, color: string, role: string) =>
      L.circleMarker([p.lat, p.lon], { radius: 8, color: "#fff", weight: 2, fillColor: color, fillOpacity: 1 })
        .bindTooltip(`${role}${p.name ? ": " + esc(p.name) : ""}`)
        .on("click", () => {
          if (!pickRef.current.pick) L.popup().setLatLng([p.lat, p.lon]).setContent(
            `<div style="font-size:12px"><b>${role}</b><br/>${p.name ? esc(p.name) + "<br/>" : ""}${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}</div>`).openOn(m);
        })
        .addTo(g);
    if (origin) pin(origin, "#2dd4bf", "Origin");
    if (destination) pin(destination, "#eb5757", "Destination");
    if (route && routeMode !== "none") {
      if (routeMode !== "tdsp") L.polyline(route.shortest.geometry, { color: "#4f8df7", weight: 5, dashArray: "2 9", lineCap: "round", opacity: 0.95 }).bindTooltip("Shortest-distance route").addTo(g);
      if (routeMode !== "shortest") {
        L.polyline(route.tdsp.geometry, { color: "#2dd4bf", weight: 11, opacity: 0.5 }).addTo(g);
        route.tdsp.segments.forEach((s) =>
          L.polyline(s.points, { color: levelColor(s.level, colors), weight: 5, opacity: 1 }).bindTooltip(`${esc(s.street)} · ${esc(s.level.toLowerCase())} (CI ${esc(s.ci)})`, { sticky: true }).addTo(g));
      }
      const pts = [...route.tdsp.geometry, ...route.shortest.geometry];
      if (pts.length) m.fitBounds(L.latLngBounds(pts as L.LatLngTuple[]), { padding: [40, 40], maxZoom: 15 });
    } else if (origin && destination) {
      m.fitBounds(L.latLngBounds([[origin.lat, origin.lon], [destination.lat, destination.lon]]), { padding: [60, 60], maxZoom: 14 });
    }
  }, [origin, destination, route, routeMode, colors]);

  const modelLabel = metrics?.registry.find((r) => r.key === model)?.label ?? model;
  const label = mapSource === "predicted" ? `Predicted · ${modelLabel}` : "Observed · historical probe counts";
  return (
    <div className="relative">
      <div ref={el} className="dark-tiles w-full overflow-hidden rounded-xl border border-ink-600" style={{ height }} role="application" aria-label="Interactive traffic map" />
      <div className="pointer-events-none absolute right-2 top-2 z-[1000] flex max-w-[75%] flex-col items-end gap-1">
        <span className="rounded-md border border-ink-500 bg-ink-900/90 px-2 py-1 text-[11px] text-slate-200">
          {label} · {mapHour.replace("T", " ")} · <span className="text-lv-moderate">dataset replay, not live traffic</span>
        </span>
        {tileFail && <span className="rounded-md border border-lv-severe/50 bg-ink-900/90 px-2 py-1 text-[11px] text-lv-severe">Basemap tiles could not be loaded (offline or blocked). Traffic overlays still work.</span>}
        {layerError && <span className="rounded-md border border-lv-severe/50 bg-ink-900/90 px-2 py-1 text-[11px] text-lv-severe">Traffic layer unavailable: {layerError}</span>}
      </div>
    </div>
  );
}
