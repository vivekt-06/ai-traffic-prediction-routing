import { useState, type ReactNode } from "react";
import { Badge, ErrorBox, Spinner } from "./components/ui";
import Context from "./pages/Context";
import Dashboard from "./pages/Dashboard";
import Forecast from "./pages/Forecast";
import Methodology from "./pages/Methodology";
import Models from "./pages/Models";
import Planner from "./pages/Planner";
import Routes from "./pages/Routes";
import { useApp } from "./store";

const NAV: [string, string][] = [["dashboard", "Dashboard"], ["forecast", "Traffic Forecast"], ["routes", "Route Optimization"], ["planner", "Departure Planner"],
  ["models", "Model Evaluation"], ["context", "Weather & Events"], ["method", "Methodology"]];

export default function App() {
  const [page, setPage] = useState("dashboard");
  const { config, bootError, metrics } = useApp();
  const views: Record<string, ReactNode> = { dashboard: <Dashboard goto={setPage} />, forecast: <Forecast />, routes: <Routes />, planner: <Planner />, models: <Models />, context: <Context />, method: <Methodology /> };
  return (
    <div className="flex min-h-full flex-col lg:flex-row">
      <aside className="border-b border-ink-600 bg-ink-950 lg:sticky lg:top-0 lg:h-screen lg:w-60 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="px-4 py-4"><div className="text-sm font-semibold text-slate-100">Traffic Intelligence</div><div className="text-[11px] leading-tight text-slate-400">Congestion prediction &amp; dynamic routing · Delhi</div></div>
        <nav className="flex gap-1 overflow-x-auto px-2 pb-2 lg:flex-col lg:overflow-visible" aria-label="Sections">
          {NAV.map(([k, l]) => (
            <button key={k} onClick={() => setPage(k)} aria-current={page === k ? "page" : undefined}
              className={`whitespace-nowrap rounded-lg px-3 py-2 text-left text-sm transition ${page === k ? "bg-accent/15 text-accent" : "text-slate-300 hover:bg-ink-800"}`}>{l}</button>))}
        </nav>
        <div className="hidden px-4 py-4 text-[11px] leading-relaxed text-slate-500 lg:absolute lg:bottom-0 lg:block">B.Tech CSE minor project<br />Map © OpenStreetMap contributors</div>
      </aside>
      <main className="min-w-0 flex-1 p-4 lg:p-6">
        <header className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-lg font-semibold text-slate-50">{NAV.find(([k]) => k === page)?.[1]}</h1>
          <div className="flex gap-2"><Badge tone="warn">Replay of historical data · not live traffic</Badge>{metrics && <Badge tone="info">{metrics.registry.filter((m) => m.servable).length} models ready</Badge>}{metrics && metrics.registry.some((m) => m.status === "error") && <Badge tone="bad">{metrics.registry.filter((m) => m.status === "error").length} model error(s) · see Model Evaluation</Badge>}</div>
        </header>
        {bootError ? <ErrorBox title="Backend not ready">{bootError}<div className="mt-2 text-xs text-slate-400">Run <code>python scripts/build_panel.py</code> and <code>python scripts/train_model.py</code>, then <code>uvicorn api.main:app --port 8000</code>.</div></ErrorBox>
          : !config || !metrics ? <Spinner label="Connecting to backend…" /> : views[page]}
      </main>
    </div>
  );
}
