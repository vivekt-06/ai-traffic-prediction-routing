import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { AppProvider } from "./store";
import "leaflet/dist/leaflet.css";   // MUST be bundled: without it Leaflet panes/tiles are unpositioned (small tile square, no overlays)
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(<React.StrictMode><AppProvider><App /></AppProvider></React.StrictMode>);
