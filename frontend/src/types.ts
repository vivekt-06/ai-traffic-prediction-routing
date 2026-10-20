export type Level = "LOW" | "MODERATE" | "HIGH" | "SEVERE";
export interface Place { name?: string; lat: number; lon: number }
export interface Landmark { name: string; lat: number; lon: number; snap_m: number }
export interface Config {
  congestion: { k_speed_loss: number; intensity_cap: number; min_speed_kmh: number; saturation_quantile: number;
    thresholds: Record<string, number>; colors: Record<string, string>; negligible_saving_minutes: number; max_snap_distance_m: number };
  landmarks: Landmark[]; landmarks_out_of_coverage: Landmark[];
  time_window: { start: string; end: string }; splits: Record<string, [string, string]>;
  map: { center: [number, number]; attribution: string; tile_policy_url: string };
}
export interface ModelMetrics { MAE: number | null; RMSE: number | null; MAPE: number | null; R2: number | null; WAPE: number | null;
  n: number; mape_coverage: number | null; congestion_level_agreement?: number }
export type ModelStatus = "ready" | "untrained" | "error";
export interface ModelCheck { name: string; ok: boolean; detail: string }
export interface ModelEntry { key: string; label: string; role: string; description: string; status: ModelStatus; status_detail: string; checks: ModelCheck[]; servable: boolean;
  reason?: string; metrics: null | { val: ModelMetrics; test: ModelMetrics; fit_seconds: number; trained_at: string; n_train_rows?: number; uses_weather: boolean; smoke_run?: boolean } }
export interface MetricsDoc { generated_at: string; split: Record<string, [string, string]>; split_rows: Record<string, number>;
  train_subsample_rows: number; target?: string; default_model: string; default_model_rule: string; registry: ModelEntry[] }
export interface RouteSeg { points: [number, number][]; level: Level; street: string; ci: number }
export interface RouteSide {
  label: string; distance_km: number; duration_min: number; free_flow_duration_min: number; delay_min: number;
  departure: string; arrival: string; n_segments: number; geometry: [number, number][]; segments: RouteSeg[];
  congestion_exposure: { mean_ci: number; mean_level: Level; minutes_in_high_or_severe: number; km_by_level: Record<Level, number>; pct_distance_high_or_severe: number };
  top_congested_segments: { segment_id: number; street: string; length_m: number; congestion_index: number; level: Level; speed_kmh: number; entered_at: string }[];
}
export interface EventItem { event_id: number; event_name: string; event_type: string; location: string; lat: number; lon: number;
  start_time: string; end_time: string; impact_radius_m: number | null; citywide: boolean; source: string; verification_status: string; note: string;
  overlap_type?: string; min_distance_m?: number | null }
export interface RouteResult {
  model: string; model_label: string; split_of_departure_hour: string; tdsp: RouteSide; shortest: RouteSide;
  snapped: { origin: { lat: number; lon: number; snap_m: number }; destination: { lat: number; lon: number; snap_m: number } };
  difference: { time_saved_min: number; extra_distance_km: number; same_path: boolean }; notes: string[]; events: EventItem[];
}
export interface Scenario { offset_min: number; label: string; departure: string; available: boolean; reason?: string; duration_min?: number;
  distance_km?: number; arrival?: string; mean_ci?: number; level?: Level; shortest_route_duration_min?: number; delay_min?: number }
export interface DepartureResult { model: string; scenarios: Scenario[]; note: string;
  recommendation: { best_offset_min: number | null; explanation: string; time_saved_vs_now_min?: number; negligible?: boolean; arrival_delta_vs_now_min?: number } }
export interface MapSummary { counts: Record<Level, number>; mean_ci: number; share_high_or_severe_pct: number; n_segments: number }

export type MapSource = "predicted" | "observed";
export interface MapSegment { id: number; street: string; level: Level; ci: number; speed: number; count: number; points: [number, number][] }
export interface MapLayerResponse { hour: string; source: MapSource; model: string | null; split: string; truncated: boolean; count_kind: string; segments: MapSegment[]; citywide_summary: MapSummary }
