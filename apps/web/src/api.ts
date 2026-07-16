export type OracleSpec = {
  provider: string;
  model?: string;
  mode?: string;
  label: string;
  noise?: number;
  items_per_call?: number;
};

export type RunReport = {
  accuracy: number;
  macro_f1: number;
  invalid_label_rate: number;
  n_total: number;
  total_cost_usd: number;
  cost_per_1k_labels_usd: number;
};

export type Run = {
  id: string;
  name: string;
  kind: string;
  status: "pending" | "running" | "completed" | "failed";
  config: { sample: string; limit: number; oracle: Record<string, unknown> };
  report?: RunReport | null;
  error?: string | null;
  artifacts_dir?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
};

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<{ status: string; database: string }>(r)),
  oracles: () => fetch("/api/oracles").then((r) => json<OracleSpec[]>(r)),
  runs: () => fetch("/api/runs").then((r) => json<Run[]>(r)),
  run: (id: string) => fetch(`/api/runs/${id}`).then((r) => json<Run>(r)),
  createRun: (body: {
    name: string;
    sample: string;
    limit: number;
    oracle: Record<string, unknown>;
  }) =>
    fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<Run>(r)),
};
