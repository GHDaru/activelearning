export type OracleSpec = {
  provider: string;
  model?: string;
  mode?: string;
  label: string;
  noise?: number;
  items_per_call?: number;
};

export type RunReport = {
  accuracy?: number;
  macro_f1?: number;
  invalid_label_rate?: number;
  n_total?: number;
  total_cost_usd?: number;
  cost_per_1k_labels_usd?: number;
  // active-learning
  strategy?: string;
  lce_macro_f1?: number;
  final_macro_f1?: number;
  final_accuracy?: number;
  n_labeled?: number;
  invalid_labels?: number;
  pool_size?: number;
  test_size?: number;
  n_classes?: number;
  oracle_id?: string;
  curve?: { n: number; macro_f1: number }[];
};

export type Run = {
  id: string;
  name: string;
  kind: string;
  status: "pending" | "running" | "completed" | "failed";
  config: {
    sample?: string;
    limit?: number;
    dataset_id?: string;
    params?: Record<string, unknown>;
    oracle: Record<string, unknown>;
  };
  report?: RunReport | null;
  error?: string | null;
  artifacts_dir?: string | null;
  created_at: string;
};

export type DatasetReport = {
  n_rows_in: number;
  n_rows_out: number;
  removed_empty: number;
  removed_operational: number;
  n_classes: number;
  n_conflicting_texts: number;
  n_conflicting_rows: number;
  conflict_examples: { texto: string; rotulos: string[] }[];
  n_exact_duplicates: number;
  n_rare_classes_lt5: number;
  class_histogram_top: [string, number][];
};

export type Experiment = {
  id: string; titulo: string; pilar: string; pergunta: string;
  descricao: string; duracao: string; requer_chave: boolean;
  presets: string[]; artefatos_disponiveis: string[]; n_artefatos: number;
  legado?: boolean; auditoria?: string | null;
  job: { pid: number; preset: string; started: number; status: string } | null;
};

export type ResultBlock = {
  label: string;
  kind: "json" | "curve" | "table" | "text" | "ausente";
  text?: string;
  data?: Record<string, unknown>;
  points?: { n: number; y: number }[];
  resumo?: Record<string, unknown>;
  rows?: Record<string, unknown>[];
};

export type DatasetStats = {
  n_rows: number;
  n_classes: number;
  vocab_size: number;
  per_class: { min: number; median: number; mean: number; max: number; lt5: number; imbalance_ratio: number | null };
  text: { chars_mean: number; chars_p50: number; chars_max: number; tokens_mean: number };
  top_classes: { label: string; n: number }[];
};

export type Dataset = {
  id: string;
  name: string;
  filename: string;
  text_column: string;
  label_column: string;
  created_at: string;
  n_rows?: number;
  n_classes?: number;
  report?: DatasetReport;
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
  createRun: (body: Record<string, unknown>) =>
    fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<Run>(r)),
  datasets: () => fetch("/api/datasets").then((r) => json<Dataset[]>(r)),
  dataset: (id: string) => fetch(`/api/datasets/${id}`).then((r) => json<Dataset>(r)),
  datasetStats: (id: string) =>
    fetch(`/api/datasets/${id}/stats`).then((r) => json<DatasetStats>(r)),
  uploadDataset: (form: FormData) =>
    fetch("/api/datasets", { method: "POST", body: form }).then((r) => json<Dataset>(r)),
  experiments: () => fetch("/api/experiments").then((r) => json<Experiment[]>(r)),
  experimentResults: (id: string) =>
    fetch(`/api/experiments/${id}/results`).then((r) => json<{ id: string; titulo: string; blocks: ResultBlock[] }>(r)),
  experimentExecute: (id: string, preset: string) =>
    fetch(`/api/experiments/${id}/execute`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset }),
    }).then((r) => json<Record<string, unknown>>(r)),
  experimentLog: (id: string) =>
    fetch(`/api/experiments/${id}/log`).then((r) => json<{ log: string }>(r)),
  downloadUrl: (id: string, which: "sanitized" | "original") =>
    `/api/datasets/${id}/download?which=${which}`,
};
