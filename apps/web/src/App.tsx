import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type OracleSpec, type Run } from "./api";

const STATUS_LABEL: Record<Run["status"], string> = {
  pending: "na fila",
  running: "executando",
  completed: "concluído",
  failed: "falhou",
};

function pct(x: number | undefined): string {
  return x === undefined || x === null ? "—" : `${(x * 100).toFixed(1)}%`;
}

export default function App() {
  const [db, setDb] = useState<string>("…");
  const [oracles, setOracles] = useState<OracleSpec[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [sample, setSample] = useState("rand");
  const [limit, setLimit] = useState(50);
  const [oracleIdx, setOracleIdx] = useState(0);
  const [noise, setNoise] = useState(0.1);
  const [submitting, setSubmitting] = useState(false);

  const refresh = useCallback(() => {
    api.runs().then(setRuns).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    api.health().then((h) => setDb(h.database)).catch(() => setDb("offline"));
    api.oracles().then(setOracles).catch((e) => setError(String(e)));
    refresh();
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [refresh]);

  const oracle = oracles[oracleIdx];
  const oracleSpec = useMemo(() => {
    if (!oracle) return { provider: "simulated", noise };
    if (oracle.provider === "simulated") return { provider: "simulated", noise };
    const { label: _label, ...spec } = oracle;
    return spec;
  }, [oracle, noise]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await api.createRun({
        name: name || `run ${new Date().toLocaleString("pt-BR")}`,
        sample,
        limit,
        oracle: oracleSpec,
      });
      setName("");
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="shell">
      <header>
        <h1>FALCO FlowBuilder</h1>
        <span className={`badge db-${db}`}>banco: {db}</span>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="card">
        <h2>Novo run — avaliação de oráculo</h2>
        <form onSubmit={submit} className="grid">
          <label>
            Nome
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="opcional" />
          </label>
          <label>
            Amostra
            <select value={sample} onChange={(e) => setSample(e.target.value)}>
              <option value="rand">S-rand (aleatória)</option>
              <option value="strat">S-strat (k por classe)</option>
            </select>
          </label>
          <label>
            Instâncias (limite)
            <input
              type="number"
              min={1}
              max={5000}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
          </label>
          <label>
            Oráculo
            <select value={oracleIdx} onChange={(e) => setOracleIdx(Number(e.target.value))}>
              {oracles.map((o, i) => (
                <option key={o.label} value={i}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          {oracle?.provider === "simulated" && (
            <label>
              Ruído ε
              <input
                type="number"
                min={0}
                max={0.9}
                step={0.05}
                value={noise}
                onChange={(e) => setNoise(Number(e.target.value))}
              />
            </label>
          )}
          <button disabled={submitting || oracles.length === 0}>
            {submitting ? "Criando…" : "Executar"}
          </button>
        </form>
        <p className="hint">
          Oráculos LLM exigem as chaves no <code>.env</code> do servidor; o simulado roda offline.
        </p>
      </section>

      <section className="card">
        <h2>Runs</h2>
        <table>
          <thead>
            <tr>
              <th>Nome</th>
              <th>Amostra</th>
              <th>n</th>
              <th>Status</th>
              <th>Acc</th>
              <th>Macro-F1</th>
              <th>US$/1k</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} className={`status-${r.status}`}>
                <td>{r.name}</td>
                <td>{r.config?.sample}</td>
                <td>{r.report?.n_total ?? r.config?.limit}</td>
                <td>{STATUS_LABEL[r.status]}</td>
                <td>{pct(r.report?.accuracy)}</td>
                <td>{pct(r.report?.macro_f1)}</td>
                <td>{r.report ? r.report.cost_per_1k_labels_usd.toFixed(3) : "—"}</td>
                <td>
                  <button onClick={() => api.run(r.id).then(setSelected)}>detalhes</button>
                </td>
              </tr>
            ))}
            {runs.length === 0 && (
              <tr>
                <td colSpan={8} className="empty">
                  Nenhum run ainda — crie o primeiro acima.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      {selected && (
        <section className="card">
          <h2>
            Run {selected.id} — {selected.name}{" "}
            <button onClick={() => setSelected(null)}>fechar</button>
          </h2>
          {selected.error && <pre className="error">{selected.error}</pre>}
          <pre>{JSON.stringify(selected.report ?? selected.config, null, 2)}</pre>
          {selected.artifacts_dir && (
            <p className="hint">
              Artefatos (JSONL rastreável): <code>{selected.artifacts_dir}</code>
            </p>
          )}
        </section>
      )}
    </main>
  );
}
