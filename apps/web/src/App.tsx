import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Dataset, type DatasetStats, type Experiment, type KgSummary, type OracleSpec, type ResultBlock, type Run } from "./api";

const STATUS_LABEL: Record<Run["status"], string> = {
  pending: "na fila",
  running: "executando",
  completed: "concluído",
  failed: "falhou",
};

const STRATEGIES = ["entropy", "least_confidence", "smallest_margin", "random", "hybrid"];

type View = "home" | "datasets" | "runs" | "experiments" | "conhecimento";

function pct(x: number | undefined | null): string {
  return x === undefined || x === null ? "—" : `${(x * 100).toFixed(1)}%`;
}

function Curve({ points }: { points: { n: number; macro_f1: number }[] }) {
  if (!points || points.length < 2) return null;
  const w = 560, h = 180, pad = 34;
  const xs = points.map((p) => p.n), ys = points.map((p) => p.macro_f1);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = 0, ymax = Math.max(...ys, 0.01) * 1.1;
  const X = (n: number) => pad + ((n - xmin) / (xmax - xmin || 1)) * (w - 2 * pad);
  const Y = (v: number) => h - pad - ((v - ymin) / (ymax - ymin || 1)) * (h - 2 * pad);
  const d = points.map((p, i) => `${i ? "L" : "M"}${X(p.n).toFixed(1)},${Y(p.macro_f1).toFixed(1)}`).join(" ");
  return (
    <svg width={w} height={h} role="img" aria-label="Curva de aprendizado (Macro F1 × rótulos)">
      <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#cbd5e1" />
      <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#cbd5e1" />
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} x1={pad} x2={w - pad} y1={Y(ymax * f)} y2={Y(ymax * f)}
          stroke="#eef1f6" />
      ))}
      <path d={d} fill="none" stroke="#2456a6" strokeWidth={2.2} />
      {points.map((p) => (
        <circle key={p.n} cx={X(p.n)} cy={Y(p.macro_f1)} r={2.8} fill="#2456a6" />
      ))}
      <text x={w - pad} y={h - 10} fontSize={11} textAnchor="end" fill="#55617a">rótulos adquiridos</text>
      <text x={8} y={pad - 8} fontSize={11} fill="#55617a">Macro F1</text>
      <text x={w - pad} y={Y(ys[ys.length - 1]) - 8} fontSize={11} textAnchor="end" fill="#2456a6">
        {(ys[ys.length - 1] * 100).toFixed(1)}%
      </text>
    </svg>
  );
}

function StatTile({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="tile" title={hint}>
      <div className="tile-value">{value}</div>
      <div className="tile-label">{label}</div>
    </div>
  );
}

function TopClasses({ stats }: { stats: DatasetStats }) {
  const top = stats.top_classes;
  if (!top?.length) return null;
  const max = top[0].n;
  return (
    <div className="topclasses">
      <h3>Top-10 classes ({stats.n_classes} no total)</h3>
      {top.map((c) => (
        <div key={c.label} className="bar-row">
          <span className="bar-label">{c.label}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(c.n / max) * 100}%` }} />
          </div>
          <span className="bar-n">{c.n.toLocaleString("pt-BR")}</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<View>("home");
  const [db, setDb] = useState("…");
  const [oracles, setOracles] = useState<OracleSpec[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<Run | null>(null);
  const [selectedDs, setSelectedDs] = useState<Dataset | null>(null);
  const [dsStats, setDsStats] = useState<DatasetStats | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [expOpen, setExpOpen] = useState<{ id: string; titulo: string; blocks: ResultBlock[] } | null>(null);
  const [expLog, setExpLog] = useState<{ id: string; log: string } | null>(null);
  const [kg, setKg] = useState<KgSummary | null>(null);
  const [kgError, setKgError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // upload
  const [file, setFile] = useState<File | null>(null);
  const [dsName, setDsName] = useState("");
  const [textCol, setTextCol] = useState("nm_item");
  const [labelCol, setLabelCol] = useState("nm_product");
  const [opsLabels, setOpsLabels] = useState("inativo");

  // run form
  const [kind, setKind] = useState<"active-learning" | "oracle-eval">("active-learning");
  const [runName, setRunName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [seed, setSeed] = useState(42);
  const [budget, setBudget] = useState(300);
  const [batchSize, setBatchSize] = useState(30);
  const [initialSize, setInitialSize] = useState(30);
  const [poolSize, setPoolSize] = useState(2000);
  const [strategy, setStrategy] = useState("entropy");
  const [oracleIdx, setOracleIdx] = useState(0);
  const [noise, setNoise] = useState(0.1);
  const [sample, setSample] = useState("rand");
  const [limit, setLimit] = useState(50);

  const refresh = useCallback(() => {
    api.runs().then(setRuns).catch((e) => setError(String(e)));
    api.datasets().then(setDatasets).catch(() => undefined);
  }, []);

  useEffect(() => {
    api.health().then((h) => setDb(h.database)).catch(() => setDb("offline"));
    api.oracles().then(setOracles).catch((e) => setError(String(e)));
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    if (view !== "experiments") return;
    const load = () => api.experiments().then(setExperiments).catch((e) => setError(String(e)));
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [view]);

  useEffect(() => {
    if (view !== "conhecimento" || kg) return;
    api.kgSummary().then((s) => { setKg(s); setKgError(null); })
      .catch((e) => setKgError(String(e)));
  }, [view, kg]);

  useEffect(() => {
    if (!expLog) return;
    const t = setInterval(() =>
      api.experimentLog(expLog.id).then((r) => setExpLog({ id: expLog.id, log: r.log })), 3000);
    return () => clearInterval(t);
  }, [expLog?.id]);

  async function openDataset(id: string) {
    setDsStats(null);
    const ds = await api.dataset(id);
    setSelectedDs(ds);
    api.datasetStats(id).then(setDsStats).catch(() => setDsStats(null));
  }

  const oracle = oracles[oracleIdx];
  const oracleSpec = useMemo(() => {
    if (!oracle || oracle.provider === "simulated") return { provider: "simulated", noise };
    const { label: _l, ...spec } = oracle;
    return spec;
  }, [oracle, noise]);

  async function doUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true); setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("name", dsName || file.name);
      form.append("text_column", textCol);
      form.append("label_column", labelCol);
      form.append("operational_labels", opsLabels);
      const ds = await api.uploadDataset(form);
      setDatasetId(ds.id);
      setFile(null); setDsName("");
      refresh();
      openDataset(ds.id);
    } catch (err) { setError(String(err)); } finally { setBusy(false); }
  }

  async function doRun(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      const body: Record<string, unknown> =
        kind === "active-learning"
          ? {
              name: runName || `AL ${new Date().toLocaleString("pt-BR")}`,
              kind, dataset_id: datasetId,
              params: { seed, budget, batch_size: batchSize, initial_size: initialSize,
                        strategy, pool_size: poolSize },
              oracle: oracleSpec,
            }
          : { name: runName || `E0 ${new Date().toLocaleString("pt-BR")}`,
              kind, sample, limit, oracle: oracleSpec };
      await api.createRun(body);
      setRunName("");
      refresh();
    } catch (err) { setError(String(err)); } finally { setBusy(false); }
  }

  const nav: { key: View; label: string; desc: string }[] = [
    { key: "home", label: "Início", desc: "o que é e como usar" },
    { key: "datasets", label: "Datasets", desc: "envio, saneamento e estatísticas" },
    { key: "runs", label: "Execuções", desc: "laço de AL e avaliação de oráculo" },
    { key: "experiments", label: "Experimentos da tese", desc: "catálogo: reproduzir e reprisar" },
    { key: "conhecimento", label: "Base de conhecimento", desc: "grafo de fichamentos e conceitos" },
  ];

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-name">FALCO</div>
          <div className="brand-sub">aprendizado ativo com oráculos LLM</div>
        </div>
        <nav>
          {nav.map((n) => (
            <button key={n.key} className={view === n.key ? "nav-item active" : "nav-item"}
              onClick={() => setView(n.key)}>
              <span>{n.label}</span>
              <small>{n.desc}</small>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className={`badge db-${db}`}>banco: {db}</span>
        </div>
      </aside>

      <main className="content">
        {error && <div className="error">{error}</div>}

        {view === "home" && (
          <>
            <section className="hero card">
              <h1>Bem-vindo ao FALCO</h1>
              <p>
                O FALCO (<i>Framework de Aprendizado Ativo com LLM para texto CurtO</i>)
                treina classificadores de texto gastando o mínimo possível em rótulos:
                em vez de anotar tudo, o laço escolhe <b>quais</b> exemplos valem a
                anotação e usa um <b>oráculo</b> — simulado ou um LLM real — para
                rotulá-los. Esta interface executa o ciclo completo sobre a
                biblioteca <code>activelearning</code>, com artefatos rastreáveis para
                cada número.
              </p>
            </section>
            <section className="steps">
              {[
                ["1 · Envie seus dados", "Um CSV com coluna de texto e coluna de rótulo. O FALCO saneia (remove vazios e rótulos operacionais), detecta conflitos e duplicatas, e calcula as estatísticas da base — classes, vocabulário, desbalanceamento.", "datasets"],
                ["2 · Configure o fluxo", "Escolha a estratégia de seleção (entropia é o padrão validado), o oráculo (simulado com ruído ε, ou um LLM via API), o orçamento de rótulos e o tamanho do lote.", "runs"],
                ["3 · Execute o laço", "O servidor roda: conjunto inicial → treina → seleciona os mais incertos → oráculo rotula → re-treina, até o orçamento ou a estagnação da validação.", "runs"],
                ["4 · Analise a curva", "Cada execução entrega a curva de aprendizado (Macro F1 × rótulos), a LCE e o relatório completo — os mesmos instrumentos dos experimentos da tese.", "runs"],
              ].map(([t, d, dest]) => (
                <div key={t} className="card step" onClick={() => setView(dest as View)}>
                  <h3>{t}</h3>
                  <p>{d}</p>
                </div>
              ))}
            </section>
            <section className="card hint-card">
              <h3>Boas práticas</h3>
              <ul>
                <li><b>Comece com o oráculo simulado</b> (offline, ε = taxa de erro) para calibrar orçamento e lote; troque pelo LLM real depois.</li>
                <li><b>Desconfie da autoavaliação</b>: métricas medidas nos próprios dados coletados são enviesadas — reserve um conjunto de teste externo para decisões de liberação.</li>
                <li><b>Pare pela curva</b>: quando a validação estagna, rótulo adicional compra ruído — e pode até piorar o Macro F1.</li>
              </ul>
            </section>
          </>
        )}

        {view === "datasets" && (
          <>
            <section className="card">
              <h2>Enviar CSV (texto, rótulo)</h2>
              <form onSubmit={doUpload} className="grid">
                <label>Arquivo CSV
                  <input type="file" accept=".csv,text/csv"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
                </label>
                <label>Nome
                  <input value={dsName} onChange={(e) => setDsName(e.target.value)}
                    placeholder={file?.name ?? "meu-dataset"} />
                </label>
                <label>Coluna de texto
                  <input value={textCol} onChange={(e) => setTextCol(e.target.value)} />
                </label>
                <label>Coluna de rótulo
                  <input value={labelCol} onChange={(e) => setLabelCol(e.target.value)} />
                </label>
                <label>Rótulos operacionais (removidos)
                  <input value={opsLabels} onChange={(e) => setOpsLabels(e.target.value)}
                    placeholder="inativo, descontinuado" />
                </label>
                <button disabled={!file || busy}>{busy ? "Enviando…" : "Enviar e sanear"}</button>
              </form>
            </section>

            {datasets.length > 0 && (
              <section className="card">
                <h2>Bases disponíveis</h2>
                <table>
                  <thead><tr><th>Nome</th><th>Linhas</th><th>Classes</th><th>CSV</th><th></th></tr></thead>
                  <tbody>
                    {datasets.map((d) => (
                      <tr key={d.id} className={selectedDs?.id === d.id ? "row-active" : ""}>
                        <td>{d.name}</td>
                        <td>{d.n_rows?.toLocaleString("pt-BR") ?? "—"}</td>
                        <td>{d.n_classes ?? "—"}</td>
                        <td><a href={api.downloadUrl(d.id, "sanitized")}>baixar</a></td>
                        <td><button onClick={() => openDataset(d.id)}>estatísticas</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}

            {selectedDs && (
              <section className="card">
                <h2>{selectedDs.name}
                  <button className="close" onClick={() => { setSelectedDs(null); setDsStats(null); }}>fechar</button>
                </h2>
                {dsStats ? (
                  <>
                    <div className="tiles">
                      <StatTile label="linhas" value={dsStats.n_rows.toLocaleString("pt-BR")} />
                      <StatTile label="classes" value={dsStats.n_classes} />
                      <StatTile label="vocabulário" value={dsStats.vocab_size.toLocaleString("pt-BR")}
                        hint="tokens únicos (minúsculas, separados por espaço)" />
                      <StatTile label="mediana por classe" value={dsStats.per_class.median} />
                      <StatTile label="maior / menor classe" value={dsStats.per_class.imbalance_ratio ?? "—"}
                        hint={`máx ${dsStats.per_class.max} · mín ${dsStats.per_class.min}`} />
                      <StatTile label="classes com < 5 exemplos" value={dsStats.per_class.lt5} />
                      <StatTile label="caracteres por texto (méd.)" value={dsStats.text.chars_mean} />
                      <StatTile label="tokens por texto (méd.)" value={dsStats.text.tokens_mean} />
                    </div>
                    <TopClasses stats={dsStats} />
                  </>
                ) : <p className="hint">calculando estatísticas…</p>}
                {selectedDs.report && (
                  <>
                    <h3>Relatório de saneamento</h3>
                    <ul>
                      <li>{selectedDs.report.n_rows_in} linhas recebidas → <b>{selectedDs.report.n_rows_out} mantidas</b>{" "}
                        ({selectedDs.report.removed_operational} operacionais + {selectedDs.report.removed_empty} vazias removidas)</li>
                      <li><b>{selectedDs.report.n_conflicting_texts} textos com rótulos conflitantes</b>{" "}
                        ({selectedDs.report.n_conflicting_rows} linhas) — mantidos e reportados</li>
                      <li>{selectedDs.report.n_exact_duplicates} duplicatas exatas — o particionamento deduplica antes do split</li>
                    </ul>
                    {selectedDs.report.conflict_examples.length > 0 && (
                      <pre>{selectedDs.report.conflict_examples.map((c) =>
                        `${c.texto}  →  {${c.rotulos.join(", ")}}`).join("\n")}</pre>
                    )}
                  </>
                )}
              </section>
            )}
          </>
        )}

        {view === "runs" && (
          <>
            <section className="card">
              <h2>Nova execução</h2>
              <form onSubmit={doRun} className="grid">
                <label>Tipo
                  <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
                    <option value="active-learning">Aprendizado ativo (curva completa)</option>
                    <option value="oracle-eval">Avaliação de oráculo (E0)</option>
                  </select>
                </label>
                <label>Nome
                  <input value={runName} onChange={(e) => setRunName(e.target.value)} placeholder="opcional" />
                </label>
                {kind === "active-learning" ? (
                  <>
                    <label>Dataset
                      <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                        <option value="">— escolha —</option>
                        {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                      </select>
                    </label>
                    <label>Semente<input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} /></label>
                    <label>Orçamento (rótulos)<input type="number" min={20} value={budget} onChange={(e) => setBudget(+e.target.value)} /></label>
                    <label>Lote<input type="number" min={1} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} /></label>
                    <label>L0 inicial<input type="number" min={1} value={initialSize} onChange={(e) => setInitialSize(+e.target.value)} /></label>
                    <label>Pool máx.<input type="number" min={100} value={poolSize} onChange={(e) => setPoolSize(+e.target.value)} /></label>
                    <label>Estratégia
                      <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                        {STRATEGIES.map((s) => <option key={s} value={s}>{s}</option>)}
                      </select>
                    </label>
                  </>
                ) : (
                  <>
                    <label>Amostra
                      <select value={sample} onChange={(e) => setSample(e.target.value)}>
                        <option value="rand">S-rand</option><option value="strat">S-strat</option>
                      </select>
                    </label>
                    <label>Instâncias<input type="number" min={1} max={5000} value={limit}
                      onChange={(e) => setLimit(+e.target.value)} /></label>
                  </>
                )}
                <label>Oráculo
                  <select value={oracleIdx} onChange={(e) => setOracleIdx(+e.target.value)}>
                    {oracles.map((o, i) => <option key={o.label} value={i}>{o.label}</option>)}
                  </select>
                </label>
                {oracle?.provider === "simulated" && (
                  <label>Ruído ε<input type="number" min={0} max={0.9} step={0.05} value={noise}
                    onChange={(e) => setNoise(+e.target.value)} /></label>
                )}
                <button disabled={busy || (kind === "active-learning" && !datasetId)}>
                  {busy ? "Criando…" : "Executar"}
                </button>
              </form>
              <p className="hint">
                Oráculos LLM exigem chaves no <code>.env</code> do servidor. O simulado
                roda offline (ε = taxa de erro do oráculo).
              </p>
            </section>

            <section className="card">
              <h2>Execuções</h2>
              <table>
                <thead><tr><th>Nome</th><th>Tipo</th><th>Status</th><th>Macro F1</th><th>LCE</th><th>Rótulos</th><th></th></tr></thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id} className={`status-${r.status}`}>
                      <td>{r.name}</td>
                      <td>{r.kind}</td>
                      <td>{STATUS_LABEL[r.status]}</td>
                      <td>{pct(r.report?.final_macro_f1 ?? r.report?.macro_f1)}</td>
                      <td>{r.report?.lce_macro_f1?.toFixed(3) ?? "—"}</td>
                      <td>{r.report?.n_labeled ?? r.report?.n_total ?? "—"}</td>
                      <td><button onClick={() => api.run(r.id).then(setSelected)}>detalhes</button></td>
                    </tr>
                  ))}
                  {runs.length === 0 && (
                    <tr><td colSpan={7} className="empty">Nenhuma execução ainda — envie um CSV e execute.</td></tr>
                  )}
                </tbody>
              </table>
            </section>

            {selected && (
              <section className="card">
                <h2>{selected.name}
                  <button className="close" onClick={() => setSelected(null)}>fechar</button>
                </h2>
                {selected.error && <pre className="error">{selected.error}</pre>}
                {selected.report?.curve && <Curve points={selected.report.curve} />}
                {selected.report && (
                  <div className="tiles">
                    {selected.report.final_macro_f1 !== undefined &&
                      <StatTile label="Macro F1 final" value={pct(selected.report.final_macro_f1)} />}
                    {selected.report.final_accuracy !== undefined &&
                      <StatTile label="acurácia final" value={pct(selected.report.final_accuracy)} />}
                    {selected.report.lce_macro_f1 !== undefined &&
                      <StatTile label="LCE" value={selected.report.lce_macro_f1?.toFixed(3) ?? "—"} />}
                    {selected.report.n_labeled !== undefined &&
                      <StatTile label="rótulos usados" value={selected.report.n_labeled ?? "—"} />}
                  </div>
                )}
                <details>
                  <summary>relatório completo (JSON)</summary>
                  <pre>{JSON.stringify(selected.report ?? selected.config, null, 2)}</pre>
                </details>
                {selected.artifacts_dir && (
                  <p className="hint">Artefatos rastreáveis: <code>{selected.artifacts_dir}</code></p>
                )}
              </section>
            )}
          </>
        )}

        {view === "experiments" && (
          <>
            <section className="card hero">
              <h1>Experimentos da tese</h1>
              <p>
                Catálogo executável do programa experimental do FALCO. <b>Reproduzir</b>{" "}
                relança o runner original (com retomada por estado — pode interromper e
                voltar). <b>Reprisar</b> carrega os artefatos gravados — os mesmos
                arquivos que sustentam cada número da tese.
              </p>
            </section>
            {experiments.map((e) => (
              <section key={e.id} className="card exp-card">
                <div className="exp-head">
                  <div>
                    <h2>{e.titulo}</h2>
                    <p className="exp-q">{e.pergunta}</p>
                    <p className="exp-desc">{e.descricao}</p>
                  </div>
                  <div className="exp-badges">
                    <span className="chip">{e.pilar}</span>
                    {e.legado && <span className="chip legacy">legado</span>}
                    {e.auditoria === "pendente" && <span className="chip warn">auditoria pendente</span>}
                    {e.requer_chave && <span className="chip warn">requer chave API</span>}
                    {e.job?.status === "executando" && <span className="chip run">executando…</span>}
                    <span className="chip ok">{e.artefatos_disponiveis.length}/{e.n_artefatos} artefatos</span>
                  </div>
                </div>
                <div className="exp-actions">
                  <button disabled={e.artefatos_disponiveis.length === 0}
                    onClick={() => { setExpLog(null); api.experimentResults(e.id).then(setExpOpen).catch((err) => setError(String(err))); }}>
                    Reprisar resultados
                  </button>
                  {e.presets.map((p) => (
                    <button key={p} className="secondary"
                      disabled={e.job?.status === "executando"}
                      onClick={() => api.experimentExecute(e.id, p)
                        .then(() => { setExpOpen(null); api.experimentLog(e.id).then((r) => setExpLog({ id: e.id, log: r.log })); })
                        .catch((err) => setError(String(err)))}>
                      Reproduzir: {p}
                    </button>
                  ))}
                  {(e.job) && (
                    <button className="secondary"
                      onClick={() => api.experimentLog(e.id).then((r) => setExpLog({ id: e.id, log: r.log }))}>
                      ver log
                    </button>
                  )}
                  <span className="hint">{e.duracao}</span>
                </div>
              </section>
            ))}

            {expLog && (
              <section className="card">
                <h2>Log — {expLog.id}
                  <button className="close" onClick={() => setExpLog(null)}>fechar</button></h2>
                <pre className="log">{expLog.log || "(aguardando saída…)"}</pre>
              </section>
            )}

            {expOpen && (
              <section className="card">
                <h2>{expOpen.titulo}
                  <button className="close" onClick={() => setExpOpen(null)}>fechar</button></h2>
                {expOpen.blocks.map((b, i) => (
                  <div key={i} className="block">
                    <h3>{b.label}</h3>
                    {b.kind === "ausente" && <p className="hint">artefato ainda não gerado (ou repositório legado ausente nesta máquina).</p>}
                    {b.kind === "text" && b.text && <pre>{b.text}</pre>}
                    {b.kind === "curve" && b.points && b.points.length > 1 && (
                      <Curve points={b.points.map((p) => ({ n: p.n, macro_f1: p.y }))} />
                    )}
                    {b.kind === "curve" && b.resumo && (
                      <div className="tiles">
                        {Object.entries(b.resumo).filter(([, v]) => v !== null && v !== undefined)
                          .map(([k, v]) => <StatTile key={k} label={k} value={String(v)} />)}
                      </div>
                    )}
                    {b.kind === "json" && (
                      <details open={i === 0}>
                        <summary>conteúdo</summary>
                        <pre>{JSON.stringify(b.data, null, 2)}</pre>
                      </details>
                    )}
                    {b.kind === "table" && b.rows && b.rows.length > 0 && (
                      <div className="scroll-x">
                        <table>
                          <thead><tr>{Object.keys(b.rows[0]).map((c) => <th key={c}>{c}</th>)}</tr></thead>
                          <tbody>
                            {b.rows.map((r, j) => (
                              <tr key={j}>{Object.keys(b.rows![0]).map((c) => <td key={c}>{String(r[c] ?? "—")}</td>)}</tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                ))}
              </section>
            )}
          </>
        )}

        {view === "conhecimento" && (
          <>
            <section className="card hero">
              <h1>Base de conhecimento</h1>
              <p>
                Grafo dos <b>fichamentos</b> da revisão de literatura: cada artigo é
                um nó, ligado aos <b>conceitos</b> que propõe ou usa (métodos, modelos,
                tarefas, bases, pilares) e a <b>outros artigos</b> por relações tipadas
                (<i>estende</i>, <i>compara</i>, <i>contradiz</i>, <i>baseia-se em</i>).
                As arestas em destaque ligam a literatura ao próprio FALCO. Nós em tom
                apagado são artigos <i>referenciados mas ainda não fichados</i>.
              </p>
              {kg && (
                <div className="tiles">
                  <StatTile label="nós" value={kg.n_nodes} />
                  <StatTile label="arestas" value={kg.n_edges} />
                  <StatTile label="artigos fichados" value={kg.n_artigos} />
                  <StatTile label="pendentes de fichamento" value={kg.n_pendentes}
                    hint="alvos de relações artigo→artigo ainda sem nota própria" />
                </div>
              )}
              <p className="hint">
                O grafo é gerado a partir do <i>front-matter</i> dos fichamentos
                (<code>build_kg.py</code>). Arraste os nós, use a roda para dar zoom e
                clique para inspecionar. <a href={api.kgViewUrl()} target="_blank"
                  rel="noreferrer">abrir em nova aba ↗</a>
              </p>
            </section>
            {kgError ? (
              <section className="card">
                <p className="error">{kgError}</p>
                <p className="hint">
                  A base de conhecimento vive no repositório da tese. Defina{" "}
                  <code>FALCO_THESIS_ROOT</code> apontando para a pasta da tese e rode{" "}
                  <code>python fichamentos/build_kg.py</code> para gerar o grafo.
                </p>
              </section>
            ) : (
              <section className="card kg-frame">
                <iframe title="Knowledge Graph — fichamentos FALCO"
                  src={api.kgViewUrl()} />
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}
