"""Backend read-only do FALCO para o Vercel (função serverless).

Serve APENAS a base de conhecimento (grafo de fichamentos) a partir de um
snapshot estático em ``api/data/`` — sem banco, sem subprocessos, sem PDFs
(direitos autorais). O backend completo (rodar experimentos, uploads, servir
PDF) roda localmente ou num host de contêiner; veja DEPLOY_AND_PUBLISH.md.

O front chama os mesmos caminhos ``/api/kg/*``; em produção o Vercel roteia
``/api/(.*)`` para esta função (ver vercel.json).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

DATA = Path(__file__).resolve().parent / "data"

app = FastAPI(title="FALCO — base de conhecimento (read-only)", version="0.1.0")


def _graph() -> dict:
    kg = DATA / "kg.json"
    if not kg.exists():
        raise HTTPException(status_code=404, detail="kg.json não empacotado")
    return json.loads(kg.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": "hosted-readonly"}


@app.get("/api/kg/summary")
def kg_summary() -> dict:
    g = _graph()
    by_node = Counter(n.get("type", "?") for n in g.get("nodes", []))
    by_edge = Counter(e.get("type", "?") for e in g.get("edges", []))
    return {
        "n_nodes": len(g.get("nodes", [])),
        "n_edges": len(g.get("edges", [])),
        "n_artigos": by_node.get("artigo", 0),
        "n_pendentes": by_node.get("artigo-pendente", 0),
        "by_node_type": dict(by_node),
        "by_edge_type": dict(by_edge),
        "generated_at": (DATA / "kg.json").stat().st_mtime,
    }


@app.get("/api/kg")
def kg_graph() -> dict:
    return _graph()


@app.get("/api/kg/view")
def kg_view():
    html = DATA / "kg.html"
    if not html.exists():
        raise HTTPException(status_code=404, detail="kg.html não empacotado")
    return FileResponse(html, media_type="text/html")
