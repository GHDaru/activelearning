#!/usr/bin/env python3
"""Atualiza o snapshot público do grafo (api/data/) a partir da tese.

Copia kg.json e kg.html de FALCO_THESIS_ROOT/fichamentos e injeta a flag
``window.KG_HOSTED=true`` no HTML (esconde o botão de PDF por direitos autorais).
Rode antes de publicar o Vercel quando os fichamentos mudarem:

    FALCO_THESIS_ROOT=../tesedaru python scripts/sync_public_kg.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS = Path(os.environ.get("FALCO_THESIS_ROOT", ROOT.parent / "tesedaru"))
FICH = THESIS / "fichamentos"
OUT = ROOT / "api" / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src_json, src_html = FICH / "kg.json", FICH / "kg.html"
    if not src_json.exists() or not src_html.exists():
        raise SystemExit(f"KG não encontrado em {FICH} — rode build_kg.py na tese primeiro.")
    shutil.copyfile(src_json, OUT / "kg.json")
    html = src_html.read_text(encoding="utf-8")
    if "window.KG_HOSTED" not in html:
        i = html.find("<script>\nconst GRAPH")
        if i == -1:
            i = html.find("<script>")
        html = html[:i] + "<script>window.KG_HOSTED=true;</script>\n" + html[i:]
    (OUT / "kg.html").write_text(html, encoding="utf-8")
    print(f"Snapshot atualizado em {OUT} (kg.json + kg.html hospedado).")


if __name__ == "__main__":
    main()
