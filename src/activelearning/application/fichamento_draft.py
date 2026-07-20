"""Geração de um *rascunho* de fichamento a partir de um PDF.

Segue o SKILL de fichamento da tese (``.claude/skills/fichamento``): extrai texto
e metadados, deriva a chave BibTeX/``id`` do nó e escreve um rascunho de
front-matter no padrão KG-ready. É deliberadamente conservador — nada é
inventado: campos incertos ficam vazios, com avisos, e ``status: a-ler`` marca
que o autor precisa revisar antes de o artigo entrar na tese.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


@dataclass
class FichamentoDraft:
    key: str
    md_path: str
    pdf_path: str
    title: str
    authors: list[str]
    year: int | None
    doi: str
    n_chars: int
    kg_regenerated: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _slug_surname(author: str) -> str:
    """'Sobrenome, Nome' ou 'Nome Sobrenome' -> Sobrenome sem acento/pontuação."""
    author = author.strip()
    surname = author.split(",")[0] if "," in author else author.split()[-1] if author else ""
    surname = unicodedata.normalize("NFKD", surname).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z]", "", surname).capitalize()


def extract_pdf(pdf_bytes: bytes) -> tuple[str, dict]:
    """Retorna (texto das primeiras páginas, metadados embutidos)."""
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = reader.pages[:4]
    text = "\n".join((p.extract_text() or "") for p in pages)
    meta = {}
    if reader.metadata:
        meta = {
            "title": (reader.metadata.title or "").strip(),
            "author": (reader.metadata.author or "").strip(),
        }
    return text, meta


def guess_fields(text: str, meta: dict, filename: str) -> dict:
    """Heurísticas conservadoras para título, autores, ano e DOI."""
    warnings: list[str] = []
    head = text[:2500]

    doi_m = _DOI_RE.search(text)
    doi = doi_m.group(0).rstrip(".") if doi_m else ""
    if not doi:
        warnings.append("DOI não localizado no texto — preencher à mão.")

    year = None
    ym = _YEAR_RE.search(head)
    if ym:
        year = int(ym.group(0))
    else:
        warnings.append("Ano não localizado — preencher à mão.")

    title = meta.get("title") or ""
    if not title or len(title) < 8:
        # primeira linha não trivial da primeira página
        for line in head.splitlines():
            s = line.strip()
            if len(s) >= 15 and not s.isupper() and not _DOI_RE.search(s):
                title = s
                break
    if not title:
        title = Path(filename).stem.replace("_", " ")
        warnings.append("Título incerto — derivado do nome do arquivo; revisar.")

    authors: list[str] = []
    if meta.get("author"):
        parts = re.split(r"[;,]| and ", meta["author"])
        authors = [a.strip() for a in parts if a.strip()]
    if not authors:
        warnings.append("Autores não extraídos do PDF — preencher à mão.")

    return {"title": title.strip(), "authors": authors, "year": year,
            "doi": doi, "warnings": warnings}


def _unique_key(base: str, fich_dir: Path) -> str:
    key = base
    suffix = ord("a")
    while (fich_dir / f"{key}.md").exists():
        key = f"{base}{chr(suffix)}"
        suffix += 1
    return key


def make_key(fields: dict, filename: str, fich_dir: Path) -> str:
    surname = _slug_surname(fields["authors"][0]) if fields["authors"] else ""
    year = fields["year"] or ""
    base = f"{surname}{year}" if surname else Path(filename).stem
    base = re.sub(r"[^A-Za-z0-9]", "", base) or "Artigo"
    return _unique_key(base, fich_dir)


def _yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"


def render_front_matter(key: str, fields: dict, pdf_rel: str) -> str:
    authors = ", ".join(f'"{a}"' for a in fields["authors"])
    return f"""---
id: {key}
title: "{fields['title'].replace('"', "'")}"
authors: [{authors}]
year: {fields['year'] or 2026}
venue: ""
doi: "{fields['doi']}"
pdf: {pdf_rel}
paper_type: metodo
pillars: [P3]
status: a-ler
proposes: []
uses_methods: []
datasets: []
metrics: []
tasks: [classificacao-de-texto]
models: []
extends: []
compares_with: []
contradicts: []
builds_on: []
falco_relation:
  - type: compara
    target: ""
    note: "(preencher: relação com FALCO/DRI-SL/LCE)"
---

# {fields['title']}

## Resumo (5-8 linhas, com as MINHAS palavras)
<!-- rascunho automático — revisar/reescrever a partir do PDF -->

## Claims relevantes
| # | Claim | Evidência | Uso na tese |
|---|-------|-----------|-------------|
| C1 |  | §, Tab., p. | cap./seção |

## Números que posso citar

## Crítica / limitações (minha leitura)

## Ideias que gera para a tese
"""


def _regenerate_kg(thesis_root: Path, warnings: list[str]) -> bool:
    """Regenera kg.json/kg.html importando build_kg.py (best-effort)."""
    import importlib.util

    build_py = thesis_root / "fichamentos" / "build_kg.py"
    if not build_py.exists():
        warnings.append("build_kg.py não encontrado — regenerar o KG manualmente.")
        return False
    try:
        import json

        spec = importlib.util.spec_from_file_location("_build_kg", build_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        graph = mod.build()
        (build_py.parent / "kg.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (build_py.parent / "kg.html").write_text(
            mod.render_html(graph), encoding="utf-8"
        )
        return True
    except Exception as exc:  # pyyaml ausente etc.
        warnings.append(f"KG não regenerado automaticamente ({exc}). Rode build_kg.py.")
        return False


def generate_draft(pdf_bytes: bytes, filename: str, thesis_root: Path) -> FichamentoDraft:
    fich_dir = thesis_root / "fichamentos"
    pdf_dir = thesis_root / "referencias-pdf"
    if not fich_dir.exists():
        raise FileNotFoundError(f"pasta de fichamentos não encontrada: {fich_dir}")

    text, meta = extract_pdf(pdf_bytes)
    fields = guess_fields(text, meta, filename)
    warnings = list(fields["warnings"])
    key = make_key(fields, filename, fich_dir)

    pdf_dir.mkdir(exist_ok=True)
    pdf_rel = f"referencias-pdf/{key}.pdf"
    (thesis_root / pdf_rel).write_bytes(pdf_bytes)

    md_path = fich_dir / f"{key}.md"
    md_path.write_text(render_front_matter(key, fields, pdf_rel), encoding="utf-8")

    if len(text) < 200:
        warnings.append("Pouco texto extraído (PDF escaneado?) — conferir manualmente.")

    kg_ok = _regenerate_kg(thesis_root, warnings)
    return FichamentoDraft(
        key=key, md_path=str(md_path), pdf_path=pdf_rel,
        title=fields["title"], authors=fields["authors"], year=fields["year"],
        doi=fields["doi"], n_chars=len(text), kg_regenerated=kg_ok, warnings=warnings,
    )
