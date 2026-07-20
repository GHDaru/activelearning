"""Testes das heurísticas de geração de rascunho de fichamento (sem I/O de rede)."""
from pathlib import Path

from activelearning.application.fichamento_draft import (
    guess_fields,
    make_key,
    render_front_matter,
)


def test_guess_fields_extrai_doi_ano_e_titulo():
    text = (
        "Deep Active Learning for Short Text\n"
        "Jane Doe and John Roe\n"
        "Published 2021. https://doi.org/10.1000/abcd.2021.99\n"
        "Abstract: we study ...\n"
    )
    f = guess_fields(text, {}, "paper.pdf")
    assert f["doi"] == "10.1000/abcd.2021.99"
    assert f["year"] == 2021
    assert f["title"] == "Deep Active Learning for Short Text"


def test_guess_fields_avisa_quando_incompleto():
    f = guess_fields("texto sem metadados uteis aqui", {}, "arquivo.pdf")
    assert f["doi"] == ""
    assert any("DOI" in w for w in f["warnings"])
    assert any("Ano" in w for w in f["warnings"])


def test_make_key_usa_sobrenome_e_ano(tmp_path: Path):
    f = {"authors": ["Doe, Jane"], "year": 2021}
    assert make_key(f, "x.pdf", tmp_path) == "Doe2021"


def test_make_key_desambigua_colisao(tmp_path: Path):
    (tmp_path / "Doe2021.md").write_text("x")
    f = {"authors": ["Doe, Jane"], "year": 2021}
    assert make_key(f, "x.pdf", tmp_path) == "Doe2021a"


def test_make_key_fallback_para_nome_do_arquivo(tmp_path: Path):
    f = {"authors": [], "year": None}
    assert make_key(f, "meu_artigo.pdf", tmp_path) == "meuartigo"


def test_render_front_matter_e_kg_ready():
    f = {"title": "T", "authors": ["Doe, Jane"], "year": 2021, "doi": "10.1/x"}
    md = render_front_matter("Doe2021", f, "referencias-pdf/Doe2021.pdf")
    assert "id: Doe2021" in md
    assert "status: a-ler" in md           # rascunho para revisão
    assert "falco_relation:" in md          # relação obrigatória com a tese
    assert 'pdf: referencias-pdf/Doe2021.pdf' in md
