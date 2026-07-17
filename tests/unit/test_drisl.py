"""Testes do DRI-SL com encoder leve (TF-IDF+SVD) — sem download de modelo."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from activelearning.adapters.strategies.drisl import (  # noqa: E402
    DrislResult,
    TfidfSvdEncoder,
    drisl_select,
)

# quatro "temas" lexicalmente distintos simulam agrupamentos semânticos
TEXTS = (
    [f"cerveja lata gelada pilsen {i}" for i in range(30)]
    + [f"arroz branco tipo1 graos {i}" for i in range(30)]
    + [f"sabao em po roupas limpeza {i}" for i in range(30)]
    + [f"chocolate barra cacau doce {i}" for i in range(10)]  # grupo minoritário
)


def test_selects_exact_target_size():
    r = drisl_select(TEXTS, target_size=20, encoder=TfidfSvdEncoder(seed=1), seed=1)
    assert isinstance(r, DrislResult)
    assert len(r.indices) == 20
    assert len(set(r.indices)) == 20  # sem repetição


def test_deterministic_given_seed():
    a = drisl_select(TEXTS, 16, TfidfSvdEncoder(seed=2), seed=2)
    b = drisl_select(TEXTS, 16, TfidfSvdEncoder(seed=2), seed=2)
    assert a.indices == b.indices


def test_covers_multiple_clusters():
    r = drisl_select(TEXTS, 12, TfidfSvdEncoder(seed=3), n_clusters=4, seed=3)
    clusters_hit = {int(r.cluster_of[i]) for i in r.indices}
    assert len(clusters_hit) >= 3  # cobre a maioria dos grupos, não só o maior


def test_allocation_proportional_to_cluster_size():
    r = drisl_select(TEXTS, 20, TfidfSvdEncoder(seed=4), n_clusters=4, seed=4)
    assert sum(r.allocation) == 20
    # nenhum grupo não-vazio deve ficar com quota desproporcional ao seu tamanho:
    # o maior aloca mais que o menor
    sizes = np.bincount(r.cluster_of, minlength=r.n_clusters)
    big, small = int(np.argmax(sizes)), int(np.argmin(sizes))
    assert r.allocation[big] >= r.allocation[small]


def test_lexical_novelty_avoids_near_duplicates():
    texts = ["cerveja lata 350"] * 20 + ["cerveja garrafa 600 especial retornavel"]
    r = drisl_select(texts, 2, TfidfSvdEncoder(seed=5), n_clusters=1, seed=5)
    # com 2 vagas num grupo só, a segunda escolha deve ser o texto que introduz
    # termos novos, não mais uma cópia de "cerveja lata 350"
    assert 20 in r.indices


def test_rejects_bad_target():
    with pytest.raises(ValueError):
        drisl_select(TEXTS, 0, TfidfSvdEncoder())
    with pytest.raises(ValueError):
        drisl_select(TEXTS, len(TEXTS) + 1, TfidfSvdEncoder())
