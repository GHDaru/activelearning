"""Testes do SgdTextClassifier (porta TaskClassifier)."""
import numpy as np
import pytest

from activelearning.adapters.classifiers.sgd_text import SgdTextClassifier

TEXTS = ["arroz branco tipo 1", "arroz agulhinha 5kg", "feijao preto 1kg",
         "feijao carioca", "leite integral 1l", "leite desnatado"]
LABELS = ["arroz", "arroz", "feijao", "feijao", "leite", "leite"]


def test_fit_predict_and_proba_shape():
    clf = SgdTextClassifier(seed=0).fit(TEXTS, LABELS)
    assert clf.classes_ == sorted(set(LABELS))
    proba = clf.predict_proba(["arroz parboilizado"])
    assert proba.shape == (1, 3)
    assert np.isclose(proba.sum(), 1.0, atol=1e-6)
    assert clf.predict(["arroz parboilizado"]) == ["arroz"]


def test_deterministic_by_seed():
    p1 = SgdTextClassifier(seed=7).fit(TEXTS, LABELS).predict_proba(TEXTS)
    p2 = SgdTextClassifier(seed=7).fit(TEXTS, LABELS).predict_proba(TEXTS)
    assert np.allclose(p1, p2)


def test_errors_before_fit_and_on_empty():
    clf = SgdTextClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(["x"])
    with pytest.raises(ValueError):
        clf.fit([], [])
