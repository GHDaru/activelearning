"""Testes do PVBinClassifier (porte do legado, config Binary [1,2] sem norma)."""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402

TEXTS = [
    "CERVEJA BRAHMA LATA 350ML", "CERV SKOL LT 350", "CERVEJA HEINEKEN 600ML",
    "ARROZ TIO JOAO TIPO1 5KG", "ARROZ BRANCO CAMIL 1KG", "ARROZ PARB 5KG",
    "FEIJAO PRETO CAMIL 1KG", "FEIJAO CARIOCA 1KG",
]
LABELS = ["cerveja", "cerveja", "cerveja", "arroz", "arroz", "arroz", "feijao", "feijao"]


@pytest.fixture()
def clf():
    return PVBinClassifier().fit(TEXTS, LABELS)


def test_classes_sorted_deterministic(clf):
    assert clf.classes_ == ["arroz", "cerveja", "feijao"]


def test_predict_recovers_training_signal(clf):
    assert clf.predict(["CERVEJA ANTARCTICA LATA"]) == ["cerveja"]
    assert clf.predict(["ARROZ INTEGRAL 2KG"]) == ["arroz"]


def test_predict_proba_shape_and_simplex(clf):
    proba = clf.predict_proba(["CERV LT", "FEIJAO 1KG"])
    assert proba.shape == (2, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_proba_argmax_matches_predict(clf):
    texts = ["CERVEJA LATA", "ARROZ 5KG", "FEIJAO PRETO"]
    proba = clf.predict_proba(texts)
    by_proba = [clf.classes_[i] for i in proba.argmax(axis=1)]
    assert by_proba == clf.predict(texts)


def test_binary_ignores_term_repetition():
    # representação binária: repetir um termo na consulta não muda o escore
    clf = PVBinClassifier().fit(TEXTS, LABELS)
    a = clf._scores(["CERVEJA GELADA"])
    b = clf._scores(["CERVEJA CERVEJA GELADA"])
    assert np.allclose(a, b)


def test_determinism_across_fits():
    a = PVBinClassifier().fit(TEXTS, LABELS).predict(["MISTERIO TOTAL 123"])
    b = PVBinClassifier().fit(list(TEXTS), list(LABELS)).predict(["MISTERIO TOTAL 123"])
    assert a == b  # sem dependência de ordem de set()


def test_requires_fit():
    with pytest.raises(RuntimeError):
        PVBinClassifier().predict(["x"])


def test_rejects_empty_or_mismatched():
    with pytest.raises(ValueError):
        PVBinClassifier().fit([], [])
    with pytest.raises(ValueError):
        PVBinClassifier().fit(["a"], ["x", "y"])
