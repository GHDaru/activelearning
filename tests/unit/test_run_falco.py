"""Testes do runner FALCO (fases, transição por validação, troca de oráculo)."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402
from activelearning.adapters.oracles.simulated_oracle import SimulatedOracle  # noqa: E402
from activelearning.adapters.strategies.drisl import TfidfSvdEncoder, drisl_select  # noqa: E402
from activelearning.application.run_falco import run_falco  # noqa: E402
from activelearning.domain.instances import CategorySchema, Instance, Label  # noqa: E402

VOCAB = {
    "cerveja": ["cerveja lata", "cerv brahma", "cerveja skol 350"],
    "arroz": ["arroz tipo1", "arroz camil 5kg", "arroz branco"],
    "feijao": ["feijao preto", "feijao carioca", "feijao 1kg"],
    "sabao": ["sabao em po", "sabao barra", "sabao ype"],
}


def _make_data(n_per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    pool, val, test = [], [], []
    i = 0
    for label, seeds_text in VOCAB.items():
        for k in range(n_per_class):
            text = f"{seeds_text[k % len(seeds_text)]} v{rng.integers(0, 999)}"
            inst = Instance(id=f"f-{i}", text=text, gold_label=Label(label))
            if k < n_per_class - 10:
                pool.append(inst)
            elif k < n_per_class - 5:
                val.append(inst)
            else:
                test.append(inst)
            i += 1
    return pool, val, test, CategorySchema.from_raw(list(VOCAB), include_rare=False)


def _drisl(texts, k):
    return drisl_select(texts, k, TfidfSvdEncoder(seed=1), seed=1).indices


def test_phases_and_budget_respected():
    pool, val, test, schema = _make_data()
    r = run_falco(
        pool, val, test, schema, PVBinClassifier,
        oracle_initial=SimulatedOracle(noise=0.2, seed=1),
        oracle_advanced=SimulatedOracle(noise=0.0, seed=1),
        drisl_selector=_drisl, budget=70, batch_size=10, seed=1,
        stagnation_patience=2,
    )
    assert r.n_labeled + r.invalid_labels <= 70
    assert r.phase_boundaries["fase1"] >= 1
    assert r.phase_boundaries["fase2"] >= r.phase_boundaries["fase1"]
    assert r.phase_boundaries["fase3"] >= r.phase_boundaries["fase2"]
    assert r.oracle_calls["initial"] > 0


def test_without_advanced_oracle_phase3_is_empty():
    pool, val, test, schema = _make_data()
    r = run_falco(
        pool, val, test, schema, PVBinClassifier,
        oracle_initial=SimulatedOracle(noise=0.1, seed=2),
        oracle_advanced=None, drisl_selector=_drisl,
        budget=60, batch_size=10, seed=2, stagnation_patience=2,
    )
    assert r.phase_boundaries["fase3"] == r.phase_boundaries["fase2"]
    assert r.oracle_calls["advanced"] == 0


def test_advanced_oracle_used_after_stagnation():
    pool, val, test, schema = _make_data()
    r = run_falco(
        pool, val, test, schema, PVBinClassifier,
        oracle_initial=SimulatedOracle(noise=0.5, seed=3),  # ruim: estagna rápido
        oracle_advanced=SimulatedOracle(noise=0.0, seed=3),
        drisl_selector=_drisl, budget=80, batch_size=10, seed=3,
        stagnation_patience=1,
    )
    assert r.oracle_calls["advanced"] > 0
    assert r.phase_boundaries["fase3"] > r.phase_boundaries["fase2"]


def test_decisions_use_validation_not_test():
    # curva de decisão registrada é a de validação; curva de teste só relato.
    pool, val, test, schema = _make_data()
    r = run_falco(
        pool, val, test, schema, PVBinClassifier,
        oracle_initial=SimulatedOracle(noise=0.0, seed=4),
        drisl_selector=_drisl, budget=50, batch_size=10, seed=4,
        stagnation_patience=2,
    )
    assert len(r.curve_val_macro_f1) == len(r.curve_macro_f1)
    assert all("val_macro_f1" in rec for rec in r.records)


def test_fallback_random_l0_without_drisl():
    pool, val, test, schema = _make_data()
    r = run_falco(
        pool, val, test, schema, PVBinClassifier,
        oracle_initial=SimulatedOracle(noise=0.0, seed=5),
        drisl_selector=None, budget=40, batch_size=10, seed=5,
    )
    assert r.phase_boundaries["fase1"] >= 1
