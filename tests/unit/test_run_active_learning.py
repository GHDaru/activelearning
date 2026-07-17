"""Testes do laço de AL simulado (PVBin + SimulatedOracle, dados sintéticos)."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from activelearning.adapters.classifiers.pvbin import PVBinClassifier  # noqa: E402
from activelearning.adapters.oracles.simulated_oracle import SimulatedOracle  # noqa: E402
from activelearning.application.run_active_learning import (  # noqa: E402
    STRATEGIES,
    run_active_learning,
)
from activelearning.domain.instances import CategorySchema, Instance, Label  # noqa: E402

VOCAB = {
    "cerveja": ["cerveja lata", "cerv brahma", "cerveja skol 350", "cerveja gelada"],
    "arroz": ["arroz tipo1", "arroz camil 5kg", "arroz branco", "arroz parboilizado"],
    "feijao": ["feijao preto", "feijao carioca", "feijao 1kg", "feijao camil"],
    "sabao": ["sabao em po", "sabao barra", "sabao ype", "sabao neutro"],
}


def _make_data(n_per_class: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    pool, test = [], []
    i = 0
    for label, seeds_text in VOCAB.items():
        for k in range(n_per_class):
            base = seeds_text[k % len(seeds_text)]
            text = f"{base} v{rng.integers(0, 1000)}"
            inst = Instance(id=f"s-{i}", text=text, gold_label=Label(label))
            (pool if k < n_per_class - 5 else test).append(inst)
            i += 1
    schema = CategorySchema.from_raw(list(VOCAB), include_rare=False)
    return pool, test, schema


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_run_and_learn(strategy):
    pool, test, schema = _make_data()
    result = run_active_learning(
        pool, test, schema,
        classifier_factory=PVBinClassifier,
        oracle=SimulatedOracle(noise=0.0),
        strategy=strategy, budget=60, batch_size=10, initial_size=10, seed=1,
    )
    assert result.n_labeled == 60
    assert result.final_macro_f1 > 0.8  # tarefa separável com oráculo perfeito
    assert 0.0 < result.lce_macro_f1 <= 1.0
    assert len(result.records) == len(result.curve_macro_f1)


def test_deterministic_given_seed():
    pool, test, schema = _make_data()
    kw = dict(
        classifier_factory=PVBinClassifier, oracle=SimulatedOracle(noise=0.0),
        strategy="entropy", budget=50, batch_size=10, initial_size=10, seed=7,
    )
    a = run_active_learning(pool, test, schema, **kw)
    b = run_active_learning(pool, test, schema, **kw)
    assert a.curve_macro_f1.scores == b.curve_macro_f1.scores


def test_noisy_oracle_degrades_learning():
    pool, test, schema = _make_data()
    kw = dict(
        classifier_factory=PVBinClassifier, strategy="random",
        budget=60, batch_size=10, initial_size=10, seed=3,
    )
    clean = run_active_learning(pool, test, schema, oracle=SimulatedOracle(noise=0.0), **kw)
    noisy = run_active_learning(pool, test, schema, oracle=SimulatedOracle(noise=0.4, seed=3), **kw)
    assert noisy.final_macro_f1 < clean.final_macro_f1


def test_initial_indices_respected():
    pool, test, schema = _make_data()
    fixed = list(range(12))
    result = run_active_learning(
        pool, test, schema, classifier_factory=PVBinClassifier,
        oracle=SimulatedOracle(noise=0.0), strategy="random",
        budget=32, batch_size=10, initial_size=99, seed=5, initial_indices=fixed,
    )
    assert result.curve_macro_f1.l_sizes[0] == len(fixed)


def test_rejects_unknown_strategy_and_overbudget():
    pool, test, schema = _make_data()
    with pytest.raises(ValueError):
        run_active_learning(pool, test, schema, PVBinClassifier,
                            SimulatedOracle(), strategy="magica", budget=10)
    with pytest.raises(ValueError):
        run_active_learning(pool, test, schema, PVBinClassifier,
                            SimulatedOracle(), budget=10**9)
