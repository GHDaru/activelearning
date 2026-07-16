"""Testes do domínio puro: sem rede, sem GPU, sem chaves de API."""
import numpy as np
import pytest

from activelearning.domain.annotation import Annotation, OracleUsage
from activelearning.domain.instances import CategorySchema, Instance, Label, normalize_label
from activelearning.domain.metrics import LearningCurve, lce
from activelearning.domain.strategies import (
    entropy_scores,
    least_confidence_scores,
    select_top_k,
    smallest_margin_scores,
)


class TestLabel:
    def test_normalizes_accents_case_and_spaces(self):
        assert Label("  Bebida   LÁCTEA ").value == "bebida lactea"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            Label("   ")


class TestCategorySchema:
    def test_enum_contains_all_labels_plus_rare(self):
        schema = CategorySchema.from_raw(["Cerveja", "arroz"])
        assert set(schema.values) == {"cerveja", "arroz", "_rare_"}

    def test_json_schema_has_enum(self):
        schema = CategorySchema.from_raw(["cerveja", "arroz"])
        json_schema = schema.to_json_schema()
        enum = json_schema["schema"]["properties"]["predicted_category"]["enum"]
        assert sorted(enum) == sorted(schema.values)
        assert json_schema["schema"]["required"] == ["predicted_category"]

    def test_validate_accepts_phrasing_variants_of_registered_labels_only(self):
        schema = CategorySchema.from_raw(["ovo de pascoa"])
        assert schema.validate("OVO DE PÁSCOA") == Label("ovo de pascoa")
        # variação de fraseado fora do schema NÃO é aceita (vira invalid, não erro mudo)
        assert schema.validate("ovos de pascoa") is None


class TestAnnotation:
    def test_invalid_label_counts_explicitly(self):
        ann = Annotation(
            instance_id="x", label=None, oracle_id="o", prompt_version="v",
            raw_response="fora da lista",
        )
        assert not ann.is_valid_label
        assert ann.is_correct(Label("arroz")) is False

    def test_correct_against_gold(self):
        ann = Annotation(
            instance_id="x", label=Label("arroz"), oracle_id="o", prompt_version="v"
        )
        assert ann.is_correct(Label("ARROZ")) is True
        assert ann.is_correct(None) is None

    def test_usage_is_additive(self):
        total = OracleUsage(10, 5, 1.0, 0.01) + OracleUsage(20, 10, 2.0, 0.02)
        assert total.input_tokens == 30
        assert total.cost_usd == pytest.approx(0.03)


class TestStrategies:
    def test_entropy_uniform_is_maximal(self):
        probs = np.array([[0.5, 0.5], [0.99, 0.01]])
        scores = entropy_scores(probs)
        assert scores[0] > scores[1]
        assert scores[0] == pytest.approx(1.0)

    def test_least_confidence_and_margin_rank_uncertain_first(self):
        probs = np.array([[0.4, 0.35, 0.25], [0.9, 0.05, 0.05]])
        assert least_confidence_scores(probs)[0] > least_confidence_scores(probs)[1]
        assert smallest_margin_scores(probs)[0] > smallest_margin_scores(probs)[1]

    def test_select_top_k_orders_descending(self):
        scores = np.array([0.1, 0.9, 0.5])
        assert select_top_k(scores, 2).tolist() == [1, 2]


class TestLCE:
    def test_perfect_curve_from_zero_start_is_below_one(self):
        curve = LearningCurve()
        curve.append(100, 0.9)
        curve.append(200, 0.9)
        # AUC_real cobre [100,200]; ideal cobre [0,200] -> 0.9*100 / (0.9*200) = 0.5
        assert lce(curve, baseline_performance=0.9) == pytest.approx(0.5)

    def test_three_points_uses_simpson(self):
        curve = LearningCurve()
        for size, score in [(0, 0.0), (50, 0.5), (100, 1.0)]:
            curve.append(size + 1, score)  # estritamente crescente a partir de 1
        value = lce(curve, baseline_performance=1.0)
        assert 0.4 < value < 0.6

    def test_requires_two_points(self):
        curve = LearningCurve()
        curve.append(10, 0.5)
        with pytest.raises(ValueError):
            lce(curve, baseline_performance=1.0)


class TestInstance:
    def test_requires_id_and_text(self):
        with pytest.raises(ValueError):
            Instance(id="", text="abc")
        with pytest.raises(ValueError):
            Instance(id="1", text="  ")

    def test_normalize_label_helper(self):
        assert normalize_label("Água c/ Gás") == "agua c/ gas"
