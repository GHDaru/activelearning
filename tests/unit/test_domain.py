"""Testes do domínio puro: sem rede, sem GPU, sem chaves de API."""
import numpy as np
import pytest

from activelearning.domain.annotation import Annotation, OracleUsage
from activelearning.domain.instances import CategorySchema, Instance, Label, normalize_label
from activelearning.domain.metrics import LearningCurve, lce, mcnemar_test, wilson_interval
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

    def test_json_schema_has_enum_and_strict(self):
        schema = CategorySchema.from_raw(["cerveja", "arroz"])
        json_schema = schema.to_json_schema()
        enum = json_schema["schema"]["properties"]["predicted_category"]["enum"]
        assert sorted(enum) == sorted(schema.values)
        # strict exige todas as propriedades em required (OpenAI structured outputs);
        # sem strict o enum vira sugestão e o modelo pode desviar (visto no piloto)
        assert json_schema["strict"] is True
        assert sorted(json_schema["schema"]["required"]) == sorted(
            json_schema["schema"]["properties"]
        )

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
        total = OracleUsage(10, 5, 1.0, 0.01, cached_input_tokens=8) + OracleUsage(
            20, 10, 2.0, 0.02, cached_input_tokens=18
        )
        assert total.input_tokens == 30
        assert total.cost_usd == pytest.approx(0.03)
        assert total.cached_input_tokens == 26
        assert total.cache_hit_rate == pytest.approx(26 / 30)

    def test_cache_hit_rate_zero_without_input(self):
        assert OracleUsage().cache_hit_rate == 0.0


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


class TestExtractJson:
    def test_plain_json(self):
        from activelearning.adapters.oracles.openai_compatible import extract_json

        assert extract_json('{"predicted_category": "arroz"}') == {
            "predicted_category": "arroz"
        }

    def test_markdown_fenced_json(self):
        from activelearning.adapters.oracles.openai_compatible import extract_json

        text = '```json\n{"predicted_category": "arroz", "rationale": "x"}\n```'
        assert extract_json(text)["predicted_category"] == "arroz"

    def test_json_with_thinking_block_and_prose(self):
        from activelearning.adapters.oracles.openai_compatible import extract_json

        text = (
            "<think>categoria provavel eh arroz</think>\n"
            'Aqui está: {"predicted_category": "arroz"} espero ter ajudado'
        )
        assert extract_json(text)["predicted_category"] == "arroz"

    def test_garbage_raises(self):
        import pytest as _pytest

        from activelearning.adapters.oracles.openai_compatible import extract_json

        with _pytest.raises(Exception):
            extract_json("sem json aqui")


class TestBatchedOracleHelpers:
    def test_parse_batch_payload_maps_by_index(self):
        from activelearning.adapters.oracles.openai_compatible import parse_batch_payload

        payload = {
            "classifications": [
                {"index": 2, "predicted_category": "arroz"},
                {"index": 1, "predicted_category": "cerveja"},
                {"index": 99, "predicted_category": "fora do range"},
                {"index": "x", "predicted_category": "indice invalido"},
            ]
        }
        mapped = parse_batch_payload(payload, n_items=3)
        assert mapped[1]["predicted_category"] == "cerveja"
        assert mapped[2]["predicted_category"] == "arroz"
        assert 3 not in mapped and 99 not in mapped

    def test_split_usage_preserves_totals(self):
        from activelearning.adapters.oracles.openai_compatible import split_usage

        total = OracleUsage(103, 47, 9.0, 0.9, cached_input_tokens=52)
        parts = split_usage(total, 4)
        assert len(parts) == 4
        assert sum(p.input_tokens for p in parts) == 103
        assert sum(p.output_tokens for p in parts) == 47
        assert sum(p.cached_input_tokens for p in parts) == 52
        assert sum(p.cost_usd for p in parts) == pytest.approx(0.9)

    def test_batch_json_schema_keeps_enum_and_is_size_independent(self):
        from activelearning.adapters.oracles.openai_compatible import batch_json_schema

        schema = CategorySchema.from_raw(["cerveja", "arroz"])
        batched = batch_json_schema(schema, constrained=True)
        items = batched["schema"]["properties"]["classifications"]["items"]
        assert sorted(items["properties"]["predicted_category"]["enum"]) == sorted(
            schema.values
        )
        assert "minItems" not in batched["schema"]["properties"]["classifications"]


class TestSchemaFreeMode:
    def test_free_schema_has_no_enum(self):
        schema = CategorySchema.from_raw(["cerveja", "arroz"])
        free = schema.to_json_schema(constrained=False)
        assert "enum" not in free["schema"]["properties"]["predicted_category"]
        constrained = schema.to_json_schema(constrained=True)
        assert "enum" in constrained["schema"]["properties"]["predicted_category"]


class TestStatistics:
    def test_wilson_interval_contains_proportion(self):
        low, high = wilson_interval(85, 100)
        assert low < 0.85 < high
        assert 0.76 < low < 0.80 and 0.90 < high < 0.92

    def test_wilson_extremes_stay_in_unit_interval(self):
        assert wilson_interval(0, 50)[0] == 0.0
        assert wilson_interval(50, 50)[1] == 1.0

    def test_mcnemar_no_discordance_is_one(self):
        assert mcnemar_test(0, 0) == 1.0

    def test_mcnemar_symmetric_discordance_not_significant(self):
        assert mcnemar_test(20, 20) > 0.8

    def test_mcnemar_asymmetric_discordance_significant(self):
        assert mcnemar_test(40, 5) < 0.001

    def test_mcnemar_small_n_uses_exact_binomial(self):
        p = mcnemar_test(9, 1)
        assert 0.02 < p < 0.03  # binomial exato bicaudal para 1 de 10


class TestInstance:
    def test_requires_id_and_text(self):
        with pytest.raises(ValueError):
            Instance(id="", text="abc")
        with pytest.raises(ValueError):
            Instance(id="1", text="  ")

    def test_normalize_label_helper(self):
        assert normalize_label("Água c/ Gás") == "agua c/ gas"
