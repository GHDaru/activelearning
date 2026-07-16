"""Use case E0: avaliar oráculos LLM contra rótulos-ouro, com custo e validade."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..domain.annotation import Annotation, OracleUsage
from ..domain.instances import CategorySchema, Instance
from ..ports.oracle import OraclePort


@dataclass
class OracleEvaluationReport:
    oracle_id: str
    prompt_version: str
    n_total: int = 0
    n_correct: int = 0
    n_invalid_label: int = 0
    usage: OracleUsage = field(default_factory=OracleUsage)
    per_class_correct: dict[str, int] = field(default_factory=dict)
    per_class_total: dict[str, int] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0

    @property
    def invalid_rate(self) -> float:
        return self.n_invalid_label / self.n_total if self.n_total else 0.0

    @property
    def macro_f1(self) -> float:
        """Macro-F1 calculado da matriz de confusão (classes do ouro)."""
        f1s: list[float] = []
        gold_classes = set(self.per_class_total)
        for cls in gold_classes:
            tp = self.confusion.get(cls, {}).get(cls, 0)
            fn = self.per_class_total.get(cls, 0) - tp
            fp = sum(
                row.get(cls, 0) for gold, row in self.confusion.items() if gold != cls
            )
            denom = 2 * tp + fp + fn
            f1s.append(2 * tp / denom if denom else 0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0

    @property
    def cost_per_1k_labels_usd(self) -> float:
        return (self.usage.cost_usd / self.n_total * 1000) if self.n_total else 0.0

    def to_summary(self) -> dict:
        return {
            "oracle_id": self.oracle_id,
            "prompt_version": self.prompt_version,
            "n_total": self.n_total,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "invalid_label_rate": round(self.invalid_rate, 4),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_cost_usd": round(self.usage.cost_usd, 4),
            "cost_per_1k_labels_usd": round(self.cost_per_1k_labels_usd, 4),
            "total_latency_seconds": round(self.usage.latency_seconds, 1),
        }


class EvaluateOracle:
    """Roda um oráculo sobre instâncias com gold_label e persiste anotações + relatório.

    Persistência incremental em JSONL: cada anotação é gravada assim que obtida,
    permitindo retomar execuções interrompidas (instâncias já presentes no arquivo
    são puladas).
    """

    def __init__(self, oracle: OraclePort, output_dir: Path) -> None:
        self._oracle = oracle
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = oracle.oracle_id.replace(":", "_").replace("/", "_").replace("@", "_")
        self._annotations_path = self._output_dir / f"annotations_{safe_id}.jsonl"
        self._report_path = self._output_dir / f"report_{safe_id}.json"

    def run(
        self, instances: list[Instance], schema: CategorySchema, batch_size: int = 25
    ) -> OracleEvaluationReport:
        done_ids = self._already_annotated()
        pending = [i for i in instances if i.id not in done_ids]
        gold_by_id = {i.id: i.gold_label for i in instances}

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            annotations = self._oracle.annotate(batch, schema)
            self._append(annotations)

        report = self._build_report(gold_by_id)
        self._report_path.write_text(
            json.dumps(report.to_summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def _already_annotated(self) -> set[str]:
        if not self._annotations_path.exists():
            return set()
        ids: set[str] = set()
        with self._annotations_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids

    def _append(self, annotations: list[Annotation]) -> None:
        with self._annotations_path.open("a", encoding="utf-8") as fh:
            for ann in annotations:
                record = asdict(ann)
                record["label"] = ann.label.value if ann.label else None
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_report(self, gold_by_id: dict) -> OracleEvaluationReport:
        report = OracleEvaluationReport(
            oracle_id=self._oracle.oracle_id,
            prompt_version=self._oracle.prompt_version,
        )
        with self._annotations_path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                gold = gold_by_id.get(record["instance_id"])
                if gold is None:
                    continue
                predicted = record.get("label")
                usage = record.get("usage", {})
                report.n_total += 1
                report.usage = report.usage + OracleUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    latency_seconds=usage.get("latency_seconds", 0.0),
                    cost_usd=usage.get("cost_usd", 0.0),
                )
                gold_value = gold.value
                report.per_class_total[gold_value] = (
                    report.per_class_total.get(gold_value, 0) + 1
                )
                if predicted is None:
                    report.n_invalid_label += 1
                    predicted = "__invalid__"
                if predicted == gold_value:
                    report.n_correct += 1
                    report.per_class_correct[gold_value] = (
                        report.per_class_correct.get(gold_value, 0) + 1
                    )
                report.confusion.setdefault(gold_value, {})
                report.confusion[gold_value][predicted] = (
                    report.confusion[gold_value].get(predicted, 0) + 1
                )
        return report
