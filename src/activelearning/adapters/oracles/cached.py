"""CachedOracle — cache persistente de anotações por instância.

Envolve qualquer OraclePort: instância já anotada (mesmo oráculo interno) é
servida do cache JSONL sem nova chamada de API. Essencial em execuções longas
(orçamentos de dezenas de milhares) e para compartilhar rótulos entre ciclos
com classificadores diferentes — o rótulo do oráculo depende só da instância,
não do classificador que a selecionou.

O arquivo de cache registra o oracle_id interno; um cache criado com outro
oráculo é rejeitado (evita misturar proveniências).
"""
from __future__ import annotations

import json
from pathlib import Path

from activelearning.domain.instances import CategorySchema, Instance


class CachedOracle:
    """Envolve outro oráculo e persiste as anotações em JSONL, por instância.

    Numa segunda passada, respostas já vistas são servidas do cache (evita
    re-pagar o LLM); só as instâncias novas chamam o oráculo interno. O cache é
    validado contra o ``oracle_id`` para não misturar proveniências.
    """

    def __init__(self, inner, cache_path: Path) -> None:
        self._inner = inner
        self._path = Path(cache_path)
        self._cache: dict[str, dict] = {}
        self.calls_inner = 0
        self.hits = 0
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                # o lote (@bN) é instrumento de vazão, não de rótulo — calibração
                # pareada b20×b50 sem diferença (p=0,58); ignora-o na checagem
                def _base(oid: str) -> str:
                    return oid.split("@b")[0] if oid else oid
                if _base(rec.get("oracle_id")) != _base(inner.oracle_id):
                    raise ValueError(
                        f"cache {self._path} pertence a {rec.get('oracle_id')!r}, "
                        f"não a {inner.oracle_id!r}")
                self._cache[rec["instance_id"]] = rec

    @property
    def oracle_id(self) -> str:  # proveniência transparente
        return self._inner.oracle_id

    @property
    def prompt_version(self) -> str:
        return getattr(self._inner, "prompt_version", "")

    def annotate(self, instances: list[Instance], schema: CategorySchema):
        from activelearning.domain.annotation import Annotation
        from activelearning.domain.instances import Label

        missing = [i for i in instances if i.id not in self._cache]
        if missing:
            fresh = self._inner.annotate(missing, schema)
            self.calls_inner += len(missing)
            with self._path.open("a", encoding="utf-8") as fh:
                for ann in fresh:
                    rec = {
                        "instance_id": ann.instance_id,
                        "label": str(ann.label) if ann.label else None,
                        "oracle_id": self._inner.oracle_id,
                    }
                    self._cache[ann.instance_id] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        missing_ids = {m.id for m in missing}
        out = []
        for i in instances:
            rec = self._cache[i.id]
            if i.id not in missing_ids:
                self.hits += 1
            label = Label(rec["label"]) if rec["label"] else None
            out.append(Annotation(instance_id=i.id, label=label,
                                  oracle_id=rec["oracle_id"],
                                  prompt_version=self.prompt_version,
                                  raw_response="(cache)" if i.id not in missing_ids else ""))
        return out
