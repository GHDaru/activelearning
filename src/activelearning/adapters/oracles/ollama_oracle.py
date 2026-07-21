"""Adapter OraclePort para modelos locais via Ollama, com saída restrita por enum.

Ollama suporta ``format`` com JSON Schema desde 0.5 — o enum é imposto na geração
(constrained decoding), então mesmo modelos pequenos (gemma3, qwen2.5) não podem
responder fora da lista.
"""
from __future__ import annotations

import json
import time

from ...domain.annotation import Annotation, OracleUsage
from ...domain.instances import CategorySchema, Instance
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, user_prompt


class OllamaOracle:
    """Oráculo LLM local via Ollama (modelos abertos, sem chave nem custo).

    Útil para rodar o laço offline com um LLM real; exige o serviço Ollama no
    ``host`` e o modelo já baixado (``ollama pull <model>``).
    """

    def __init__(
        self,
        model: str = "gemma3",
        temperature: float = 0.0,
        host: str | None = None,
        max_retries: int = 2,
    ) -> None:
        import ollama  # import tardio: adapter, não domínio

        self._client = ollama.Client(host=host) if host else ollama.Client()
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self.oracle_id = f"ollama:{model}@T{temperature}"
        self.prompt_version = PROMPT_VERSION

    def annotate(self, batch: list[Instance], schema: CategorySchema) -> list[Annotation]:
        output_schema = schema.to_json_schema()["schema"]
        return [self._annotate_one(instance, schema, output_schema) for instance in batch]

    def _annotate_one(
        self, instance: Instance, schema: CategorySchema, output_schema: dict
    ) -> Annotation:
        started = time.monotonic()
        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(instance)},
                    ],
                    format=output_schema,
                    options={"temperature": self._temperature},
                )
                raw = response["message"]["content"]
                payload = json.loads(raw)
                label = schema.validate(str(payload.get("predicted_category", "")))
                usage = OracleUsage(
                    input_tokens=int(response.get("prompt_eval_count", 0) or 0),
                    output_tokens=int(response.get("eval_count", 0) or 0),
                    latency_seconds=time.monotonic() - started,
                    cost_usd=0.0,  # execução local
                )
                return Annotation(
                    instance_id=instance.id,
                    label=label,
                    oracle_id=self.oracle_id,
                    prompt_version=self.prompt_version,
                    raw_response=raw,
                    rationale=str(payload.get("rationale", "")),
                    usage=usage,
                )
            except Exception as exc:  # noqa: BLE001 - fronteira de rede
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_retries:
                    time.sleep(2**attempt)
        return Annotation(
            instance_id=instance.id,
            label=None,
            oracle_id=self.oracle_id,
            prompt_version=self.prompt_version,
            raw_response=f"ERROR after {self._max_retries} attempts: {last_error}",
            usage=OracleUsage(latency_seconds=time.monotonic() - started),
        )
