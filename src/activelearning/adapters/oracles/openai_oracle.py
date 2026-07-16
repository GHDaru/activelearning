"""Adapter OraclePort para a API OpenAI, com saída estruturada restrita por enum."""
from __future__ import annotations

import json
import os
import time

from ...domain.annotation import Annotation, OracleUsage
from ...domain.instances import CategorySchema, Instance
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, user_prompt

# Preços por 1M tokens (entrada, saída) — atualizar conforme tabela vigente.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class OpenAIOracle:
    """Oráculo via OpenAI Chat Completions com ``response_format`` json_schema.

    O ``enum`` de ``predicted_category`` vem de CategorySchema.to_json_schema(),
    então o modelo é estruturalmente impedido de responder fora da lista.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI  # import tardio: adapter, não domínio

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Variável de ambiente {api_key_env} não definida.")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self.oracle_id = f"openai:{model}@T{temperature}"
        self.prompt_version = PROMPT_VERSION

    def annotate(self, batch: list[Instance], schema: CategorySchema) -> list[Annotation]:
        json_schema = schema.to_json_schema()
        return [self._annotate_one(instance, schema, json_schema) for instance in batch]

    def _annotate_one(
        self, instance: Instance, schema: CategorySchema, json_schema: dict
    ) -> Annotation:
        started = time.monotonic()
        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(instance)},
                    ],
                    response_format={"type": "json_schema", "json_schema": json_schema},
                )
                raw = response.choices[0].message.content or ""
                payload = json.loads(raw)
                label = schema.validate(str(payload.get("predicted_category", "")))
                usage = OracleUsage(
                    input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    output_tokens=response.usage.completion_tokens if response.usage else 0,
                    latency_seconds=time.monotonic() - started,
                    cost_usd=self._cost(response.usage),
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

    def _cost(self, usage) -> float:  # noqa: ANN001 - tipo do SDK
        if usage is None:
            return 0.0
        prices = _PRICING_USD_PER_MTOK.get(self._model)
        if prices is None:
            return 0.0
        return (usage.prompt_tokens * prices[0] + usage.completion_tokens * prices[1]) / 1e6
