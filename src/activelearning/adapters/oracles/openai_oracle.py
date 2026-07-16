"""Adapter OraclePort para a API OpenAI, com saída estruturada restrita por enum
e prompt caching.

Prompt caching (spec OpenAI):
- prefixos idênticos com >= 1024 tokens são cacheados automaticamente; mensagens,
  tools e o SCHEMA de saída estruturada contam para o prefixo;
- por isso a ordem das partes importa: system prompt (estático) + response_format
  (estático, inclui o enum de categorias) vêm antes; a única parte variável é a
  mensagem de usuário com a descrição do produto, ao final;
- ``prompt_cache_key`` roteia chamadas com o mesmo prefixo para o mesmo shard,
  aumentando a taxa de acerto (recomendação: ~15 req/min por chave);
- tokens servidos do cache chegam em ``usage.prompt_tokens_details.cached_tokens``
  e custam 50% do preço de input (família gpt-4o); não há custo de escrita.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from ...domain.annotation import Annotation, OracleUsage
from ...domain.instances import CategorySchema, Instance
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, user_prompt

# Preços por 1M tokens: (input, cached_input, output) — atualizar conforme tabela vigente.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
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
        cache_key = self._cache_key(json_schema)
        return [self._annotate_one(i, schema, json_schema, cache_key) for i in batch]

    def _cache_key(self, json_schema: dict) -> str:
        """Chave estável derivada do prefixo estático (prompt + schema/enum).

        Muda se (e somente se) o prefixo cacheável muda — schemas diferentes não
        disputam o mesmo shard de cache.
        """
        digest = hashlib.sha256(
            (SYSTEM_PROMPT + PROMPT_VERSION + json.dumps(json_schema, sort_keys=True)).encode()
        ).hexdigest()[:16]
        return f"falco-oracle-{digest}"

    def _annotate_one(
        self, instance: Instance, schema: CategorySchema, json_schema: dict, cache_key: str
    ) -> Annotation:
        started = time.monotonic()
        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    messages=[
                        # Prefixo ESTÁTICO primeiro (cacheável): system prompt.
                        # O response_format (schema+enum) também integra o prefixo.
                        {"role": "system", "content": SYSTEM_PROMPT},
                        # Única parte variável, ao final:
                        {"role": "user", "content": user_prompt(instance)},
                    ],
                    response_format={"type": "json_schema", "json_schema": json_schema},
                    prompt_cache_key=cache_key,
                )
                raw = response.choices[0].message.content or ""
                payload = json.loads(raw)
                label = schema.validate(str(payload.get("predicted_category", "")))
                usage = self._usage(response, started)
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

    def _usage(self, response, started: float) -> OracleUsage:  # noqa: ANN001 - tipo do SDK
        usage = response.usage
        if usage is None:
            return OracleUsage(latency_seconds=time.monotonic() - started)
        input_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        return OracleUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.monotonic() - started,
            cost_usd=self._cost(input_tokens, cached, output_tokens),
            cached_input_tokens=cached,
        )

    def _cost(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        prices = _PRICING_USD_PER_MTOK.get(self._model)
        if prices is None:
            return 0.0
        in_price, cached_price, out_price = prices
        uncached = max(input_tokens - cached_tokens, 0)
        return (
            uncached * in_price + cached_tokens * cached_price + output_tokens * out_price
        ) / 1e6
