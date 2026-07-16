"""Adapter OraclePort genérico para APIs compatíveis com OpenAI Chat Completions.

Cobre a API da OpenAI e provedores OpenAI-compatible (Huawei ModelArts MaaS,
DeepSeek direto, etc. — tipicamente servidos por vLLM, que suporta structured
outputs via ``response_format`` json_schema).

Modos de schema:
- ``constrained=True`` (produção): rótulo restrito por enum na decodificação.
- ``constrained=False`` (APENAS sub-experimento RQ4/E0): string livre com lista de
  categorias no system prompt — reproduz o instrumento do legado; validação
  pós-hoc contabiliza ``invalid_label``.

Prompt caching (OpenAI): prefixo estático (system + schema) idêntico entre
chamadas; ``prompt_cache_key`` estável melhora o roteamento; tokens cacheados
são lidos de ``usage.prompt_tokens_details.cached_tokens`` e descontados no custo.
Provedores que não reportam cache simplesmente resultam em cached=0.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from ...domain.annotation import Annotation, OracleUsage
from ...domain.instances import CategorySchema, Instance
from .prompt import (
    PROMPT_VERSION,
    PROMPT_VERSION_FREE,
    SYSTEM_PROMPT,
    system_prompt_free,
    user_prompt,
)


class OpenAICompatibleOracle:
    """OraclePort sobre qualquer endpoint Chat Completions compatível com OpenAI."""

    def __init__(
        self,
        model: str,
        provider_name: str = "openai",
        base_url: str | None = None,
        temperature: float = 0.0,
        api_key_env: str = "OPENAI_API_KEY",
        constrained: bool = True,
        pricing_usd_per_mtok: tuple[float, float, float] | None = None,
        use_prompt_cache_key: bool = True,
        extra_body: dict | None = None,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI  # import tardio: adapter, não domínio

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Variável de ambiente {api_key_env} não definida.")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._constrained = constrained
        self._pricing = pricing_usd_per_mtok
        self._use_cache_key = use_prompt_cache_key
        self._extra_body = extra_body or {}
        self._max_retries = max_retries
        mode_suffix = "" if constrained else "#free"
        self.oracle_id = f"{provider_name}:{model}@T{temperature}{mode_suffix}"
        self.prompt_version = PROMPT_VERSION if constrained else PROMPT_VERSION_FREE

    def annotate(self, batch: list[Instance], schema: CategorySchema) -> list[Annotation]:
        json_schema = schema.to_json_schema(constrained=self._constrained)
        system = SYSTEM_PROMPT if self._constrained else system_prompt_free(schema)
        cache_key = self._cache_key(system, json_schema)
        return [self._annotate_one(i, schema, json_schema, system, cache_key) for i in batch]

    def _cache_key(self, system: str, json_schema: dict) -> str:
        digest = hashlib.sha256(
            (system + self.prompt_version + json.dumps(json_schema, sort_keys=True)).encode()
        ).hexdigest()[:16]
        return f"falco-oracle-{digest}"

    def _annotate_one(
        self,
        instance: Instance,
        schema: CategorySchema,
        json_schema: dict,
        system: str,
        cache_key: str,
    ) -> Annotation:
        started = time.monotonic()
        last_error = ""
        request_kwargs: dict = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                # Prefixo ESTÁTICO primeiro (cacheável); parte variável ao final.
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt(instance)},
            ],
            "response_format": {"type": "json_schema", "json_schema": json_schema},
        }
        if self._use_cache_key:
            request_kwargs["prompt_cache_key"] = cache_key
        if self._extra_body:
            request_kwargs["extra_body"] = self._extra_body

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(**request_kwargs)
                raw = response.choices[0].message.content or ""
                payload = json.loads(raw)
                label = schema.validate(str(payload.get("predicted_category", "")))
                return Annotation(
                    instance_id=instance.id,
                    label=label,
                    oracle_id=self.oracle_id,
                    prompt_version=self.prompt_version,
                    raw_response=raw,
                    rationale=str(payload.get("rationale", "")),
                    usage=self._usage(response, started),
                )
            except Exception as exc:  # noqa: BLE001 - fronteira de rede
                last_error = f"{type(exc).__name__}: {exc}"
                # Alguns provedores OpenAI-compatible rejeitam prompt_cache_key
                if "prompt_cache_key" in last_error and "prompt_cache_key" in request_kwargs:
                    request_kwargs.pop("prompt_cache_key")
                    continue
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
        if self._pricing is None:
            return 0.0
        in_price, cached_price, out_price = self._pricing
        uncached = max(input_tokens - cached_tokens, 0)
        return (
            uncached * in_price + cached_tokens * cached_price + output_tokens * out_price
        ) / 1e6


class HuaweiMaasOracle(OpenAICompatibleOracle):
    """Oráculo via Huawei ModelArts Studio (MaaS) — endpoint OpenAI-compatible.

    Base URL por região, e.g. ``https://api-ap-southeast-1.modelarts-maas.com/v1``.
    Backend Ascend-vLLM: suporta structured outputs via response_format json_schema.
    Para modelos DeepSeek, o modo thinking/reasoning é DESATIVADO por padrão —
    structured output com thinking ativo tem defeito conhecido no vLLM.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        api_key_env: str = "MAAS_API_KEY",
        constrained: bool = True,
        pricing_usd_per_mtok: tuple[float, float, float] | None = None,
        disable_thinking: bool = True,
        max_retries: int = 3,
    ) -> None:
        base_url = base_url or os.environ.get(
            "MAAS_BASE_URL", "https://api-ap-southeast-1.modelarts-maas.com/v1"
        )
        extra_body: dict = {}
        if disable_thinking:
            # Convenção vLLM/DeepSeek para desligar o modo thinking na inferência.
            extra_body["chat_template_kwargs"] = {"thinking": False}
        super().__init__(
            model=model,
            provider_name="huawei-maas",
            base_url=base_url,
            temperature=temperature,
            api_key_env=api_key_env,
            constrained=constrained,
            pricing_usd_per_mtok=pricing_usd_per_mtok,
            use_prompt_cache_key=False,  # parâmetro específico da OpenAI
            extra_body=extra_body,
            max_retries=max_retries,
        )
