"""Adapter OraclePort genérico para APIs compatíveis com OpenAI Chat Completions.

Cobre OpenAI, Huawei ModelArts MaaS, OpenRouter e outros endpoints compatíveis.

Modos de instrumento (``mode``):
- ``enum``        : saída estruturada com rótulo restrito por enum na decodificação.
                    Requer suporte a structured outputs no provedor. Modo de produção.
- ``json-prompt`` : para provedores SEM structured output (e.g. GLM-5.2 no MaaS,
                    vários modelos :free do OpenRouter). Lista de categorias e formato
                    JSON instruídos no system prompt; parse tolerante; validação
                    pós-hoc com contabilização explícita de ``invalid_label``
                    (fallback previsto no Princípio III da constituição).
- ``free``        : saída estruturada SEM enum — réplica do instrumento do legado,
                    usada EXCLUSIVAMENTE no sub-experimento RQ4 do E0.

ATENÇÃO (validade de medição): comparar modelos medidos em modos diferentes exige
cautela — o modo é gravado em ``oracle_id`` e ``prompt_version`` de cada anotação.

Prompt caching (OpenAI): prefixo estático idêntico entre chamadas +
``prompt_cache_key``; tokens cacheados lidos de
``usage.prompt_tokens_details.cached_tokens``. Provedores sem cache ⇒ cached=0.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
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

PROMPT_VERSION_JSON_PROMPT = "v2-json-prompt"

_MODE_SUFFIX = {"enum": "", "free": "#free", "json-prompt": "#prompt"}
_MODE_PROMPT_VERSION = {
    "enum": PROMPT_VERSION,
    "free": PROMPT_VERSION_FREE,
    "json-prompt": PROMPT_VERSION_JSON_PROMPT,
}


def system_prompt_json_instruction(schema: CategorySchema) -> str:
    """System prompt do modo json-prompt: lista + formato JSON instruídos no texto."""
    return (
        system_prompt_free(schema)
        + "\n\nFormato da resposta: responda APENAS um objeto JSON válido, sem "
        'markdown, no formato {"predicted_category": "<categoria exata da lista>", '
        '"rationale": "<justificativa breve>"}.'
    )


def extract_json(text: str) -> dict:
    """Parse tolerante para o modo json-prompt.

    Remove blocos de reasoning (<think>...</think>), cercas de markdown e captura
    o primeiro objeto JSON presente no texto.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def user_prompt_batch(instances: list[Instance]) -> str:
    """Prompt de usuário para classificação em lote (itens numerados a partir de 1)."""
    lines = [f'{i + 1}. "{inst.text}"' for i, inst in enumerate(instances)]
    return "Classifique cada um dos produtos abaixo:\n" + "\n".join(lines)


def batch_json_schema(schema: CategorySchema, constrained: bool) -> dict:
    """Schema de saída para lote: array de {index, predicted_category, rationale}.

    Sem minItems/maxItems: o schema fica idêntico entre chamadas (prefixo cacheável
    mesmo com lote parcial no fim); a contagem é validada no código.
    """
    single = schema.to_json_schema(constrained)["schema"]
    category_property = single["properties"]["predicted_category"]
    return {
        "name": "oracle_classification_batch",
        "schema": {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "Número do item na lista (a partir de 1).",
                            },
                            "predicted_category": category_property,
                            "rationale": {"type": "string"},
                        },
                        "required": ["index", "predicted_category"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["classifications"],
            "additionalProperties": False,
        },
    }


def parse_batch_payload(payload: dict, n_items: int) -> dict[int, dict]:
    """Mapeia a resposta em lote por índice 1..n_items. Índices fora do range são
    ignorados; itens ausentes simplesmente não aparecem no dicionário."""
    result: dict[int, dict] = {}
    for entry in payload.get("classifications", []):
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= index <= n_items and index not in result:
            result[index] = entry
    return result


def split_usage(usage: OracleUsage, n: int) -> list[OracleUsage]:
    """Divide o usage de uma chamada em lote entre os n itens (totais preservados:
    o primeiro item recebe o resto da divisão inteira)."""
    if n <= 1:
        return [usage]
    base_in, base_out, base_cached = (
        usage.input_tokens // n,
        usage.output_tokens // n,
        usage.cached_input_tokens // n,
    )
    parts = []
    for i in range(n):
        first = i == 0
        parts.append(
            OracleUsage(
                input_tokens=usage.input_tokens - base_in * (n - 1) if first else base_in,
                output_tokens=usage.output_tokens - base_out * (n - 1) if first else base_out,
                latency_seconds=usage.latency_seconds / n,
                cost_usd=usage.cost_usd / n,
                cached_input_tokens=(
                    usage.cached_input_tokens - base_cached * (n - 1) if first else base_cached
                ),
            )
        )
    return parts


class OpenAICompatibleOracle:
    """OraclePort sobre qualquer endpoint Chat Completions compatível com OpenAI."""

    def __init__(
        self,
        model: str,
        provider_name: str = "openai",
        base_url: str | None = None,
        temperature: float = 0.0,
        api_key_env: str = "OPENAI_API_KEY",
        mode: str = "enum",
        pricing_usd_per_mtok: tuple[float, float, float] | None = None,
        use_prompt_cache_key: bool = True,
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
        max_retries: int = 3,
        requests_per_minute: float | None = None,
        rate_limit_retries: int = 5,
        items_per_call: int = 1,
    ) -> None:
        from openai import OpenAI  # import tardio: adapter, não domínio

        if mode not in _MODE_SUFFIX:
            raise ValueError(f"mode deve ser um de {sorted(_MODE_SUFFIX)}, não {mode!r}.")
        if items_per_call < 1:
            raise ValueError("items_per_call deve ser >= 1.")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Variável de ambiente {api_key_env} não definida.")
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, default_headers=extra_headers or None
        )
        self._model = model
        self._temperature = temperature
        self._mode = mode
        self._pricing = pricing_usd_per_mtok
        self._use_cache_key = use_prompt_cache_key
        self._extra_body = extra_body or {}
        self._max_retries = max_retries
        # Throttle proativo (tiers gratuitos: MaaS 3 rpm, OpenRouter :free ~20 rpm)
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._rate_limit_retries = rate_limit_retries
        self._last_request_at = 0.0
        self._items_per_call = items_per_call
        # Lote por chamada muda o instrumento de medição -> registrado no id.
        batch_suffix = f"@b{items_per_call}" if items_per_call > 1 else ""
        self.oracle_id = (
            f"{provider_name}:{model}@T{temperature}{_MODE_SUFFIX[mode]}{batch_suffix}"
        )
        self.prompt_version = _MODE_PROMPT_VERSION[mode]

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _build_request_parts(self, schema: CategorySchema) -> tuple[str, dict | None]:
        batched = self._items_per_call > 1
        if self._mode == "enum":
            system = SYSTEM_PROMPT
            response_format = {
                "type": "json_schema",
                "json_schema": (
                    batch_json_schema(schema, constrained=True)
                    if batched
                    else schema.to_json_schema(constrained=True)
                ),
            }
        elif self._mode == "free":
            system = system_prompt_free(schema)
            response_format = {
                "type": "json_schema",
                "json_schema": (
                    batch_json_schema(schema, constrained=False)
                    if batched
                    else schema.to_json_schema(constrained=False)
                ),
            }
        else:  # json-prompt
            system = system_prompt_json_instruction(schema)
            if batched:
                system += (
                    "\n\nVocê receberá uma lista numerada de produtos. Responda um "
                    'objeto JSON no formato {"classifications": [{"index": <número do '
                    'item>, "predicted_category": "<categoria exata da lista>", '
                    '"rationale": "<justificativa breve>"}, ...]} contendo exatamente '
                    "um elemento por item da lista."
                )
            response_format = None
        return system, response_format

    def annotate(self, batch: list[Instance], schema: CategorySchema) -> list[Annotation]:
        system, response_format = self._build_request_parts(schema)
        cache_key = self._cache_key(system, response_format)
        if self._items_per_call <= 1:
            return [
                self._annotate_one(i, schema, system, response_format, cache_key)
                for i in batch
            ]
        annotations: list[Annotation] = []
        for start in range(0, len(batch), self._items_per_call):
            group = batch[start : start + self._items_per_call]
            annotations.extend(
                self._annotate_group(group, schema, system, response_format, cache_key)
            )
        return annotations

    def _annotate_group(
        self,
        group: list[Instance],
        schema: CategorySchema,
        system: str,
        response_format: dict | None,
        cache_key: str,
    ) -> list[Annotation]:
        """Uma chamada de API para um grupo de instâncias; mapeia a resposta por índice."""
        started = time.monotonic()
        last_error = ""
        request_kwargs: dict = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt_batch(group)},
            ],
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        if self._use_cache_key:
            request_kwargs["prompt_cache_key"] = cache_key
        if self._extra_body:
            request_kwargs["extra_body"] = self._extra_body

        rate_limit_hits = 0
        attempt = 0
        while attempt < self._max_retries:
            attempt += 1
            try:
                self._throttle()
                response = self._client.chat.completions.create(**request_kwargs)
                raw = response.choices[0].message.content or ""
                by_index = parse_batch_payload(extract_json(raw), len(group))
                usages = split_usage(self._usage(response, started), len(group))
                annotations = []
                for position, (instance, usage) in enumerate(zip(group, usages), start=1):
                    entry = by_index.get(position)
                    if entry is None:
                        annotations.append(
                            Annotation(
                                instance_id=instance.id,
                                label=None,
                                oracle_id=self.oracle_id,
                                prompt_version=self.prompt_version,
                                raw_response=f"MISSING_IN_BATCH_RESPONSE: {raw[:500]}",
                                usage=usage,
                            )
                        )
                        continue
                    label = schema.validate(str(entry.get("predicted_category", "")))
                    annotations.append(
                        Annotation(
                            instance_id=instance.id,
                            label=label,
                            oracle_id=self.oracle_id,
                            prompt_version=self.prompt_version,
                            raw_response=json.dumps(entry, ensure_ascii=False),
                            rationale=str(entry.get("rationale", "")),
                            usage=usage,
                        )
                    )
                return annotations
            except Exception as exc:  # noqa: BLE001 - fronteira de rede
                last_error = f"{type(exc).__name__}: {exc}"
                if "prompt_cache_key" in last_error and "prompt_cache_key" in request_kwargs:
                    request_kwargs.pop("prompt_cache_key")
                    attempt -= 1
                    continue
                is_rate_limit = "RateLimitError" in last_error or " 429 " in last_error
                if is_rate_limit and rate_limit_hits < self._rate_limit_retries:
                    rate_limit_hits += 1
                    attempt -= 1
                    time.sleep(min(20.0 * rate_limit_hits, 65.0))
                    continue
                if attempt < self._max_retries:
                    time.sleep(2**attempt)
        error_usage = OracleUsage(latency_seconds=(time.monotonic() - started) / len(group))
        return [
            Annotation(
                instance_id=instance.id,
                label=None,
                oracle_id=self.oracle_id,
                prompt_version=self.prompt_version,
                raw_response=f"ERROR after {self._max_retries} attempts: {last_error}",
                usage=error_usage,
            )
            for instance in group
        ]

    def _cache_key(self, system: str, response_format: dict | None) -> str:
        digest = hashlib.sha256(
            (
                system + self.prompt_version + json.dumps(response_format, sort_keys=True)
            ).encode()
        ).hexdigest()[:16]
        return f"falco-oracle-{digest}"

    def _annotate_one(
        self,
        instance: Instance,
        schema: CategorySchema,
        system: str,
        response_format: dict | None,
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
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        if self._use_cache_key:
            request_kwargs["prompt_cache_key"] = cache_key
        if self._extra_body:
            request_kwargs["extra_body"] = self._extra_body

        rate_limit_hits = 0
        attempt = 0
        while attempt < self._max_retries:
            attempt += 1
            try:
                self._throttle()
                response = self._client.chat.completions.create(**request_kwargs)
                raw = response.choices[0].message.content or ""
                payload = extract_json(raw)
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
                    attempt -= 1
                    continue
                # 429: espera longa própria (janela de rate limit), não conta
                # como tentativa comum e tem contador separado.
                is_rate_limit = "RateLimitError" in last_error or " 429 " in last_error
                if is_rate_limit and rate_limit_hits < self._rate_limit_retries:
                    rate_limit_hits += 1
                    attempt -= 1
                    time.sleep(min(20.0 * rate_limit_hits, 65.0))
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
    """Oráculo via Huawei ModelArts Studio (MaaS), endpoint OpenAI-compatible.

    Base URL por região (nota: caminho ``/v2``), e.g.
    ``https://api-ap-southeast-1.modelarts-maas.com/v2``.
    Nomes de modelo conforme GET /v2/models (e.g. ``deepseek-v4-flash``,
    ``deepseek-v4-pro``, ``glm-5.2`` — minúsculas).

    GLM-5.2 NÃO suporta structured output (console MaaS) ⇒ usar ``mode="json-prompt"``.
    Thinking é desativado por padrão (sintaxe MaaS: ``thinking: {type: disabled}``)
    para latência/custo e para evitar interação ruim com parsing de JSON.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        api_key_env: str = "MAAS_API_KEY",
        mode: str = "json-prompt",
        pricing_usd_per_mtok: tuple[float, float, float] | None = None,
        disable_thinking: bool = True,
        requests_per_minute: float | None = 3.0,
        items_per_call: int = 1,
        max_retries: int = 3,
    ) -> None:
        base_url = base_url or os.environ.get(
            "MAAS_BASE_URL", "https://api-ap-southeast-1.modelarts-maas.com/v2"
        )
        extra_body: dict = {}
        if disable_thinking:
            extra_body["thinking"] = {"type": "disabled"}
        super().__init__(
            model=model,
            provider_name="huawei-maas",
            base_url=base_url,
            temperature=temperature,
            api_key_env=api_key_env,
            mode=mode,
            pricing_usd_per_mtok=pricing_usd_per_mtok,
            use_prompt_cache_key=False,  # parâmetro específico da OpenAI
            extra_body=extra_body,
            requests_per_minute=requests_per_minute,
            items_per_call=items_per_call,
            max_retries=max_retries,
        )


class OpenRouterOracle(OpenAICompatibleOracle):
    """Oráculo via OpenRouter (agregador OpenAI-compatible; inclui modelos :free).

    Base URL ``https://openrouter.ai/api/v1``; nomes conforme GET /api/v1/models
    (e.g. ``nvidia/nemotron-3-ultra-550b-a55b:free``).
    Reasoning é desativado por padrão (``reasoning: {enabled: false}``) — para
    classificação de texto curto o custo/latência do reasoning não se justifica e
    modelos :free têm cotas apertadas. Muitos modelos :free não suportam structured
    output ⇒ default ``mode="json-prompt"``.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        api_key_env: str = "OPENROUTER_API_KEY",
        mode: str = "json-prompt",
        pricing_usd_per_mtok: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
        reasoning_enabled: bool = False,
        requests_per_minute: float | None = 18.0,
        items_per_call: int = 1,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model=model,
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            api_key_env=api_key_env,
            mode=mode,
            pricing_usd_per_mtok=pricing_usd_per_mtok,
            use_prompt_cache_key=False,
            extra_body={"reasoning": {"enabled": reasoning_enabled}},
            extra_headers={
                "HTTP-Referer": "https://github.com/GHDaru/activelearning",
                "X-Title": "activelearning-falco",
            },
            requests_per_minute=requests_per_minute,
            items_per_call=items_per_call,
            max_retries=max_retries,
        )
