"""Adapter OraclePort para a API OpenAI (caso particular do OpenAI-compatible).

Mantém a tabela de preços dos modelos OpenAI e o prompt caching automático
(prefixo estático + ``prompt_cache_key``; tokens cacheados a 50% do preço de
input na família gpt-4o, sem custo de escrita).
"""
from __future__ import annotations

from .openai_compatible import OpenAICompatibleOracle

# Preços por 1M tokens: (input, cached_input, output) — atualizar conforme tabela vigente.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
}


class OpenAIOracle(OpenAICompatibleOracle):
    """Oráculo LLM via API da OpenAI, com saída estruturada restrita ao esquema.

    Desenhado para *prompt caching*: o prefixo estático (system + enum de
    categorias) é idêntico entre chamadas e reportado em ``OracleUsage``
    (``cached_input_tokens``, ``cache_hit_rate``). Requer ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key_env: str = "OPENAI_API_KEY",
        mode: str = "enum",
        items_per_call: int = 1,
        max_retries: int = 3,
        prompt_variant: str = "v3",
    ) -> None:
        super().__init__(
            model=model,
            provider_name="openai",
            base_url=None,
            temperature=temperature,
            api_key_env=api_key_env,
            mode=mode,
            prompt_variant=prompt_variant,
            pricing_usd_per_mtok=_PRICING_USD_PER_MTOK.get(model),
            use_prompt_cache_key=True,
            items_per_call=items_per_call,
            max_retries=max_retries,
        )
