"""Fábrica de oráculos a partir de uma spec JSON (mesma dos configs do E0)."""
from __future__ import annotations


def build_oracle(spec: dict):
    """Constrói um oráculo (``OraclePort``) a partir de uma spec JSON.

    A spec é a mesma dos configs do E0. Campo obrigatório ``provider`` — um de
    ``simulated``, ``openai``, ``huawei-maas``, ``openrouter``, ``nvidia``,
    ``openai-compatible``, ``gemini``, ``ollama``. Demais chaves (``model``,
    ``mode``, ``temperature``, ``pricing_usd_per_mtok``, ``items_per_call``, …)
    dependem do provider. Levanta ``ValueError`` para provider desconhecido.
    """
    kind = spec["provider"]
    mode = spec.get("mode", "enum")
    temperature = float(spec.get("temperature", 0.0))
    pricing = spec.get("pricing_usd_per_mtok")
    pricing = tuple(pricing) if pricing else None

    if kind == "simulated":
        from activelearning.adapters.oracles.simulated_oracle import SimulatedOracle

        return SimulatedOracle(
            noise=float(spec.get("noise", 0.0)), seed=int(spec.get("seed", 42))
        )
    if kind == "openai":
        from activelearning.adapters.oracles.openai_oracle import OpenAIOracle

        return OpenAIOracle(
            model=spec["model"], temperature=temperature, mode=mode,
            items_per_call=int(spec.get("items_per_call", 1)),
        )
    if kind == "huawei-maas":
        from activelearning.adapters.oracles.openai_compatible import HuaweiMaasOracle

        return HuaweiMaasOracle(
            model=spec["model"],
            base_url=spec.get("base_url"),
            temperature=temperature,
            mode=mode,
            pricing_usd_per_mtok=pricing,
            disable_thinking=bool(spec.get("disable_thinking", True)),
            requests_per_minute=spec.get("requests_per_minute", 3.0),
            items_per_call=int(spec.get("items_per_call", 1)),
        )
    if kind == "openrouter":
        from activelearning.adapters.oracles.openai_compatible import OpenRouterOracle

        return OpenRouterOracle(
            model=spec["model"],
            temperature=temperature,
            mode=mode,
            pricing_usd_per_mtok=pricing or (0.0, 0.0, 0.0),
            reasoning_enabled=bool(spec.get("reasoning_enabled", False)),
            requests_per_minute=spec.get("requests_per_minute", 18.0),
            items_per_call=int(spec.get("items_per_call", 1)),
        )
    if kind == "nvidia":
        from activelearning.adapters.oracles.openai_compatible import NvidiaNimOracle

        return NvidiaNimOracle(
            model=spec["model"],
            temperature=temperature,
            mode=mode,
            pricing_usd_per_mtok=pricing or (0.0, 0.0, 0.0),
            enable_thinking=bool(spec.get("enable_thinking", False)),
            requests_per_minute=spec.get("requests_per_minute", 30.0),
            items_per_call=int(spec.get("items_per_call", 1)),
        )
    if kind == "openai-compatible":
        from activelearning.adapters.oracles.openai_compatible import OpenAICompatibleOracle

        return OpenAICompatibleOracle(
            model=spec["model"],
            provider_name=spec.get("name", "openai-compatible"),
            base_url=spec["base_url"],
            temperature=temperature,
            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
            mode=mode,
            pricing_usd_per_mtok=pricing,
            use_prompt_cache_key=False,
        )
    if kind == "gemini":
        from activelearning.adapters.oracles.gemini_oracle import GeminiOracle

        return GeminiOracle(model=spec["model"], temperature=temperature)
    if kind == "ollama":
        from activelearning.adapters.oracles.ollama_oracle import OllamaOracle

        return OllamaOracle(model=spec["model"], temperature=temperature)
    raise ValueError(f"Provider desconhecido: {kind}")
