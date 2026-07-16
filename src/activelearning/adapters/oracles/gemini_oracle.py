"""Adapter OraclePort para Google Gemini (google-genai), saída restrita por enum."""
from __future__ import annotations

import json
import os
import time

from ...domain.annotation import Annotation, OracleUsage
from ...domain.instances import CategorySchema, Instance
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, user_prompt

_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash-latest": (0.075, 0.30),
    "gemini-1.5-pro-latest": (1.25, 5.00),
}


class GeminiOracle:
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        api_key_env: str = "GEMINI_API_KEY",
        max_retries: int = 3,
    ) -> None:
        from google import genai  # import tardio: adapter, não domínio

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Variável de ambiente {api_key_env} não definida.")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self.oracle_id = f"gemini:{model}@T{temperature}"
        self.prompt_version = PROMPT_VERSION

    def annotate(self, batch: list[Instance], schema: CategorySchema) -> list[Annotation]:
        response_schema = self._to_gemini_schema(schema)
        return [self._annotate_one(instance, schema, response_schema) for instance in batch]

    @staticmethod
    def _to_gemini_schema(schema: CategorySchema) -> dict:
        base = schema.to_json_schema()["schema"]
        base.pop("additionalProperties", None)  # não suportado pelo response_schema
        return base

    def _annotate_one(
        self, instance: Instance, schema: CategorySchema, response_schema: dict
    ) -> Annotation:
        started = time.monotonic()
        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_prompt(instance),
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "temperature": self._temperature,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                    },
                )
                raw = response.text or ""
                payload = json.loads(raw)
                label = schema.validate(str(payload.get("predicted_category", "")))
                meta = getattr(response, "usage_metadata", None)
                input_tokens = int(getattr(meta, "prompt_token_count", 0) or 0)
                output_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
                usage = OracleUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=time.monotonic() - started,
                    cost_usd=self._cost(input_tokens, output_tokens),
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

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        prices = _PRICING_USD_PER_MTOK.get(self._model)
        if prices is None:
            return 0.0
        return (input_tokens * prices[0] + output_tokens * prices[1]) / 1e6
