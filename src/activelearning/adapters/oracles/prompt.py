"""Prompt canônico do oráculo de classificação (versão v2-enum).

A lista de categorias NÃO é injetada como texto persuasivo: ela é imposta pelo
``enum`` do schema de saída estruturada (CategorySchema.to_json_schema). O texto
do prompt apenas contextualiza a tarefa.
"""
from __future__ import annotations

from ...domain.instances import Instance

PROMPT_VERSION = "v2-enum"

SYSTEM_PROMPT = (
    "Você é um especialista em catalogação de produtos de varejo brasileiro. "
    "Dada a descrição curta de um produto (frequentemente com abreviações), "
    "expanda mentalmente as abreviações e escolha a categoria mais adequada. "
    "Se nenhuma categoria se aplicar, use '_rare_'. "
    "Responda somente no formato estruturado solicitado."
)


def user_prompt(instance: Instance) -> str:
    return f'Descrição do produto: "{instance.text}"'
