"""Prompts canônicos do oráculo de classificação.

Modo ``enum`` (produção): a lista de categorias NÃO é injetada como texto
persuasivo — ela é imposta pelo ``enum`` do schema de saída estruturada
(CategorySchema.to_json_schema). O texto do prompt apenas contextualiza a tarefa.

Modo ``free`` (APENAS para o sub-experimento RQ4 do E0): reproduz o instrumento
do legado — rótulo como string livre, com a lista de categorias no system prompt.
Em ambos os modos o prefixo é estático (cacheável); só a descrição varia.
"""
from __future__ import annotations

from ...domain.instances import CategorySchema, Instance

PROMPT_VERSION = "v2-enum"
PROMPT_VERSION_FREE = "v2-free"

SYSTEM_PROMPT = (
    "Você é um especialista em catalogação de produtos de varejo brasileiro. "
    "Dada a descrição curta de um produto (frequentemente com abreviações), "
    "expanda mentalmente as abreviações e escolha a categoria mais adequada. "
    "Se nenhuma categoria se aplicar, use '_rare_'. "
    "Responda somente no formato estruturado solicitado."
)


def system_prompt_free(schema: CategorySchema) -> str:
    """System prompt do modo livre: lista no texto, sem enum (instrumento do legado)."""
    categories = ", ".join(schema.values)
    return (
        "Você é um especialista em catalogação de produtos de varejo brasileiro. "
        "Dada a descrição curta de um produto (frequentemente com abreviações), "
        "expanda mentalmente as abreviações e escolha a categoria mais adequada "
        "dentre as categorias válidas listadas abaixo. Responda APENAS com uma "
        "categoria exata da lista, sem criar variações. Se nenhuma se aplicar, "
        "use '_rare_'. Responda somente no formato estruturado solicitado.\n\n"
        f"Categorias válidas: {categories}"
    )


def user_prompt(instance: Instance) -> str:
    return f'Descrição do produto: "{instance.text}"'
