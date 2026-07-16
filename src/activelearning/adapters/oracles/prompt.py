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

PROMPT_VERSION = "v3-enum"
PROMPT_VERSION_FREE = "v3-free"

_TASK_CONTEXT = (
    "Você é um especialista em catalogação de produtos do varejo brasileiro. "
    "As descrições vêm de cupons fiscais e cadastros de e-commerce: são curtas, "
    "geralmente em caixa alta e cheias de abreviações (ex.: CERV=cerveja, "
    "REFR=refrigerante, BISC=biscoito, CHOC=chocolate, RECH=recheado, "
    "PCT=pacote, LT=lata, CX=caixa, UN=unidade). "
    "Expanda mentalmente as abreviações e classifique o TIPO do produto, "
    "ignorando marca, tamanho e embalagem. "
    "Escolha sempre a categoria mais específica aplicável; na dúvida entre uma "
    "categoria plausível e '_rare_', prefira a categoria plausível — use "
    "'_rare_' SOMENTE se nenhuma categoria da lista descrever o tipo do produto. "
    "Exemplos: 'CERV BRAHMA LT 350ML' → cerveja; "
    "'BISC RECH CHOC 130G' → biscoito; 'REQUEIJAO LIGHT CP 200G' → requeijao. "
)

SYSTEM_PROMPT = _TASK_CONTEXT + "Responda somente no formato estruturado solicitado."


def system_prompt_free(schema: CategorySchema) -> str:
    """System prompt do modo livre: lista no texto, sem enum (instrumento do legado)."""
    categories = ", ".join(schema.values)
    return (
        _TASK_CONTEXT
        + "Escolha dentre as categorias válidas listadas abaixo. Responda APENAS "
        "com uma categoria exata da lista, sem criar variações. "
        "Responda somente no formato estruturado solicitado.\n\n"
        f"Categorias válidas: {categories}"
    )


def user_prompt(instance: Instance) -> str:
    return f'Descrição do produto: "{instance.text}"'
