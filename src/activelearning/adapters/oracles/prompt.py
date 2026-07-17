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


# ---------------------------------------------------------------------------
# Variantes de prompt (E0-P, ablação de instrumento). Derivadas da anatomia de
# erros do E0 (categorias-irmãs e classes guarda-chuva). Exemplos do v4b são
# descrições INVENTADAS análogas — nunca instâncias das amostras de avaliação
# (decisão D-004 no tesedaru).
# ---------------------------------------------------------------------------

RULES_V4A = (
    "Regras de fronteira DESTE catálogo (siga-as estritamente): "
    "(1) medicamentos e fármacos em geral → 'outro farma'; "
    "(2) produto bucal líquido (enxaguante/antisséptico) → 'antisseptico bucal'; "
    "(3) pó para preparo de suco/refresco (ex.: pós sabor fruta) → 'preparo para suco'; "
    "(4) néctares e bebidas de fruta prontas → 'suco pronto'; "
    "(5) limpador multiuso/perfumado sem superfície específica → 'multiuso'; "
    "só use 'produto de limpeza de piso' se o texto citar piso/chão; "
    "(6) sabão/detergente EM PÓ ou líquido PARA ROUPAS → 'lava roupas' "
    "(não 'detergente', que é para louças); "
    "(7) fones/headphones COM fio → 'acessorio de audio'; SEM fio → 'fone de ouvido'; "
    "(8) álcool e álcool gel (inclusive antisséptico de mãos) → 'alcool'; "
    "(9) leguminosas: use a variante 'em conserva' SOMENTE se o texto indicar "
    "conserva/lata; a granel ou seca use a classe simples (ex.: 'ervilha seca'); "
    "(10) pães industrializados embalados de marca → "
    "'padaria e confeitaria industrializado'. "
)

FEWSHOT_V4B = (
    "Exemplos adicionais de fronteiras difíceis (descrição → categoria): "
    "'FONE C FIO BASICO PRETO P2' → acessorio de audio; "
    "'HEADPHONE BLUETOOTH XPTO' → fone de ouvido; "
    "'ENXAG BUCAL MENTA 500ML' → antisseptico bucal; "
    "'PO P/ SUCO SABOR UVA 25G' → preparo para suco; "
    "'NECTAR DE PESSEGO 1L TP' → suco pronto; "
    "'LIMPADOR PERFUMADO LAVANDA 500ML' → multiuso; "
    "'DET PO ROUPA BRILHANTE 800G' → lava roupas; "
    "'ANALGESICO COMP C/10' → outro farma; "
    "'ALCOOL GEL HIGIENIZADOR 70 500ML' → alcool; "
    "'PAO DE FORMA INTEGRAL MARCA X 500G' → padaria e confeitaria industrializado. "
)

PROMPT_VARIANTS = {
    "v3": "",
    "v4a": RULES_V4A,
    "v4b": RULES_V4A + FEWSHOT_V4B,
}


def variant_addition(variant: str) -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"variante de prompt desconhecida: {variant!r}")
    return PROMPT_VARIANTS[variant]
