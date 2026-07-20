# falco-active-learning

**Aprendizado Ativo com oráculos LLM para classificação de texto curto** — o motor
experimental do framework **FALCO** (Framework de Aprendizado Ativo com LLM para
texto CurtO), da tese de doutorado de Gilsiley Henrique Darú (PPGMNE/UFPR).

A biblioteca ataca o custo dominante da classificação de texto curto — **a
rotulagem** — usando LLMs justamente nas fases em que o aprendizado ativo (AA)
clássico é mais frágil: a **partida a frio** (via DRI-SL, construção do conjunto
inicial sem nenhum rótulo) e a **rotulagem** (via oráculos LLM com custo
instrumentado).

```bash
pip install falco-active-learning
```

## Por que existe

- **Texto curto de varejo** (descrições de cupom fiscal: `CERV BRAHMA LT 350ML`),
  4–50 caracteres, centenas de categorias de cauda longa.
- O gargalo não é o classificador — é obter rótulos. O FALCO usa o LLM para
  **selecionar** e **rotular**, com contratos de saída restritos por *enum* e
  custo medido.
- **Nenhum número sem artefato rastreável**: toda afirmação empírica remete a um
  artefato versionado e reexecutável.

## Mapa da documentação

| Página | Conteúdo |
|--------|----------|
| [Instalação](instalacao.md) | pip, extras opcionais, requisitos |
| [Início rápido](quickstart.md) | primeiros exemplos executáveis |
| [Conceitos](conceitos.md) | o modelo de domínio (Instance, Label, CategorySchema, estratégias, LCE) |
| [Guia da biblioteca](biblioteca.md) | classificadores, DRI-SL, oráculos, laço de AL, runner FALCO |
| [Arquitetura](architecture.md) | DDD + Hexagonal (domínio/ports/adapters) |
| [Referência da API](api.md) | documentação gerada dos módulos públicos |
| [Software (FlowBuilder)](flowbuilder.md) | a interface web sobre a biblioteca |
| [Experimentos](experimentos.md) | executar e parametrizar E0/E0-P/E1/E4/... |
| [Reprodutibilidade](reproducibilidade.md) | ambiente, dados, custos, publicação |

## Arquitetura em uma frase

Domínio puro (sem I/O) no centro; capacidades externas (LLMs, classificadores,
persistência) entram por **ports** e **adapters**. Isso mantém a lógica de AA
testável sem rede, GPU ou chaves de API.
