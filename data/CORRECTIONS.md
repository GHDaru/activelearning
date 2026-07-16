# Correções e achados de auditoria — dataset.csv

Auditoria realizada em 16/07/2026 sobre o *Retail Product Description-Ptbr*
(origem: repositório legado `activetextclassification`, mesmo arquivo).

## Arquivos

| Arquivo | Conteúdo | Linhas (sem cabeçalho) |
|---|---|---|
| `data_old/dataset_original.csv` | **Original intocado**, como veio do legado | 250.365 |
| `dataset.csv` | **Versão corrigida** usada pelos experimentos | 250.221 |
| `data_old/removed_inativo.csv` | Exatamente as linhas removidas (diff antigo→novo) | 144 |

A ordem relativa das linhas mantidas é preservada — os identificadores de
instância (`e0-{índice}`) gerados pelos runners permanecem estáveis em relação às
execuções anteriores (que aplicavam o mesmo filtro em tempo de carga).

## Correção aplicada (única edição na base)

**Remoção do rótulo operacional `inativo` (144 linhas).** O valor registra um
status de cadastro do varejista, não uma categoria de produto (ex.:
`UVA PASSA ESCURA 150G CRFO` → `inativo`). Mantê-lo poluiria o espaço de rótulos
(enum do oráculo) com uma classe sem semântica de produto. Todas as linhas
removidas estão em `data_old/removed_inativo.csv`.

Antes desta materialização, a mesma exclusão era feita em tempo de carga pelo
runner (`exclude_labels` em `experiments/e0/run_e0.py`, default `["inativo"]`).
O default foi mantido como salvaguarda idempotente — sobre a base corrigida ele
não remove nada.

## Achados NÃO corrigidos (decisão deliberada — tratados no desenho experimental)

1. **Conflitos de rótulo-ouro** — 719 descrições distintas (1.807 linhas; 0,7%)
   aparecem com 2+ categorias diferentes (ex.: `MILHARINA QUAKER 500G` ocorre com
   *farinha de milho*, *floco de milho*, *fubá* e *polenta*). **Mantidos**: são
   ruído real do domínio; impõem teto de ~99,3% à acurácia mensurável, reportado
   na tese (Metodologia, seção de auditoria). Corrigi-los exigiria re-anotação
   humana e mudaria a natureza da base.
2. **Duplicatas exatas (texto+rótulo)** — 19.356 linhas (7,7%). **Mantidas**:
   refletem frequência real de produtos; inócuas para avaliação de oráculo.
   Consequência obrigatória registrada: nos experimentos com treinamento (E2/E3),
   o particionamento deduplica por descrição normalizada ANTES do split
   (prevenção de vazamento treino→teste).
3. **Rótulos "outro …"** — 32 classes agregadoras (~9,9 mil linhas). **Mantidas**:
   categorias legítimas do catálogo, ainda que difíceis por definição.

## Rastreabilidade

- Commit da correção: ver histórico do git deste repositório.
- Números citados na tese: `tesedaru/3-metodo/texto.tex`, seção
  "Auditoria de qualidade dos rótulos".
