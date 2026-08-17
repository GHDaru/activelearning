# Re-coleta do ciclo E5 — 2026-08-17

O `annotation_cache_nemotron.jsonl` original (julho/2026, 9.357 registros) foi
perdido — nunca versionado (excluído pela regra `experiments/*/results/*.jsonl`
do `.gitignore`) e sem cópia recuperável em Drive ou máquinas locais. Esta
pasta documenta a **re-coleta** ordenada pelo autor, executada pela sessão
`executor02` com o mesmo oráculo (`nvidia/nemotron-3-ultra-550b-a55b`, T=0,
prompt v3, lotes de 50), mesmo pool (semente de dados 42) e mesmo comando —
agora **com `--cache`**, que o comando da rodada original não registrou por
escrito e cuja omissão custou uma rodada descartada (ver mensagem de conclusão
na coordenação da tese).

- `cycle_sgd_b15k.json` / `cycle_sgd_b15k_records.jsonl` — sumário e curva do
  ciclo SGD da re-coleta. **Não substituem** os arquivos homônimos da pasta de
  cima (julho), que são os citados pela tese; a re-coleta existe para
  reconstituir o CACHE (braços A/B/C do E3′), não para refazer as curvas.
- O cache re-coletado NÃO é versionado (decisão do principal: é dado, não
  código) — vive como dataset privado do Kaggle `ghdaru/falco-annotation-cache`
  e alimenta os kernels do E3′.
- Trajetórias diferem do original (T=0 não é determinismo perfeito no
  provedor; o laço realimenta a seleção): SGD parou em 3.699 rotulados
  (julho: 4.742), com 51 inválidos (julho: 208). A tese deve tratar A/B/C
  como re-coleta, não como reprodução bit a bit.
