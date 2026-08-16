# Spec 004 — Site da tese: navegar, reproduzir e executar

## Objetivo
Dar ao visitante de fora do projeto uma porta única para a tese FALCO: navegar
pelos experimentos, entender como reproduzir cada número e **executar** um
experimento de aprendizado ativo sem instalar nada. Hoje esse material existe
(235 artefatos em `experiments/`, o `REPRODUCIBILITY.md`, o catálogo de
experimentos da API e o front do FlowBuilder), mas só é alcançável por quem
clona o repositório e sobe o backend na própria máquina — o que está publicado
no Vercel é apenas a função read-only do grafo.

O princípio I da constituição diz que **reprodutibilidade é o produto**. Um
artefato que só o autor consegue rodar cumpre a letra e não o espírito: esta
spec fecha a distância entre "o número tem artefato rastreável" e "qualquer
pessoa alcança o artefato e o refaz".

## Decisões que originam a spec
Tomadas pelo autor em 16/08/2026:

1. **Espinha no `activelearning`**, estendendo `apps/web` — e não um motor novo
   no `tesedaru`. O software e os dados já vivem aqui; o texto da tese entra
   como conteúdo, não como fundação.
2. **Backend real hospedado** — execução de verdade, não simulação no
   navegador. Exige host de contêiner (o serverless do Vercel não comporta
   processo longo nem disco).
3. **Vercel com subdomínio próprio** — mesmo caminho já provado no livro
   `harness_engineering`: um endereço canônico só.

## Arquitetura
Duas origens, um site:

```
visitante → https://<subdominio>            (Vercel: front estático, apps/web/dist)
                    │
                    └── VITE_API_BASE ──→ https://<backend>   (contêiner: FastAPI completo)
                                              /api/experiments/{id}/execute
                                              /api/runs            (aprendizado ativo)
                                              /api/datasets, /api/kg/*
```

O front continua funcionando **sem** `VITE_API_BASE`: nesse caso usa `/api`
relativo — o proxy do Vite em desenvolvimento e a função read-only do grafo no
Vercel. Definir a variável é o que liga o botão de executar.

Por que não um só lugar: `vercel.json` roteia `/api/*` para `api/index.py`, uma
função serverless com teto de tempo e sem disco persistente. Um run de
aprendizado ativo é justamente processo longo com artefatos em disco (Princípio
I). Separar as origens é o que torna a execução possível sem abrir mão do
artefato rastreável — e é o custo de CORS que o backend já previa
(`FLOWBUILDER_CORS`).

## Escopo desta spec (R1 — fundação)
1. `apps/web` resolve a base da API por `VITE_API_BASE` (vazio = origem atual).
2. O backend completo empacotado para contêiner (`Dockerfile`), com blueprint
   de host versionado (`render.yaml`) — o deploy deixa de ser receita de prosa.
3. `DEPLOY_AND_PUBLISH.md` com os passos **[VOCÊ]** do backend e do subdomínio.

## Fora do escopo (R2 em diante)
- Navegação do texto da tese dentro do site (LaTeX → conteúdo web).
- Página por experimento com a receita de reprodução ao lado do resultado.
- Limite de custo/abuso por visitante nas execuções com oráculo LLM real.

Ficam fora porque dependem desta fundação: sem backend alcançável e sem
endereço, as telas seguintes não teriam o que mostrar.

## Critérios de aceite (verificáveis)
1. `npm run build` em `apps/web` passa com e sem `VITE_API_BASE` definida.
2. Com `VITE_API_BASE` vazia, todo caminho de rede do front continua relativo
   (`/api/...`) — nenhuma URL absoluta hardcoded no bundle.
3. Com `VITE_API_BASE=https://exemplo`, as chamadas do front — inclusive as que
   viram atributo `href`/`src` (download de dataset, visualização do grafo) —
   apontam para essa origem.
4. `docker build` produz imagem que sobe a API e responde `/api/health` com
   `{"status":"ok"}`.
5. `render.yaml` declara o serviço com o comando de start, a versão de Python e
   as variáveis de ambiente **sem nenhum segredo versionado**.
6. A suíte existente (`python -m pytest tests/ -q`) segue verde.

## Riscos assumidos
- **Custo e abuso**: com oráculo LLM real, cada visitante gasta chave do autor.
  Esta spec entrega a fundação com o oráculo **simulado** como padrão do
  catálogo público; o teto por visitante é trabalho de R2 e o backend não deve
  ser anunciado antes disso.
- **Host adormecido**: plano gratuito de contêiner hiberna e a primeira chamada
  demora. O front precisa tratar a espera como estado normal (R2), não erro.
