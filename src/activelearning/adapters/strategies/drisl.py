"""DRI-SL — Diversidade Representativa Inicial via densidade Semântica e variedade Lexical.

Constrói um L0 de tamanho alvo I a partir do pool SEM acesso a rótulos
(Pilar P2 da tese; estratégia da Fase 1 do FALCO):

1. densidade semântica: instâncias são projetadas por um encoder de sentenças e
   agrupadas por k-médias em N_c grupos; a alocação de amostras é proporcional
   ao tamanho de cada grupo (grupos maiores contribuem mais);
2. variedade lexical intragrupo: dentro de cada grupo, amostras são escolhidas
   iterativamente maximizando um escore de novidade — a soma dos pesos TF-IDF
   (perfil do grupo) dos termos que a instância introduz e que ainda não estão
   cobertos pelas já selecionadas.

O encoder é uma PORTA (callable texts -> matriz [n, d]): o padrão da tese é
SBERT multilíngue (SbertEncoder); TfidfSvdEncoder é a alternativa leve para
testes e ambientes sem GPU/download de modelo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Encoder = Callable[[Sequence[str]], np.ndarray]


class TfidfSvdEncoder:
    """Encoder leve: TF-IDF + SVD truncada (LSA). Determinístico por semente."""

    def __init__(self, n_components: int = 128, seed: int = 42) -> None:
        self._n_components = n_components
        self._seed = seed

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        matrix = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", lowercase=True, strip_accents="ascii"
        ).fit_transform(texts)
        k = min(self._n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
        if k < 2:
            return np.asarray(matrix.todense(), dtype=np.float32)
        svd = TruncatedSVD(n_components=k, random_state=self._seed)
        return svd.fit_transform(matrix).astype(np.float32)


class SbertEncoder:
    """Encoder da tese: SBERT multilíngue (Reimers & Gurevych, 2019)."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self._model_name = model_name
        self._model = None

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return np.asarray(
            self._model.encode(list(texts), show_progress_bar=False, batch_size=256)
        )


@dataclass
class DrislResult:
    indices: list[int]
    cluster_of: np.ndarray
    n_clusters: int
    allocation: list[int]


def _allocate(group_of: "np.ndarray", n_groups: int, target_size: int) -> "np.ndarray":
    """Alocação proporcional ao tamanho do grupo (mínimo 1 por grupo não vazio)."""
    sizes = np.bincount(group_of, minlength=n_groups)
    raw = sizes / sizes.sum() * target_size
    allocation = np.maximum(np.floor(raw).astype(int), (sizes > 0).astype(int))
    while allocation.sum() > target_size:  # apara excesso dos maiores
        allocation[int(np.argmax(allocation))] -= 1
    while allocation.sum() < target_size:  # distribui resto pelas maiores frações
        frac = raw - allocation
        frac[sizes == 0] = -1
        allocation[int(np.argmax(frac))] += 1
    return allocation


def _novelty_select(texts: Sequence[str], group_of: "np.ndarray",
                    allocation: "np.ndarray") -> list[int]:
    """Dentro de cada grupo, seleção gulosa por novidade lexical (perfil TF-IDF)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", lowercase=True, strip_accents="ascii"
    )
    tfidf = vectorizer.fit_transform(texts)
    selected: list[int] = []
    for c in range(len(allocation)):
        members = np.flatnonzero(group_of == c)
        quota = int(allocation[c])
        if quota <= 0 or len(members) == 0:
            continue
        profile = np.asarray(tfidf[members].mean(axis=0)).ravel()
        covered: set[int] = set()
        chosen: list[int] = []
        candidate_terms = {i: set(tfidf[i].indices) for i in members}
        for _ in range(min(quota, len(members))):
            best, best_score = -1, -1.0
            for i in members:
                if i in chosen:
                    continue
                new_terms = candidate_terms[i] - covered
                score = float(profile[list(new_terms)].sum()) if new_terms else 0.0
                if score > best_score:
                    best, best_score = int(i), score
            chosen.append(best)
            covered |= candidate_terms[best]
        selected.extend(chosen)
    return selected


def drisl_select(
    texts: Sequence[str],
    target_size: int,
    encoder: Encoder,
    n_clusters: int | None = None,
    seed: int = 42,
) -> DrislResult:
    """Seleciona ``target_size`` índices do pool via densidade semântica + novidade lexical.

    Forma clássica (cold start): grupos vêm de k-means sobre embeddings.
    """
    from sklearn.cluster import KMeans

    n = len(texts)
    if not 0 < target_size <= n:
        raise ValueError("target_size deve estar em (0, len(texts)].")
    n_clusters = n_clusters or max(2, min(int(math.sqrt(target_size)), n // 2))

    embeddings = encoder(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_of = kmeans.fit_predict(embeddings)
    allocation = _allocate(cluster_of, n_clusters, target_size)
    selected = _novelty_select(texts, cluster_of, allocation)
    return DrislResult(
        indices=selected,
        cluster_of=cluster_of,
        n_clusters=n_clusters,
        allocation=allocation.tolist(),
    )


def drisl_select_by_groups(
    texts: Sequence[str],
    target_size: int,
    groups: Sequence[str],
    lexical_novelty: bool = True,
    seed: int = 42,
) -> DrislResult:
    """Variante guiada pelo classificador (proposta do autor, 18/07/2026).

    Após o cold start, o k-means é substituído pelos grupos que o PRÓPRIO
    classificador induz: ``groups[i]`` é a classe prevista para ``texts[i]``.
    A alocação proporcional e a novidade lexical intra-grupo são as mesmas do
    DRI-SL clássico — muda apenas a origem do agrupamento, que passa a ser
    semanticamente alinhada ao espaço de rótulos e evolui com o modelo.
    """
    n = len(texts)
    if not 0 < target_size <= n:
        raise ValueError("target_size deve estar em (0, len(texts)].")
    uniq = sorted(set(groups))
    gid = {g: i for i, g in enumerate(uniq)}
    group_of = np.asarray([gid[g] for g in groups])
    allocation = _allocate(group_of, len(uniq), target_size)
    if lexical_novelty:
        selected = _novelty_select(texts, group_of, allocation)
    else:
        # ablação: sorteio simples dentro do grupo = estratificação pela predição
        rng = np.random.default_rng(seed)
        selected = []
        for c in range(len(uniq)):
            members = np.flatnonzero(group_of == c)
            quota = min(int(allocation[c]), len(members))
            if quota > 0:
                selected.extend(int(x) for x in rng.choice(members, size=quota, replace=False))
    return DrislResult(
        indices=selected,
        cluster_of=group_of,
        n_clusters=len(uniq),
        allocation=allocation.tolist(),
    )
