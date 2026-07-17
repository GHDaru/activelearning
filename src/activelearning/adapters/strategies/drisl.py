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


def drisl_select(
    texts: Sequence[str],
    target_size: int,
    encoder: Encoder,
    n_clusters: int | None = None,
    seed: int = 42,
) -> DrislResult:
    """Seleciona ``target_size`` índices do pool via densidade semântica + novidade lexical."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    if not 0 < target_size <= n:
        raise ValueError("target_size deve estar em (0, len(texts)].")
    n_clusters = n_clusters or max(2, min(int(math.sqrt(target_size)), n // 2))

    embeddings = encoder(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_of = kmeans.fit_predict(embeddings)

    # alocação proporcional ao tamanho do grupo (mínimo 1 por grupo não vazio)
    sizes = np.bincount(cluster_of, minlength=n_clusters)
    raw = sizes / sizes.sum() * target_size
    allocation = np.maximum(np.floor(raw).astype(int), (sizes > 0).astype(int))
    while allocation.sum() > target_size:  # apara excesso dos maiores
        allocation[int(np.argmax(allocation))] -= 1
    while allocation.sum() < target_size:  # distribui resto pelas maiores frações
        frac = raw - allocation
        frac[sizes == 0] = -1
        allocation[int(np.argmax(frac))] += 1

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", lowercase=True, strip_accents="ascii"
    )
    tfidf = vectorizer.fit_transform(texts)

    selected: list[int] = []
    for c in range(n_clusters):
        members = np.flatnonzero(cluster_of == c)
        quota = int(allocation[c])
        if quota <= 0 or len(members) == 0:
            continue
        # perfil TF-IDF do grupo = peso médio de cada termo dentro do grupo
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

    return DrislResult(
        indices=selected,
        cluster_of=cluster_of,
        n_clusters=n_clusters,
        allocation=allocation.tolist(),
    )
