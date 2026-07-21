"""PVBin — classificador vetorial por protótipo de classe (porte do legado).

Porte fiel do ``ProductVectorizer`` de ``activetextclassification`` (read-only),
na configuração vencedora da dissertação do autor \\citep{Daru2024Dissertacao}:
representação binária, uni+bigramas, sem normalização. Cada classe vira UM
protótipo (concatenação de seus documentos); a predição é o argmax da similaridade
protótipo·consulta, com softmax sobre os escores para probabilidades.

Diferenças deliberadas em relação ao legado (registradas para a validação B1c):
- ordem de classes determinística (ordenada), não dependente de ``set()``;
- API alinhada à porta TaskClassifier: ``predict_proba`` retorna
  ``(n_amostras, n_classes)`` com colunas em ``self.classes_``;
- sem dependência de pandas.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer

_METHOD_OPTIONS = {
    "binary": {"binary": True, "use_idf": False},
    "termfrequency": {"binary": False, "use_idf": False},
    "tfidf": {"use_idf": True, "smooth_idf": False},
}


def _make_vectorizer(method: str, ngram_range: tuple[int, int], norm: str | None) -> TfidfVectorizer:
    if method not in _METHOD_OPTIONS:
        raise ValueError(f"método desconhecido: {method!r} (opções: {sorted(_METHOD_OPTIONS)})")
    return TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
        strip_accents="ascii",
        ngram_range=ngram_range,
        norm=norm,
        **_METHOD_OPTIONS[method],
    )


class PVBinClassifier:
    """Classificador por protótipo. Configuração padrão = Binary [1,2] sem norma."""

    def __init__(
        self,
        method: str = "binary",
        ngram_range: tuple[int, int] = (1, 2),
        norm: str | None = None,
        query_method: str = "binary",
        query_norm: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        self._method = method
        self._ngram_range = tuple(ngram_range)
        self._norm = norm
        self._query_method = query_method
        self._query_norm = query_norm
        self._temperature = float(temperature)
        self._doc_vectorizer: TfidfVectorizer | None = None
        self._query_vectorizer: TfidfVectorizer | None = None
        self._prototype_matrix = None
        self.classes_: list[str] = []

    # ------------------------------------------------------------------ API
    def fit(self, texts: list[str], labels: list[str]) -> "PVBinClassifier":
        """Treina um protótipo por classe (concatenação dos textos) e o vetoriza.

        ``texts`` e ``labels`` devem ter o mesmo tamanho (> 0). Retorna ``self``.
        """
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts e labels devem ter o mesmo tamanho (> 0).")
        texts_arr = np.asarray(texts, dtype=object)
        labels_arr = np.asarray(labels, dtype=object)
        # ordem determinística (o legado usava set(), com ordem instável)
        self.classes_ = sorted(set(labels))
        prototypes = [
            " ".join(texts_arr[labels_arr == c]) for c in self.classes_
        ]
        self._doc_vectorizer = _make_vectorizer(self._method, self._ngram_range, self._norm)
        self._doc_vectorizer.fit(prototypes)
        self._query_vectorizer = _make_vectorizer(
            self._query_method, self._ngram_range, self._query_norm
        )
        self._query_vectorizer.fit(prototypes)
        self._prototype_matrix = self._doc_vectorizer.transform(prototypes)
        return self

    def _scores(self, texts: list[str]) -> np.ndarray:
        if self._prototype_matrix is None:
            raise RuntimeError("PVBin não treinado — chame fit() primeiro.")
        query = self._query_vectorizer.transform(list(texts))
        # (n_classes, n_amostras) -> transpõe para (n_amostras, n_classes)
        return np.asarray((self._prototype_matrix @ query.T).todense()).T

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Distribuição de probabilidade por classe (softmax dos escores/temperatura).

        Retorna array ``(n_amostras, n_classes)`` alinhado a ``classes_``.
        """
        scores = self._scores(texts)
        return softmax(scores / self._temperature, axis=1)

    def predict(self, texts: list[str]) -> list[str]:
        """Rótulo mais provável de cada texto (``argmax`` sobre ``classes_``)."""
        scores = self._scores(texts)
        return [self.classes_[i] for i in scores.argmax(axis=1)]

    def score_macro_f1(self, texts: list[str], gold: list[str]) -> float:
        """Macro F1 das predições contra ``gold`` (implacável com classes raras)."""
        from sklearn.metrics import f1_score

        return float(f1_score(gold, self.predict(texts), average="macro", zero_division=0))

    def score_accuracy(self, texts: list[str], gold: list[str]) -> float:
        """Acurácia das predições contra ``gold``."""
        pred = self.predict(texts)
        return float(np.mean([p == g for p, g in zip(pred, gold)]))
