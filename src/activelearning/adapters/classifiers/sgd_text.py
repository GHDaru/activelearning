"""SGD logístico sobre TF-IDF — classificador rápido para laços de AL.

Escolhido como o "algoritmo eficiente de treinar" do ciclo E2E (D-007):
regressão logística por gradiente estocástico em representação esparsa
(1,2)-gramas. Treina em segundos mesmo com centenas de classes e fornece
``predict_proba`` nativo (necessário às estratégias de incerteza).

Por que não XGBoost: com 621 classes o custo é um conjunto de árvores POR
classe sobre matriz esparsa de alta dimensão — minutos a horas por retreino,
exatamente o que um laço com dezenas de retreinos não comporta; e a
probabilidade calibrada exigiria etapa extra. O ponto do braço "clássico
rápido" é custo marginal de retreino desprezível.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


class SgdTextClassifier:
    """Porta TaskClassifier: fit/predict/predict_proba com colunas em classes_."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        alpha: float = 1e-5,
        max_iter: int = 20,
        seed: int = 42,
    ) -> None:
        self._ngram_range = tuple(ngram_range)
        self._alpha = float(alpha)
        self._max_iter = int(max_iter)
        self._seed = int(seed)
        self._vectorizer: TfidfVectorizer | None = None
        self._model: SGDClassifier | None = None
        self.classes_: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> "SgdTextClassifier":
        """Treina TF-IDF + regressão logística (SGD, ``log_loss``). Retorna ``self``."""
        if not texts or len(texts) != len(labels):
            raise ValueError("fit exige texts e labels não vazios e do mesmo tamanho.")
        self._vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", lowercase=True, strip_accents="ascii",
            ngram_range=self._ngram_range,
        )
        x = self._vectorizer.fit_transform(texts)
        self._model = SGDClassifier(
            loss="log_loss", alpha=self._alpha, max_iter=self._max_iter,
            random_state=self._seed, tol=1e-3,
        )
        self._model.fit(x, labels)
        self.classes_ = list(self._model.classes_)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probabilidades por classe ``(n_amostras, n_classes)``, alinhadas a ``classes_``."""
        if self._model is None or self._vectorizer is None:
            raise RuntimeError("Chame fit antes de predict_proba.")
        return self._model.predict_proba(self._vectorizer.transform(texts))

    def predict(self, texts: list[str]) -> list[str]:
        """Rótulo mais provável de cada texto."""
        proba = self.predict_proba(texts)
        return [self.classes_[i] for i in proba.argmax(axis=1)]
