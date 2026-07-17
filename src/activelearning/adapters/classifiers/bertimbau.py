"""BERTimbau — classificador forte por ajuste fino (E2/E3, bloco H).

Adapter da porta TaskClassifier sobre ``neuralmind/bert-base-portuguese-cased``
\\citep{Souza2020BERTimbau}. Desenhado para a estação com GPU (RTX 3090), mas
executável em CPU para smoke tests com subconjuntos pequenos — o custo em CPU
cresce linearmente com épocas × instâncias × comprimento máximo de tokens.

Dependências opcionais (torch, transformers) são importadas tardiamente para
não onerar quem só usa o PVBin.
"""
from __future__ import annotations

import numpy as np


class BertimbauClassifier:
    """Ajuste fino de BERTimbau para classificação de texto curto.

    API alinhada à porta TaskClassifier (mesma do PVBin): ``fit(texts, labels)``,
    ``predict``, ``predict_proba`` com colunas na ordem de ``self.classes_``.
    """

    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 32,
        learning_rate: float = 5e-5,
        seed: int = 42,
        device: str | None = None,
        progress: bool = False,
    ) -> None:
        self._model_name = model_name
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._max_length = int(max_length)
        self._learning_rate = float(learning_rate)
        self._seed = int(seed)
        self._device_arg = device
        self._progress = bool(progress)
        self.classes_: list[str] = []
        self._model = None
        self._tokenizer = None

    # -- infraestrutura -----------------------------------------------------
    def _lazy_imports(self):
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        return torch, AutoTokenizer, AutoModelForSequenceClassification

    def _device(self, torch):
        if self._device_arg:
            return torch.device(self._device_arg)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _encode(self, torch, texts: list[str]):
        enc = self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        return enc

    # -- porta TaskClassifier ----------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> "BertimbauClassifier":
        if not texts or len(texts) != len(labels):
            raise ValueError("fit exige texts e labels não vazios e do mesmo tamanho.")
        torch, AutoTokenizer, AutoModel = self._lazy_imports()
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        self.classes_ = sorted(set(labels))
        label_to_id = {c: i for i, c in enumerate(self.classes_)}
        y = torch.tensor([label_to_id[l] for l in labels])

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(
            self._model_name, num_labels=len(self.classes_)
        )
        device = self._device(torch)
        self._model.to(device)
        self._model.train()

        enc = self._encode(torch, texts)
        dataset = torch.utils.data.TensorDataset(
            enc["input_ids"], enc["attention_mask"], y
        )
        generator = torch.Generator().manual_seed(self._seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, generator=generator
        )
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._learning_rate
        )
        for epoch in range(self._epochs):
            total = 0.0
            for step, (ids, mask, yy) in enumerate(loader):
                ids, mask, yy = ids.to(device), mask.to(device), yy.to(device)
                optimizer.zero_grad()
                out = self._model(input_ids=ids, attention_mask=mask, labels=yy)
                out.loss.backward()
                optimizer.step()
                total += float(out.loss.detach())
                if self._progress and step % 10 == 0:
                    print(f"  época {epoch+1}/{self._epochs} passo {step+1}/{len(loader)} "
                          f"loss={float(out.loss.detach()):.4f}", flush=True)
            if self._progress:
                print(f"  época {epoch+1}: loss médio {total/max(1,len(loader)):.4f}", flush=True)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Chame fit antes de predict_proba.")
        torch, _, _ = self._lazy_imports()
        device = self._device(torch)
        self._model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), self._batch_size):
                enc = self._encode(torch, texts[i : i + self._batch_size])
                out = self._model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                probs.append(torch.softmax(out.logits, dim=-1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.classes_[i] for i in proba.argmax(axis=1)]
