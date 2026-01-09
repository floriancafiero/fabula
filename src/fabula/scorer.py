from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class TransformersScorer:
    """
    Minimal scorer using Hugging Face Transformers.

    Imports torch/transformers lazily so fabula can be imported without them.
    """
    model: str
    device: Optional[str] = None  # "cpu", "cuda", "cuda:0"
    batch_size: int = 16
    max_length: int = 512
    pooling: str = "none"  # none, mean, max, attention
    pooling_stride_tokens: Optional[int] = None

    def __post_init__(self):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "TransformersScorer requires 'transformers' and 'torch'. "
                "Install with: pip install fabula[transformers]"
            ) from e

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        self.model_obj = AutoModelForSequenceClassification.from_pretrained(self.model)
        self.model_obj.eval()

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_obj.to(self.device)

        cfg = getattr(self.model_obj, "config", None)
        id2label = getattr(cfg, "id2label", None) if cfg is not None else None
        self.id2label = {int(k): v for k, v in id2label.items()} if isinstance(id2label, dict) else None
        if self.pooling not in {"none", "mean", "max", "attention"}:
            raise ValueError("pooling must be one of: none, mean, max, attention.")
        if self.pooling_stride_tokens is not None and self.pooling_stride_tokens <= 0:
            raise ValueError("pooling_stride_tokens must be positive.")

    def _row_to_dict(self, row: Iterable[float]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for j, p in enumerate(row):
            label = self.id2label.get(j, str(j)) if self.id2label else str(j)
            d[label] = float(p)
        return d

    def _pool_chunk_probs(self, probs: List[List[float]]) -> List[float]:
        if not probs:
            return []
        if self.pooling == "mean":
            n = len(probs)
            return [sum(row[i] for row in probs) / n for i in range(len(probs[0]))]
        if self.pooling == "max":
            return [max(row[i] for row in probs) for i in range(len(probs[0]))]
        if self.pooling == "attention":
            import math
            weights = [max(row) for row in probs]
            max_w = max(weights)
            exp_w = [math.exp(w - max_w) for w in weights]
            norm = sum(exp_w) or 1.0
            exp_w = [w / norm for w in exp_w]
            return [
                sum(w * row[i] for w, row in zip(exp_w, probs))
                for i in range(len(probs[0]))
            ]
        return probs[0]

    def _predict_batch_probs(self, enc) -> List[List[float]]:
        torch = self.torch
        probs_list: List[List[float]] = []
        self.model_obj.eval()
        with torch.no_grad():
            for i in range(0, enc["input_ids"].shape[0], self.batch_size):
                batch = {k: v[i:i + self.batch_size].to(self.device) for k, v in enc.items()}
                logits = self.model_obj(**batch).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                probs_list.extend(probs.tolist())
        return probs_list

    def _predict_with_pooling(self, text: str) -> Dict[str, float]:
        stride = self.pooling_stride_tokens
        if stride is None:
            stride = max(1, self.max_length // 4)
        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            stride=stride,
            return_tensors="pt",
        )
        probs_list = self._predict_batch_probs(enc)
        pooled = self._pool_chunk_probs(probs_list)
        return self._row_to_dict(pooled)

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        torch = self.torch
        texts = list(texts)
        results: List[Dict[str, float]] = []

        if self.pooling != "none":
            for text in texts:
                results.append(self._predict_with_pooling(text))
            return results

        self.model_obj.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model_obj(**enc).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

                for row in probs:
                    results.append(self._row_to_dict(row))

        return results


def valence_from_probs(probs: Dict[str, float]) -> Optional[float]:
    if not probs:
        return None

    keys = {k.lower() for k in probs.keys()}

    def get(name: str) -> float:
        for k, v in probs.items():
            if k.lower() == name:
                return float(v)
        return 0.0

    if "positive" in keys and "negative" in keys:
        return get("positive") - get("negative")
    if "pos" in keys and "neg" in keys:
        return get("pos") - get("neg")
    if "neutral" in keys and "positive" in keys and "negative" in keys:
        return get("positive") - get("negative")

    return None
