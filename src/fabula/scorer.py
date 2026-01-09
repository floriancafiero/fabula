from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


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
    temperature: float = 1.0

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
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

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        torch = self.torch
        texts = list(texts)
        results: List[Dict[str, float]] = []

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
                logits = logits / float(self.temperature)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

                for row in probs:
                    d: Dict[str, float] = {}
                    for j, p in enumerate(row):
                        label = self.id2label.get(j, str(j)) if self.id2label else str(j)
                        d[label] = float(p)
                    results.append(d)

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
