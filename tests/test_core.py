import math

from fabula.core import Fabula
from fabula.scorer import valence_from_probs


class DummyScorer:
    def predict_proba(self, texts):
        out = []
        for t in texts:
            if "good" in t.lower():
                out.append({"positive": 0.9, "negative": 0.1})
            else:
                out.append({"positive": 0.2, "negative": 0.8})
        return out


def test_valence_from_probs_positive_minus_negative():
    probs = {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
    assert valence_from_probs(probs) == 0.6


def test_fabula_score_and_arc_without_transformers():
    fb = Fabula(scorer=DummyScorer())
    text = "Good day. bad day."

    df = fb.score(text)
    arc = fb.arc(text, n_points=8, smooth_window=3)

    assert len(df) == 2
    assert set(["label", "score", "probs", "entropy"]).issubset(df.columns)
    assert len(arc.x) == 8
    assert len(arc.y) == 8
    assert math.isclose(df.iloc[0]["score"], 0.8)
