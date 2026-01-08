import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Sequence

from fabula.core import Fabula

@dataclass
class DummyScorer:
    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        return [{"POSITIVE": 0.6, "NEGATIVE": 0.4} for _ in texts]


@dataclass
class EmotionDummyScorer:
    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        return [{"JOIE": 0.2, "TRISTESSE": 0.8} for _ in texts]


def test_fabula_score_and_arc():
    f = Fabula(scorer=DummyScorer())
    df = f.score("Bonjour. Triste.")
    assert isinstance(df, pd.DataFrame)
    assert "score" in df.columns

    arc = f.arc("Bonjour. Triste.", n_points=10, smooth_window=3)
    assert len(arc.x) == 10
    assert len(arc.y) == 10


def test_fabula_emotion_scores():
    f = Fabula(scorer=EmotionDummyScorer(), analysis="emotion")
    df = f.score("Bonjour. Triste.")
    assert df["score"].iloc[0] == 0.8
