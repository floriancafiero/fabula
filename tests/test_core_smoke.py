import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Sequence

from fabula.core import Fabula
from fabula.schemas import Segment

@dataclass
class DummyScorer:
    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        return [{"POSITIVE": 0.6, "NEGATIVE": 0.4} for _ in texts]


@dataclass
class EmotionDummyScorer:
    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        return [{"JOIE": 0.2, "TRISTESSE": 0.8} for _ in texts]

@dataclass
class VariedDummyScorer:
    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        out = []
        for text in texts:
            if "chunk" in text:
                out.append({"POSITIVE": 0.1, "NEGATIVE": 0.9})
            else:
                out.append({"POSITIVE": 0.9, "NEGATIVE": 0.1})
        return out


class StaticSegmenter:
    def __init__(self, segments: List[Segment]) -> None:
        self._segments = segments

    def segment(self, text: str) -> List[Segment]:
        return self._segments


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


def test_fabula_emotion_arc_series():
    f = Fabula(scorer=EmotionDummyScorer(), analysis="emotion")
    arc = f.arc("Bonjour. Triste.", n_points=5, smooth_window=1, score_col="probs")
    assert arc.y is None
    assert arc.y_series is not None
    assert set(arc.y_series) == {"JOIE", "TRISTESSE"}
    assert len(arc.x) == 5
    assert len(arc.y_series["JOIE"]) == 5


def test_fabula_chunk_blending():
    fine_segments = [
        Segment(idx=0, text="fine one", rel_pos=0.1),
        Segment(idx=1, text="fine two", rel_pos=0.9),
    ]
    coarse_segments = [
        Segment(idx=0, text="chunk one", rel_pos=0.0),
        Segment(idx=1, text="chunk two", rel_pos=1.0),
    ]
    f = Fabula(
        scorer=VariedDummyScorer(),
        segmenter=StaticSegmenter(fine_segments),
        coarse_segmenter=StaticSegmenter(coarse_segments),
        chunk_weight=0.5,
        chunk_attention_tau=1.0,
    )
    df = f.score("ignored")
    assert "chunk_probs" in df.columns
    assert df["chunk_probs"].iloc[0]["NEGATIVE"] > 0.1
    assert df["score"].iloc[0] is not None
