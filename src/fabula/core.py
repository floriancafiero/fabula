from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from .arc import resample_to_n, smooth_moving_average
from .schemas import ArcResult
from .scorer import TransformersScorer, valence_from_probs
from .segment import RegexSentenceSegmenter


@dataclass
class Fabula:
    scorer: TransformersScorer
    segmenter: Any = None  # must implement .segment(text) -> List[Segment]

    def __post_init__(self):
        if self.segmenter is None:
            self.segmenter = RegexSentenceSegmenter()

    def score(self, text: str) -> pd.DataFrame:
        segs = self.segmenter.segment(text)
        probs_list = self.scorer.predict_proba([s.text for s in segs])

        rows: List[Dict[str, Any]] = []
        for s, probs in zip(segs, probs_list):
            label = max(probs.items(), key=lambda kv: kv[1])[0] if probs else ""
            score = valence_from_probs(probs)
            rows.append(
                {
                    "idx": s.idx,
                    "rel_pos": float(s.rel_pos),
                    "text": s.text,
                    "label": label,
                    "score": score,
                    "probs": probs,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                    "start_token": s.start_token,
                    "end_token": s.end_token,
                }
            )

        return pd.DataFrame(rows)

    def arc(
        self,
        text: str,
        n_points: int = 100,
        smooth_window: int = 7,
        score_col: str = "score",
        fallback_to_maxprob: bool = True,
    ) -> ArcResult:
        df = self.score(text)

        raw_x = df["rel_pos"].astype(float).tolist()
        if score_col not in df.columns:
            raise ValueError(f"Missing column: {score_col}")

        raw_y = df[score_col].astype(float).tolist()

        if fallback_to_maxprob:
            mask = pd.isna(df[score_col])
            if bool(mask.any()):
                probs_list = df["probs"].tolist()
                raw_y = [
                    (
                        max(p.values())
                        if missing and isinstance(p, dict) and len(p)
                        else y
                    )
                    for y, p, missing in zip(raw_y, probs_list, mask.tolist())
                ]

        x_rs, y_rs = resample_to_n(raw_x, raw_y, n_points=n_points)
        y_sm = smooth_moving_average(y_rs, window=smooth_window)

        return ArcResult(x=x_rs, y=y_sm, raw_x=raw_x, raw_y=raw_y)
