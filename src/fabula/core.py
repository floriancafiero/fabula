from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .arc import resample_to_n, smooth_series
from .schemas import ArcResult
from .scorer import TransformersScorer, valence_from_probs
from .segment import RegexSentenceSegmenter

_ANALYSIS_TYPES = {"sentiment", "emotion"}


@dataclass
class Fabula:
    scorer: TransformersScorer
    segmenter: Any = None  # must implement .segment(text) -> List[Segment]
    analysis: str = "sentiment"

    def __post_init__(self):
        if self.segmenter is None:
            self.segmenter = RegexSentenceSegmenter()
        if self.analysis not in _ANALYSIS_TYPES:
            raise ValueError(f"Unknown analysis type: {self.analysis}")

    def _score_from_probs(self, probs: Dict[str, float]) -> Optional[float]:
        if self.analysis == "sentiment":
            return valence_from_probs(probs)
        if self.analysis == "emotion":
            return max(probs.values()) if probs else None
        raise ValueError(f"Unknown analysis type: {self.analysis}")

    def score(self, text: str) -> pd.DataFrame:
        segs = self.segmenter.segment(text)
        probs_list = self.scorer.predict_proba([s.text for s in segs])

        rows: List[Dict[str, Any]] = []
        for s, probs in zip(segs, probs_list):
            label = max(probs.items(), key=lambda kv: kv[1])[0] if probs else ""
            score = self._score_from_probs(probs)
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
        smooth_method: str = "moving_average",
        smooth_sigma: Optional[float] = None,
        smooth_pad_mode: str = "reflect",
        uncertainty_samples: int = 0,
        uncertainty_ci: float = 0.95,
        uncertainty_concentration: float = 50.0,
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
        y_sm = smooth_series(
            y_rs,
            method=smooth_method,
            window=smooth_window,
            sigma=smooth_sigma,
            pad_mode=smooth_pad_mode,
        )

        y_low = None
        y_high = None
        if uncertainty_samples > 0:
            if not (0.0 < uncertainty_ci < 1.0):
                raise ValueError("uncertainty_ci must be between 0 and 1")
            if uncertainty_concentration <= 0:
                raise ValueError("uncertainty_concentration must be > 0")

            samples = _sample_scores(
                df=df,
                analysis=self.analysis,
                score_col=score_col,
                fallback_to_maxprob=fallback_to_maxprob,
                n_samples=uncertainty_samples,
                concentration=uncertainty_concentration,
            )
            arc_samples = []
            for ys in samples:
                x_s, y_s = resample_to_n(raw_x, ys, n_points=n_points)
                y_s = smooth_series(
                    y_s,
                    method=smooth_method,
                    window=smooth_window,
                    sigma=smooth_sigma,
                    pad_mode=smooth_pad_mode,
                )
                arc_samples.append(y_s)

            arr = np.asarray(arc_samples, dtype=float)
            alpha = (1.0 - uncertainty_ci) / 2.0
            y_low = np.quantile(arr, alpha, axis=0).tolist()
            y_high = np.quantile(arr, 1.0 - alpha, axis=0).tolist()

        return ArcResult(x=x_rs, y=y_sm, raw_x=raw_x, raw_y=raw_y, y_low=y_low, y_high=y_high)


def _sample_scores(
    df: pd.DataFrame,
    analysis: str,
    score_col: str,
    fallback_to_maxprob: bool,
    n_samples: int,
    concentration: float,
) -> List[List[float]]:
    rng = np.random.default_rng()
    probs_list = df["probs"].tolist()
    raw_scores = df[score_col].astype(float).tolist()

    samples: List[List[float]] = [[] for _ in range(n_samples)]
    for probs, raw_score in zip(probs_list, raw_scores):
        if isinstance(probs, dict) and len(probs):
            labels = list(probs.keys())
            p = np.array([float(probs[k]) for k in labels], dtype=float)
            if p.sum() <= 0:
                p = np.ones_like(p) / float(len(p))
            else:
                p = p / p.sum()
            alpha = p * concentration
            draw = rng.dirichlet(alpha, size=n_samples)
            for i in range(n_samples):
                sample_probs = {label: float(draw[i][j]) for j, label in enumerate(labels)}
                if analysis == "sentiment":
                    score = valence_from_probs(sample_probs)
                else:
                    score = max(sample_probs.values()) if sample_probs else None
                samples[i].append(float(score) if score is not None else float("nan"))
        else:
            if fallback_to_maxprob and isinstance(probs, dict) and len(probs):
                value = float(max(probs.values()))
            else:
                value = float(raw_score) if not pd.isna(raw_score) else float("nan")
            for i in range(n_samples):
                samples[i].append(value)

    return samples
