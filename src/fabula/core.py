from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .arc import resample_to_n, smooth_series
from .schemas import ArcResult
from .scorer import TransformersScorer, valence_from_probs
from .segment import RegexSentenceSegmenter

_ANALYSIS_TYPES = {"sentiment", "emotion"}


@dataclass
class Fabula:
    scorer: TransformersScorer
    segmenter: Any = None  # must implement .segment(text) -> List[Segment]
    coarse_segmenter: Any = None  # optional chunk segmenter
    analysis: str = "sentiment"
    chunk_weight: float = 0.3
    chunk_attention_tau: float = 0.1

    def __post_init__(self):
        if self.segmenter is None:
            self.segmenter = RegexSentenceSegmenter()
        if self.analysis not in _ANALYSIS_TYPES:
            raise ValueError(f"Unknown analysis type: {self.analysis}")
        if not 0.0 <= self.chunk_weight <= 1.0:
            raise ValueError("chunk_weight must be in [0, 1].")
        if self.chunk_attention_tau <= 0:
            raise ValueError("chunk_attention_tau must be > 0.")

    def _attention_pool_probs(
        self,
        probs_list: Iterable[Dict[str, float]],
        positions: Iterable[float],
        query_pos: float,
    ) -> Dict[str, float]:
        probs_list = list(probs_list)
        positions = list(positions)
        if not probs_list:
            return {}

        weights = []
        for pos in positions:
            dist = abs(query_pos - pos)
            weights.append(math.exp(-dist / self.chunk_attention_tau))

        norm = sum(weights) or 1.0
        weights = [w / norm for w in weights]

        pooled: Dict[str, float] = {}
        for w, probs in zip(weights, probs_list):
            for label, val in probs.items():
                pooled[label] = pooled.get(label, 0.0) + w * float(val)
        return pooled

    def _blend_probs(
        self,
        fine: Dict[str, float],
        coarse: Dict[str, float],
    ) -> Dict[str, float]:
        if not coarse:
            return fine
        if not fine:
            return coarse
        labels = set(fine) | set(coarse)
        blend: Dict[str, float] = {}
        for label in labels:
            blend[label] = (
                (1.0 - self.chunk_weight) * float(fine.get(label, 0.0))
                + self.chunk_weight * float(coarse.get(label, 0.0))
            )
        return blend

    def _score_from_probs(self, probs: Dict[str, float]) -> Optional[float]:
        if self.analysis == "sentiment":
            return valence_from_probs(probs)
        if self.analysis == "emotion":
            return max(probs.values()) if probs else None
        raise ValueError(f"Unknown analysis type: {self.analysis}")

    def score(self, text: str) -> pd.DataFrame:
        segs = self.segmenter.segment(text)
        probs_list = self.scorer.predict_proba([s.text for s in segs])

        coarse_segs = []
        coarse_probs_list: List[Dict[str, float]] = []
        coarse_positions: List[float] = []
        if self.coarse_segmenter is not None:
            coarse_segs = self.coarse_segmenter.segment(text)
            coarse_probs_list = self.scorer.predict_proba([s.text for s in coarse_segs])
            coarse_positions = [c.rel_pos for c in coarse_segs]

        rows: List[Dict[str, Any]] = []
        for s, probs in zip(segs, probs_list):
            pooled_probs: Dict[str, float] = {}
            if coarse_segs:
                pooled_probs = self._attention_pool_probs(
                    coarse_probs_list,
                    coarse_positions,
                    s.rel_pos,
                )

            merged_probs = self._blend_probs(probs, pooled_probs)
            label = max(merged_probs.items(), key=lambda kv: kv[1])[0] if merged_probs else ""
            score = self._score_from_probs(merged_probs)
            rows.append(
                {
                    "idx": s.idx,
                    "rel_pos": float(s.rel_pos),
                    "text": s.text,
                    "label": label,
                    "score": score,
                    "probs": merged_probs,
                    "chunk_probs": pooled_probs if pooled_probs else None,
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
        score_col: str = "score",
        score_cols: Optional[Sequence[str]] = None,
        fallback_to_maxprob: bool = True,
    ) -> ArcResult:
        df = self.score(text)

        raw_x = df["rel_pos"].astype(float).tolist()
        if score_cols:
            missing = [col for col in score_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {', '.join(missing)}")
            raw_y_series = {
                col: df[col].astype(float).tolist()
                for col in score_cols
            }
            y_series = {}
            x_rs: List[float] = []
            for col, raw_vals in raw_y_series.items():
                x_rs, y_rs = resample_to_n(raw_x, raw_vals, n_points=n_points)
                y_series[col] = smooth_series(
                    y_rs,
                    method=smooth_method,
                    window=smooth_window,
                    sigma=smooth_sigma,
                    pad_mode=smooth_pad_mode,
                )
            return ArcResult(
                x=x_rs,
                y=None,
                raw_x=raw_x,
                raw_y=None,
                y_series=y_series,
                raw_y_series=raw_y_series,
            )

        if score_col == "probs":
            probs_list = df["probs"].tolist()
            labels = sorted(
                {label for p in probs_list if isinstance(p, dict) for label in p}
            )
            raw_y_series = {
                label: [
                    float(p.get(label, 0.0)) if isinstance(p, dict) else float("nan")
                    for p in probs_list
                ]
                for label in labels
            }
            y_series = {}
            x_rs = []
            for label, raw_vals in raw_y_series.items():
                x_rs, y_rs = resample_to_n(raw_x, raw_vals, n_points=n_points)
                y_series[label] = smooth_series(
                    y_rs,
                    method=smooth_method,
                    window=smooth_window,
                    sigma=smooth_sigma,
                    pad_mode=smooth_pad_mode,
                )
            return ArcResult(
                x=x_rs,
                y=None,
                raw_x=raw_x,
                raw_y=None,
                y_series=y_series,
                raw_y_series=raw_y_series,
            )

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

        return ArcResult(x=x_rs, y=y_sm, raw_x=raw_x, raw_y=raw_y)
