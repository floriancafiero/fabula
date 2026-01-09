from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Segment:
    idx: int
    text: str
    rel_pos: float  # in [0, 1]
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    start_token: Optional[int] = None
    end_token: Optional[int] = None


@dataclass(frozen=True)
class ScoredSegment(Segment):
    probs: Optional[Dict[str, float]] = None  # label -> probability
    label: str = ""
    score: Optional[float] = None   # continuous valence if available


@dataclass(frozen=True)
class ArcResult:
    x: List[float]       # resampled positions in [0, 1]
    y: List[float]       # smoothed scores
    raw_x: List[float]   # raw segment positions
    raw_y: List[float]   # raw segment scores
    y_low: Optional[List[float]] = None   # lower CI bound if computed
    y_high: Optional[List[float]] = None  # upper CI bound if computed
