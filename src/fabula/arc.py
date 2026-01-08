from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np


def resample_to_n(x: Sequence[float], y: Sequence[float], n_points: int = 100) -> Tuple[List[float], List[float]]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) == 0:
        xs = np.linspace(0.0, 1.0, n_points)
        return xs.tolist(), [float("nan")] * n_points

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    uniq_x, inv = np.unique(x_arr, return_inverse=True)
    if len(uniq_x) != len(x_arr):
        sums = np.zeros_like(uniq_x, dtype=float)
        counts = np.zeros_like(uniq_x, dtype=float)
        for i, yi in zip(inv, y_arr):
            sums[i] += yi
            counts[i] += 1.0
        y_arr = sums / np.maximum(counts, 1.0)
        x_arr = uniq_x

    xs = np.linspace(0.0, 1.0, n_points)
    ys = np.interp(xs, x_arr, y_arr, left=y_arr[0], right=y_arr[-1])
    return xs.tolist(), ys.tolist()


def smooth_moving_average(y: Sequence[float], window: int = 7) -> List[float]:
    y_arr = np.asarray(y, dtype=float)
    if window <= 1 or len(y_arr) == 0:
        return y_arr.tolist()

    window = int(min(window, len(y_arr)))
    kernel = np.ones(window, dtype=float) / float(window)
    ys = np.convolve(y_arr, kernel, mode="same")
    return ys.tolist()

