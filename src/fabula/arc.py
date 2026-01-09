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


def smooth_moving_average(y: Sequence[float], window: int = 7, pad_mode: str = "reflect") -> List[float]:
    y_arr = np.asarray(y, dtype=float)
    if window <= 1 or len(y_arr) == 0:
        return y_arr.tolist()

    window = int(min(window, len(y_arr)))
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=float) / float(window)
    ys = _convolve_with_padding(y_arr, kernel, pad_mode=pad_mode)
    return ys.tolist()


def smooth_gaussian(y: Sequence[float], window: int = 7, sigma: float | None = None, pad_mode: str = "reflect") -> List[float]:
    y_arr = np.asarray(y, dtype=float)
    if window <= 1 or len(y_arr) == 0:
        return y_arr.tolist()

    window = int(min(window, len(y_arr)))
    if window % 2 == 0:
        window += 1

    if sigma is None or sigma <= 0:
        sigma = max(window / 6.0, 1e-6)

    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    ys = _convolve_with_padding(y_arr, kernel, pad_mode=pad_mode)
    return ys.tolist()


def smooth_series(
    y: Sequence[float],
    method: str = "moving_average",
    window: int = 7,
    sigma: float | None = None,
    pad_mode: str = "reflect",
) -> List[float]:
    method = method.lower()
    if method in {"none", "raw"}:
        return list(np.asarray(y, dtype=float))
    if method in {"moving_average", "mean", "ma"}:
        return smooth_moving_average(y, window=window, pad_mode=pad_mode)
    if method in {"gaussian", "gauss"}:
        return smooth_gaussian(y, window=window, sigma=sigma, pad_mode=pad_mode)
    raise ValueError(f"Unknown smoothing method: {method}")


def _convolve_with_padding(y_arr: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    pad = len(kernel) // 2
    if pad <= 0:
        return y_arr

    pad_kwargs = {}
    if pad_mode == "constant":
        pad_kwargs["constant_values"] = 0.0

    try:
        padded = np.pad(y_arr, pad_width=pad, mode=pad_mode, **pad_kwargs)
    except ValueError as e:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}") from e

    return np.convolve(padded, kernel, mode="valid")
