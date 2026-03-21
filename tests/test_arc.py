import pytest
import math

from fabula.arc import resample_to_n, smooth_series


def test_resample_handles_duplicate_x_by_averaging_y():
    x = [0.0, 0.5, 0.5, 1.0]
    y = [0.0, 0.2, 0.4, 1.0]

    xs, ys = resample_to_n(x, y, n_points=3)

    assert xs == [0.0, 0.5, 1.0]
    assert ys[1] == pytest.approx(0.3)


def test_resample_empty_returns_nan_series():
    xs, ys = resample_to_n([], [], n_points=4)

    assert len(xs) == 4
    assert len(ys) == 4
    assert all(math.isnan(v) for v in ys)


def test_smooth_series_supports_aliases_and_none():
    y = [0.0, 1.0, 0.0, 1.0, 0.0]

    ma = smooth_series(y, method="ma", window=3)
    gauss = smooth_series(y, method="gauss", window=3, sigma=1.0)
    raw = smooth_series(y, method="none", window=3)

    assert len(ma) == len(y)
    assert len(gauss) == len(y)
    assert raw == y
