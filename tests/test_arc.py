from fabula.arc import resample_to_n, smooth_moving_average

def test_resample_basic():
    x = [0.0, 0.5, 1.0]
    y = [0.0, 1.0, 0.0]
    xs, ys = resample_to_n(x, y, n_points=5)
    assert len(xs) == 5
    assert len(ys) == 5
    assert ys[0] == 0.0
    assert ys[-1] == 0.0

def test_smooth_window_1_identity():
    y = [0.0, 1.0, 0.0]
    ys = smooth_moving_average(y, window=1)
    assert ys == y

