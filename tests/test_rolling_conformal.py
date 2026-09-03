"""
Automated unit tests for Rolling Window Conformal Recalibration (P3).
"""
import numpy as np
import pytest

from shift_ami.conformal.rolling import RollingWindowConformal


def test_rolling_conformal_sliding_window():
    cal_scores = np.ones(50) * 2.0  # Initial calibration scores = 2.0
    rolling = RollingWindowConformal(alpha=0.10, window_size=50)
    rolling.initialize_with_calibration_scores(cal_scores)

    assert np.isclose(rolling.current_quantile, 2.0, atol=1e-2)

    # Inject 50 new scores of 10.0 (simulating large residual shift)
    for _ in range(50):
        rolling.update(10.0)

    # Rolling window should now be completely filled with 10.0
    assert np.isclose(rolling.current_quantile, 10.0, atol=1e-2)


def test_rolling_buffer_maxlen():
    rolling = RollingWindowConformal(alpha=0.10, window_size=10)
    for i in range(100):
        rolling.update(float(i))
    assert len(rolling.buffer) == 10
    assert rolling.buffer[0] == 90.0
    assert rolling.buffer[-1] == 99.0
