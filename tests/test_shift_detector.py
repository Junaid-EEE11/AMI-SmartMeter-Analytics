"""
Automated unit tests for Distribution Shift Detectors.
"""
import numpy as np
import pytest

from shift_ami.conformal.shift_detector import WassersteinShiftDetector, StandardizedResidualShiftDetector


def test_wasserstein_shift_detector_stationary_vs_shifted():
    rng = np.random.default_rng(42)
    ref_scores = rng.normal(loc=10.0, scale=2.0, size=500)

    detector = WassersteinShiftDetector(window_length=100, reference_length=500, threshold=1.5)
    detector.set_reference(ref_scores)

    # 1. Feed stationary data (same mean and variance)
    stat_scores = rng.normal(loc=10.0, scale=2.0, size=100)
    for s in stat_scores:
        score, flag = detector.update_and_detect(s)
    # Stationary data should NOT trigger shift
    assert not flag
    assert score < 1.5

    # 2. Feed massive shifted distribution (mean shifted from 10 to 20)
    shifted_scores = rng.normal(loc=20.0, scale=4.0, size=100)
    flags = []
    for s in shifted_scores:
        score, flag = detector.update_and_detect(s)
        flags.append(flag)

    # Shifted regime MUST trigger detection flag
    assert any(flags)
    assert score > 2.0


def test_standardized_residual_detector():
    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, 200)
    det = StandardizedResidualShiftDetector(window_length=50, threshold=2.0)
    det.set_reference(ref)

    # Feed large mean shift
    for _ in range(50):
        z, flag = det.update_and_detect(5.0)
    assert flag
    assert z > 2.0
