"""Tests for NLI probability calibration."""

from __future__ import annotations

import pytest


def test_isotonic_calibrator_fit_and_calibrate() -> None:
    """Isotonic calibrator should produce monotonic calibrated probabilities."""
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")

    from nerve.nli_calibration import fit_isotonic_calibrator

    # Synthetic data: predictions slightly overconfident
    predictions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 1]  # Perfectly calibrated would be ~50% at 0.5

    calibrator = fit_isotonic_calibrator(predictions, labels)

    # Calibrator should have knots
    assert len(calibrator.x_knots) > 0
    assert len(calibrator.y_knots) > 0

    # Calibrated probabilities should be in [0, 1]
    for p in predictions:
        cal_p = calibrator.calibrate(p)
        assert 0.0 <= cal_p <= 1.0


def test_isotonic_calibrator_monotonic() -> None:
    """Isotonic calibration should preserve monotonicity."""
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")

    from nerve.nli_calibration import fit_isotonic_calibrator

    predictions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 1, 0, 1, 0, 1, 1, 1]

    calibrator = fit_isotonic_calibrator(predictions, labels)

    # Calibrated values should be monotonically increasing
    calibrated = [calibrator.calibrate(p) for p in predictions]
    for i in range(len(calibrated) - 1):
        assert calibrated[i] <= calibrated[i + 1]


def test_platt_calibrator_fit_and_calibrate() -> None:
    """Platt calibrator should produce calibrated probabilities."""
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")

    from nerve.nli_calibration import fit_platt_calibrator

    predictions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 1]

    calibrator = fit_platt_calibrator(predictions, labels)

    # Parameters should be learned
    assert calibrator.params.A != 0.0 or calibrator.params.B != 0.0

    # Calibrated probabilities should be in [0, 1]
    for p in predictions:
        cal_p = calibrator.calibrate(p)
        assert 0.0 <= cal_p <= 1.0


def test_calibrate_probabilities() -> None:
    """calibrate_probabilities should apply calibrator to sequence."""
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")

    from nerve.nli_calibration import (
        calibrate_probabilities,
        fit_isotonic_calibrator,
    )

    predictions = [0.1, 0.3, 0.5, 0.7, 0.9]
    labels = [0, 0, 1, 1, 1]

    calibrator = fit_isotonic_calibrator(predictions, labels)
    result = calibrate_probabilities(predictions, calibrator)

    assert len(result.calibrated_probs) == len(predictions)
    assert result.calibration_method == "isotonic"
    assert result.original_probs == list(predictions)


def test_calibrate_nli_result() -> None:
    """calibrate_nli_result should calibrate both support and contradict probs."""
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")

    from nerve.nli_calibration import (
        NLICalibrationConfig,
        calibrate_nli_result,
        fit_isotonic_calibrator,
    )

    # Fit calibrators on synthetic data
    support_preds = [0.1, 0.3, 0.5, 0.7, 0.9]
    support_labels = [0, 0, 1, 1, 1]
    support_cal = fit_isotonic_calibrator(support_preds, support_labels)

    contradict_preds = [0.1, 0.3, 0.5, 0.7, 0.9]
    contradict_labels = [0, 0, 0, 1, 1]
    contradict_cal = fit_isotonic_calibrator(contradict_preds, contradict_labels)

    config = NLICalibrationConfig(
        method="isotonic",
        support_calibrator=support_cal,
        contradict_calibrator=contradict_cal,
    )

    cal_support, cal_contradict = calibrate_nli_result(0.6, 0.4, config)

    assert 0.0 <= cal_support <= 1.0
    assert 0.0 <= cal_contradict <= 1.0


def test_compute_calibration_error() -> None:
    """compute_calibration_error should return ECE and MCE metrics."""
    from nerve.nli_calibration import compute_calibration_error

    # Perfectly calibrated predictions
    predictions = [0.2] * 10 + [0.8] * 10
    labels = [0, 1] * 5 + [0, 1] * 5  # 50% accuracy in each bin

    metrics = compute_calibration_error(predictions, labels, n_bins=5)

    assert "ece" in metrics
    assert "mce" in metrics
    assert "mean_confidence" in metrics
    assert "mean_accuracy" in metrics
    assert metrics["ece"] >= 0.0
    assert metrics["mce"] >= 0.0


def test_calibrator_edge_cases() -> None:
    """Test calibrators with edge case inputs."""
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")

    from nerve.nli_calibration import fit_isotonic_calibrator, fit_platt_calibrator

    # Too few samples
    iso_cal = fit_isotonic_calibrator([0.5], [1])
    assert len(iso_cal.x_knots) == 0

    platt_cal = fit_platt_calibrator([0.5], [1])
    assert platt_cal.params.A == 1.0
    assert platt_cal.params.B == 0.0

    # Calibrating with empty calibrator should return original
    result = iso_cal.calibrate(0.5)
    assert result == 0.5


def test_isotonic_extrapolation() -> None:
    """Isotonic calibrator should handle values outside training range."""
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")

    from nerve.nli_calibration import fit_isotonic_calibrator

    predictions = [0.3, 0.4, 0.5, 0.6, 0.7]
    labels = [0, 0, 1, 1, 1]

    calibrator = fit_isotonic_calibrator(predictions, labels)

    # Values outside range should be clipped
    assert 0.0 <= calibrator.calibrate(0.1) <= 1.0
    assert 0.0 <= calibrator.calibrate(0.99) <= 1.0
