"""NLI probability calibration using isotonic regression and Platt scaling.

This module provides post-hoc calibration methods for NLI confidence scores
to ensure that predicted probabilities are well-calibrated (i.e., when the
model predicts 70% confidence, it should be correct ~70% of the time).

Supported calibration methods:
- Isotonic regression: Non-parametric monotonic calibration
- Platt scaling: Parametric sigmoid-based calibration (logistic regression)

Usage:
    1. Collect NLI predictions and ground truth labels on a validation set
    2. Fit a calibrator using `fit_isotonic_calibrator` or `fit_platt_calibrator`
    3. Apply the calibrator to new predictions using `calibrate_probabilities`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class CalibrationResult:
    """Result of probability calibration."""

    original_probs: list[float]
    calibrated_probs: list[float]
    calibration_method: str


@dataclass
class PlattParameters:
    """Parameters for Platt scaling: sigmoid(A * logit + B)."""

    A: float
    B: float


@dataclass
class IsotonicCalibrator:
    """Fitted isotonic regression calibrator.

    Stores the mapping from uncalibrated to calibrated probabilities.
    """

    x_knots: list[float]  # Original probability knots
    y_knots: list[float]  # Calibrated probability knots

    def calibrate(self, prob: float) -> float:
        """Calibrate a single probability using linear interpolation."""
        if not self.x_knots or not self.y_knots:
            return prob

        # Binary search for interval
        if prob <= self.x_knots[0]:
            return self.y_knots[0]
        if prob >= self.x_knots[-1]:
            return self.y_knots[-1]

        # Find interval and interpolate
        for i in range(len(self.x_knots) - 1):
            if self.x_knots[i] <= prob <= self.x_knots[i + 1]:
                # Linear interpolation
                x0, x1 = self.x_knots[i], self.x_knots[i + 1]
                y0, y1 = self.y_knots[i], self.y_knots[i + 1]
                if x1 - x0 < 1e-9:
                    return y0
                t = (prob - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        return prob


@dataclass
class PlattCalibrator:
    """Fitted Platt scaling calibrator."""

    params: PlattParameters

    def calibrate(self, prob: float) -> float:
        """Calibrate a single probability using Platt scaling."""
        import math

        # Convert prob to logit
        eps = 1e-7
        prob = max(eps, min(1 - eps, prob))
        logit = math.log(prob / (1 - prob))

        # Apply sigmoid(A * logit + B)
        scaled = self.params.A * logit + self.params.B
        return 1.0 / (1.0 + math.exp(-scaled))


def fit_isotonic_calibrator(
    predictions: Sequence[float],
    labels: Sequence[int],
) -> IsotonicCalibrator:
    """Fit an isotonic regression calibrator.

    Args:
        predictions: Predicted probabilities (0-1)
        labels: Ground truth binary labels (0 or 1)

    Returns:
        Fitted IsotonicCalibrator
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "Isotonic calibration requires scikit-learn and numpy. "
            "Install with: pip install scikit-learn numpy"
        ) from e

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have same length")

    if len(predictions) < 2:
        # Not enough data for calibration
        return IsotonicCalibrator(x_knots=[], y_knots=[])

    X = np.array(predictions)
    y = np.array(labels)

    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(X, y)

    # Extract calibration mapping
    x_knots = iso_reg.X_thresholds_.tolist()
    y_knots = iso_reg.y_thresholds_.tolist()

    return IsotonicCalibrator(x_knots=x_knots, y_knots=y_knots)


def fit_platt_calibrator(
    predictions: Sequence[float],
    labels: Sequence[int],
    *,
    max_iter: int = 100,
) -> PlattCalibrator:
    """Fit a Platt scaling calibrator.

    Platt scaling fits a logistic regression on the log-odds of predictions
    to produce calibrated probabilities.

    Args:
        predictions: Predicted probabilities (0-1)
        labels: Ground truth binary labels (0 or 1)
        max_iter: Maximum iterations for optimization

    Returns:
        Fitted PlattCalibrator
    """
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "Platt calibration requires scikit-learn and numpy. "
            "Install with: pip install scikit-learn numpy"
        ) from e

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have same length")

    if len(predictions) < 2:
        return PlattCalibrator(params=PlattParameters(A=1.0, B=0.0))

    # Convert to logits
    eps = 1e-7
    probs = np.clip(predictions, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))

    X = logits.reshape(-1, 1)
    y = np.array(labels)

    # Fit logistic regression
    lr = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    lr.fit(X, y)

    # Extract parameters: sigmoid(A * logit + B)
    A = float(lr.coef_[0, 0])
    B = float(lr.intercept_[0])

    return PlattCalibrator(params=PlattParameters(A=A, B=B))


def calibrate_probabilities(
    probabilities: Sequence[float],
    calibrator: IsotonicCalibrator | PlattCalibrator,
) -> CalibrationResult:
    """Apply a fitted calibrator to a sequence of probabilities.

    Args:
        probabilities: Raw predicted probabilities
        calibrator: Fitted calibrator (isotonic or Platt)

    Returns:
        CalibrationResult with original and calibrated probabilities
    """
    calibrated = [calibrator.calibrate(p) for p in probabilities]

    method = "isotonic" if isinstance(calibrator, IsotonicCalibrator) else "platt"

    return CalibrationResult(
        original_probs=list(probabilities),
        calibrated_probs=calibrated,
        calibration_method=method,
    )


@dataclass
class NLICalibrationConfig:
    """Configuration for NLI probability calibration."""

    method: str = "isotonic"  # "isotonic" or "platt"
    support_calibrator: Optional[IsotonicCalibrator | PlattCalibrator] = None
    contradict_calibrator: Optional[IsotonicCalibrator | PlattCalibrator] = None


def calibrate_nli_result(
    p_support: float,
    p_contradict: float,
    config: NLICalibrationConfig,
) -> tuple[float, float]:
    """Calibrate NLI support and contradiction probabilities.

    Args:
        p_support: Raw support probability
        p_contradict: Raw contradiction probability
        config: Calibration configuration with fitted calibrators

    Returns:
        Tuple of (calibrated_support, calibrated_contradict)
    """
    cal_support = p_support
    cal_contradict = p_contradict

    if config.support_calibrator is not None:
        cal_support = config.support_calibrator.calibrate(p_support)

    if config.contradict_calibrator is not None:
        cal_contradict = config.contradict_calibrator.calibrate(p_contradict)

    return cal_support, cal_contradict


def compute_calibration_error(
    predictions: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute calibration error metrics.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth binary labels
        n_bins: Number of bins for ECE/MCE computation

    Returns:
        Dictionary with calibration metrics:
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
        - mean_confidence: Average predicted probability
        - mean_accuracy: Overall accuracy
    """
    if len(predictions) != len(labels) or len(predictions) == 0:
        return {"ece": 0.0, "mce": 0.0, "mean_confidence": 0.0, "mean_accuracy": 0.0}

    # Bin predictions
    bin_counts = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins

    for pred, label in zip(predictions, labels):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bin_counts[bin_idx] += 1
        bin_correct[bin_idx] += label
        bin_conf_sum[bin_idx] += pred

    # Compute ECE and MCE
    ece = 0.0
    mce = 0.0
    n = len(predictions)

    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_acc = bin_correct[i] / bin_counts[i]
            bin_conf = bin_conf_sum[i] / bin_counts[i]
            calibration_gap = abs(bin_acc - bin_conf)

            ece += (bin_counts[i] / n) * calibration_gap
            mce = max(mce, calibration_gap)

    mean_confidence = sum(predictions) / len(predictions)
    mean_accuracy = sum(labels) / len(labels)

    return {
        "ece": ece,
        "mce": mce,
        "mean_confidence": mean_confidence,
        "mean_accuracy": mean_accuracy,
    }
