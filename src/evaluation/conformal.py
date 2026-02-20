"""
conformal.py
------------
Split conformal prediction for time-series forecasting.

Why conformal prediction over naive ±σ intervals?
--------------------------------------------------
Traditional approaches assume Gaussian residuals and use ±1.96σ. This fails when:
  - Sales distributions are right-skewed (zero-inflated, Poisson-like)
  - Residuals are heteroskedastic (bigger errors on high-selling items)
  - The model is miscalibrated

Conformal prediction is:
  ✅ Distribution-free — no Gaussian assumptions
  ✅ Guaranteed finite-sample coverage — if you ask for 95%, you get ≥95%
  ✅ Model-agnostic — works on top of any forecaster
  ✅ Simple to implement and explain in interviews

Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification" (2022)
https://arxiv.org/abs/2107.07511

Algorithm (Split Conformal):
  1. Split calibration data into cal_train / cal_test
  2. Fit model on cal_train, predict cal_test
  3. Compute nonconformity scores: s_i = |y_i - ŷ_i|
  4. Find quantile q̂ = (1-α) quantile of {s_1, ..., s_n, ∞}
  5. For new prediction ŷ: interval = [ŷ - q̂, ŷ + q̂]
     This guarantees P(y ∈ interval) ≥ 1 - α
"""

from __future__ import annotations
import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConformalForecaster:
    """
    Wraps any point-forecast model with split conformal prediction intervals.

    Usage:
        base_model = LGBMForecaster(...)
        conformal = ConformalForecaster(base_model, coverage=0.95)
        conformal.calibrate(cal_train_df, cal_test_df)
        lower, point, upper = conformal.predict_with_intervals(train_df, test_df)

    Args:
        model_fn : any callable(train_df, test_df) → np.ndarray
        coverage : target coverage level, e.g. 0.95 for 95% intervals
    """

    def __init__(
        self,
        model_fn: Callable,
        coverage: float = 0.95,
    ) -> None:
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self.model_fn = model_fn
        self.coverage = coverage
        self._q_hat: float | None = None  # Calibrated quantile

    def calibrate(
        self,
        cal_train_df: pd.DataFrame,
        cal_test_df: pd.DataFrame,
    ) -> "ConformalForecaster":
        """
        Estimate the conformal quantile from a calibration split.

        Args:
            cal_train_df : training data for calibration
            cal_test_df  : holdout data to compute nonconformity scores
        """
        logger.info("Calibrating conformal intervals...")

        preds = self.model_fn(cal_train_df, cal_test_df)
        actuals = cal_test_df["sales"].values

        # Nonconformity scores: absolute residuals
        scores = np.abs(actuals - preds)

        # Conformal quantile: ceil((n+1)(1-α))/n quantile
        # The +∞ correction ensures finite-sample validity
        n = len(scores)
        level = np.ceil((n + 1) * self.coverage) / n
        level = min(level, 1.0)

        self._q_hat = float(np.quantile(scores, level))
        logger.info(
            f"Conformal quantile q̂ = {self._q_hat:.3f} "
            f"(n={n}, target coverage={self.coverage:.0%})"
        )
        return self

    def predict_with_intervals(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate point forecasts + conformal prediction intervals.

        Returns:
            (lower, point, upper) — each shape (len(test_df),)
            Intervals satisfy: P(y_true ∈ [lower, upper]) ≥ coverage
        """
        if self._q_hat is None:
            raise RuntimeError("Call .calibrate() before .predict_with_intervals()")

        point = self.model_fn(train_df, test_df)
        lower = np.maximum(0, point - self._q_hat)
        upper = point + self._q_hat

        return lower, point, upper

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute empirical coverage and interval width.

        Empirical coverage should be ≥ target coverage if calibration was valid.

        Args:
            y_true : actual values
            lower  : lower bounds
            upper  : upper bounds

        Returns:
            dict with 'coverage', 'mean_width', 'efficiency'
        """
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        width = (upper - lower).mean()

        # Efficiency: smaller interval is better (given valid coverage)
        efficiency = 1.0 / (width + 1e-6)

        result = {
            "empirical_coverage": round(float(covered), 4),
            "target_coverage": self.coverage,
            "coverage_gap": round(float(covered - self.coverage), 4),
            "mean_interval_width": round(float(width), 4),
            "q_hat": round(self._q_hat, 4),
        }

        logger.info(
            f"Coverage: {covered:.1%} (target {self.coverage:.1%}) | "
            f"Width: {width:.2f} | q̂={self._q_hat:.3f}"
        )
        return result


def compute_empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Standalone helper: fraction of actuals within [lower, upper]."""
    return float(((y_true >= lower) & (y_true <= upper)).mean())
