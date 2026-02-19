"""
metrics.py
----------
Forecasting evaluation metrics: MAE, RMSE, MAPE, and WRMSSE (M5 official).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.

    Uses epsilon to avoid division by zero on zero-sales days.
    Returns value in [0, 100].
    """
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(100.0 * np.mean(np.abs(y_true - y_pred) / denom))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE — less sensitive to zero actuals than standard MAPE.
    Returns value in [0, 200].
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(100.0 * np.mean(np.abs(y_true - y_pred) / denom))


def wrmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: np.ndarray,
    weight: float = 1.0,
) -> float:
    """
    Weighted Root Mean Squared Scaled Error — the official M5 metric.

    WRMSSE = weight * sqrt(mean((y_true - y_pred)^2) / scale)

    Args:
        y_true  : actual values
        y_pred  : predicted values
        scale   : per-series scale (mean squared naive forecast error on training)
        weight  : series weight (based on dollar-value sales volume)
    """
    msse = np.mean((y_true - y_pred) ** 2) / (scale + 1e-8)
    return float(weight * np.sqrt(msse))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all standard metrics and return as a dict."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def metrics_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of per-fold metric dicts into a summary DataFrame
    with mean and std across folds.
    """
    df = pd.DataFrame(results)
    summary = pd.DataFrame({
        "metric": df.columns,
        "mean": df.mean().values,
        "std": df.std().values,
        "min": df.min().values,
        "max": df.max().values,
    })
    return summary
