"""
backtest.py
-----------
Walk-forward (expanding-window) backtesting engine.

Why walk-forward?
-----------------
Simple train/test splits leak temporal information — the model sees the future.
Walk-forward validation mirrors real deployment: train on all data up to time t,
predict t+1 ... t+h, slide forward, repeat. This gives unbiased estimates of
how the model would actually perform in production.

                  Window 1        Window 2        Window 3
Train:   [======]                [=========]     [============]
Predict:         [----h----]              [----h----]       [----h----]
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics, metrics_dataframe

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Stores per-fold and aggregate backtesting results."""
    model_name: str
    fold_results: list[dict] = field(default_factory=list)
    predictions: list[pd.DataFrame] = field(default_factory=list)

    @property
    def summary(self) -> pd.DataFrame:
        return metrics_dataframe(self.fold_results)

    @property
    def mean_mae(self) -> float:
        return float(np.mean([r["mae"] for r in self.fold_results]))

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([r["rmse"] for r in self.fold_results]))

    @property
    def mean_mape(self) -> float:
        return float(np.mean([r["mape"] for r in self.fold_results]))

    def __repr__(self) -> str:
        return (
            f"BacktestResult(model={self.model_name}, "
            f"folds={len(self.fold_results)}, "
            f"MAE={self.mean_mae:.3f}, RMSE={self.mean_rmse:.3f}, MAPE={self.mean_mape:.1f}%)"
        )


def walk_forward_backtest(
    df: pd.DataFrame,
    model_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    model_name: str,
    date_col: str = "date",
    target_col: str = "sales",
    id_col: str = "id",
    horizon: int = 28,
    n_splits: int = 3,
    min_train_days: int = 365,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run walk-forward cross-validation.

    Args:
        df           : long-format dataframe sorted by date
        model_fn     : callable(train_df, test_df) → np.ndarray of predictions
                       Must return predictions aligned to test_df rows.
        model_name   : label for reporting
        date_col     : name of date column
        target_col   : name of target column
        id_col       : item identifier column
        horizon      : forecast horizon in days
        n_splits     : number of walk-forward folds
        min_train_days: minimum history required before first fold
        verbose      : log fold progress

    Returns:
        BacktestResult with per-fold metrics and predictions
    """
    df = df.sort_values([id_col, date_col]).copy()
    all_dates = sorted(df[date_col].unique())
    result = BacktestResult(model_name=model_name)

    # Build fold cutoff dates
    # Last fold ends `horizon` days before the final date
    total_usable = len(all_dates) - horizon
    fold_size = max(1, total_usable // n_splits)

    cutoffs = []
    for i in range(n_splits):
        idx = min_train_days + (i + 1) * fold_size - 1
        if idx < len(all_dates) - horizon:
            cutoffs.append(all_dates[idx])

    if not cutoffs:
        raise ValueError(
            f"Not enough data for {n_splits} folds with min_train_days={min_train_days}. "
            f"Total dates: {len(all_dates)}"
        )

    for fold_num, cutoff in enumerate(cutoffs, 1):
        train_df = df[df[date_col] <= cutoff].copy()
        test_end = pd.Timestamp(cutoff) + pd.Timedelta(days=horizon)
        test_df = df[
            (df[date_col] > cutoff) & (df[date_col] <= test_end)
        ].copy()

        if test_df.empty:
            logger.warning(f"Fold {fold_num}: empty test set, skipping.")
            continue

        if verbose:
            logger.info(
                f"[{model_name}] Fold {fold_num}/{len(cutoffs)} | "
                f"Train: up to {cutoff.date()} ({len(train_df):,} rows) | "
                f"Test: {len(test_df):,} rows"
            )

        try:
            preds = model_fn(train_df, test_df)
        except Exception as e:
            logger.error(f"Model failed on fold {fold_num}: {e}")
            continue

        y_true = test_df[target_col].values
        metrics = compute_all_metrics(y_true, preds)
        metrics["fold"] = fold_num
        metrics["cutoff"] = str(cutoff.date())
        result.fold_results.append(metrics)

        pred_df = test_df[[id_col, date_col, target_col]].copy()
        pred_df["prediction"] = preds
        pred_df["model"] = model_name
        pred_df["fold"] = fold_num
        result.predictions.append(pred_df)

        if verbose:
            logger.info(
                f"  → MAE={metrics['mae']:.3f} | "
                f"RMSE={metrics['rmse']:.3f} | "
                f"MAPE={metrics['mape']:.1f}%"
            )

    return result


def compare_models(results: list[BacktestResult]) -> pd.DataFrame:
    """
    Build a leaderboard DataFrame comparing multiple BacktestResult objects.

    Returns a DataFrame sorted by MAE ascending.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.model_name,
                "folds": len(r.fold_results),
                "mae": round(r.mean_mae, 4),
                "rmse": round(r.mean_rmse, 4),
                "mape": round(r.mean_mape, 2),
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    leaderboard.index += 1  # Rank from 1
    return leaderboard
