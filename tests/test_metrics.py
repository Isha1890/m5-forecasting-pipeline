"""
test_metrics.py
---------------
Unit tests for the metrics and backtesting modules.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import mae, rmse, mape, smape, compute_all_metrics
from src.evaluation.backtest import walk_forward_backtest, compare_models, BacktestResult


# ── Metric Tests ──────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_basic(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_rmse_perfect(self):
        y = np.array([5.0, 10.0, 15.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_basic(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])
        assert rmse(y_true, y_pred) == pytest.approx(2.0)

    def test_mape_no_zero_division(self):
        """MAPE should not divide by zero when y_true = 0."""
        y_true = np.zeros(5)
        y_pred = np.ones(5)
        result = mape(y_true, y_pred)
        assert np.isfinite(result)

    def test_mape_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_smape_symmetric(self):
        """SMAPE should be symmetric: smape(a, b) == smape(b, a)."""
        a = np.array([1.0, 2.0, 5.0])
        b = np.array([2.0, 1.0, 3.0])
        assert smape(a, b) == pytest.approx(smape(b, a), rel=1e-5)

    def test_compute_all_metrics_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_all_metrics(y, y)
        assert set(result.keys()) == {"mae", "rmse", "mape", "smape"}

    def test_compute_all_metrics_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_all_metrics(y, y)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mape"] == pytest.approx(0.0)

    def test_metrics_non_negative(self):
        """All metrics should return non-negative values."""
        y_true = np.random.rand(100) * 10
        y_pred = np.random.rand(100) * 10
        result = compute_all_metrics(y_true, y_pred)
        for k, v in result.items():
            assert v >= 0, f"{k} should be non-negative, got {v}"


# ── Backtesting Tests ─────────────────────────────────────────────────────────

def make_fake_df(n_items: int = 3, n_days: int = 400) -> pd.DataFrame:
    """Generate a minimal long-format sales DataFrame for testing."""
    rows = []
    base = pd.Timestamp("2022-01-01")
    for item in range(n_items):
        for day in range(n_days):
            rows.append({
                "id": f"item_{item}",
                "date": base + pd.Timedelta(days=day),
                "sales": float(np.random.poisson(5)),
                "item_id": f"item_{item}",
                "store_id": "CA_1",
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def naive_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Seasonal naive: use last week's sales for each test row."""
    mean_sales = train_df["sales"].mean()
    return np.full(len(test_df), mean_sales)


class TestBacktest:
    def test_backtest_returns_result(self):
        df = make_fake_df(n_items=2, n_days=400)
        result = walk_forward_backtest(
            df=df,
            model_fn=naive_model,
            model_name="Naive",
            horizon=28,
            n_splits=2,
            min_train_days=90,
            verbose=False,
        )
        assert isinstance(result, BacktestResult)
        assert len(result.fold_results) > 0

    def test_backtest_metrics_present(self):
        df = make_fake_df(n_items=2, n_days=400)
        result = walk_forward_backtest(
            df=df,
            model_fn=naive_model,
            model_name="Naive",
            horizon=28,
            n_splits=2,
            min_train_days=90,
            verbose=False,
        )
        for fold in result.fold_results:
            assert "mae" in fold
            assert "rmse" in fold
            assert "mape" in fold

    def test_backtest_predictions_aligned(self):
        df = make_fake_df(n_items=2, n_days=400)
        result = walk_forward_backtest(
            df=df,
            model_fn=naive_model,
            model_name="Naive",
            horizon=28,
            n_splits=1,
            min_train_days=90,
            verbose=False,
        )
        for pred_df in result.predictions:
            assert "prediction" in pred_df.columns
            assert "sales" in pred_df.columns
            assert not pred_df["prediction"].isna().any()

    def test_compare_models(self):
        r1 = BacktestResult("ModelA", fold_results=[{"mae": 1.0, "rmse": 1.5, "mape": 10.0, "smape": 9.0}])
        r2 = BacktestResult("ModelB", fold_results=[{"mae": 0.8, "rmse": 1.2, "mape": 8.0, "smape": 7.5}])
        lb = compare_models([r1, r2])
        assert lb.iloc[0]["model"] == "ModelB"  # Lower MAE ranks first
        assert "mae" in lb.columns

    def test_insufficient_data_raises(self):
        df = make_fake_df(n_items=1, n_days=50)  # Too little data
        with pytest.raises(ValueError):
            walk_forward_backtest(
                df=df,
                model_fn=naive_model,
                model_name="Naive",
                horizon=28,
                n_splits=3,
                min_train_days=365,  # Impossible given only 50 days
                verbose=False,
            )
