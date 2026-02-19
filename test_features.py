"""
test_features.py
----------------
Unit tests for feature engineering module.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import (
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
    add_target_encoding,
)


def make_sales_df(n_items=2, n_days=100):
    rows = []
    base = pd.Timestamp("2023-01-01")
    for item in range(n_items):
        for day in range(n_days):
            rows.append({
                "id": f"item_{item}",
                "date": base + pd.Timedelta(days=day),
                "sales": float(np.random.poisson(5)),
                "state_id": "CA",
                "cat_id": "FOODS",
                "item_id": f"item_{item}",
                "store_id": "CA_1",
                "month": (base + pd.Timedelta(days=day)).month,
                "event_name_1": "",
                "snap_CA": 1 if day % 10 < 3 else 0,
                "snap_TX": 0,
                "snap_WI": 0,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestLagFeatures:
    def test_lag_columns_created(self):
        df = make_sales_df()
        result = add_lag_features(df, lag_days=[7, 14])
        assert "lag_7" in result.columns
        assert "lag_14" in result.columns

    def test_lag_values_correct(self):
        """For a single item, lag_1 of row i should equal sales of row i-1."""
        df = make_sales_df(n_items=1, n_days=50)
        result = add_lag_features(df.sort_values("date"), lag_days=[1])
        result = result.sort_values("date").reset_index(drop=True)
        # Row 1 lag_1 should equal row 0's sales
        assert result.loc[1, "lag_1"] == result.loc[0, "sales"]

    def test_no_cross_item_leakage(self):
        """Lag features should not bleed from item_0 into item_1."""
        df = make_sales_df(n_items=2, n_days=50)
        result = add_lag_features(df, lag_days=[1])
        for item_id, group in result.groupby("id"):
            group = group.sort_values("date")
            # First row of each item should have NaN lag (no prior history)
            assert pd.isna(group["lag_1"].iloc[0])


class TestRollingFeatures:
    def test_rolling_columns_created(self):
        df = make_sales_df()
        result = add_rolling_features(df, windows=[7, 28])
        assert "rolling_mean_7" in result.columns
        assert "rolling_mean_28" in result.columns

    def test_rolling_non_negative(self):
        df = make_sales_df()
        result = add_rolling_features(df, windows=[7])
        # Rolling mean of non-negative sales should be non-negative
        valid = result["rolling_mean_7"].dropna()
        assert (valid >= 0).all()


class TestCalendarFeatures:
    def test_calendar_columns_exist(self):
        df = make_sales_df()
        result = add_calendar_features(df)
        expected = ["dayofweek", "is_weekend", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_cyclical_encoding_bounds(self):
        df = make_sales_df()
        result = add_calendar_features(df)
        # sin/cos values should be in [-1, 1]
        assert result["dow_sin"].between(-1, 1).all()
        assert result["dow_cos"].between(-1, 1).all()

    def test_is_weekend_binary(self):
        df = make_sales_df()
        result = add_calendar_features(df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})


class TestTargetEncoding:
    def test_encoding_columns_created(self):
        df = make_sales_df()
        result = add_target_encoding(df, group_cols=["cat_id", "store_id"])
        assert "te_cat_id" in result.columns
        assert "te_store_id" in result.columns

    def test_encoding_values_numeric(self):
        df = make_sales_df()
        result = add_target_encoding(df, group_cols=["cat_id"])
        assert pd.api.types.is_numeric_dtype(result["te_cat_id"])

    def test_encoding_near_global_mean_with_high_smoothing(self):
        """High smoothing should pull encoded values toward global mean."""
        df = make_sales_df()
        global_mean = df["sales"].mean()
        result = add_target_encoding(df, group_cols=["cat_id"], smoothing=1e6)
        # With extreme smoothing, encoded value ≈ global mean
        assert abs(result["te_cat_id"].mean() - global_mean) < 0.5
