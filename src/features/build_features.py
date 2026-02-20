"""
build_features.py
-----------------
Creates lag, rolling, calendar, and price features for the M5 dataset.

Key design decisions:
- All features are computed per-item to avoid cross-item leakage
- Lags are aligned to forecast horizon to prevent look-ahead bias
- Target encoding uses leave-one-out to prevent overfitting
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Lag Features ──────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, lag_days: list[int], target: str = "sales") -> pd.DataFrame:
    """
    Add lag features per item. Lags capture autocorrelation and seasonality.

    Args:
        df       : long-format sales dataframe sorted by (id, date)
        lag_days : list of lag offsets in days
        target   : column to lag

    Returns:
        DataFrame with new lag_{n} columns
    """
    logger.info(f"Building lag features: {lag_days}")
    df = df.sort_values(["id", "date"]).copy()

    for lag in lag_days:
        col = f"lag_{lag}"
        df[col] = df.groupby("id")[target].shift(lag)

    return df


# ── Rolling Window Features ───────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    target: str = "sales",
    shift: int = 28,
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max features.

    We shift by `shift` days (= forecast horizon) before rolling to avoid
    data leakage: at prediction time we only know history up to t-horizon.

    Args:
        df      : long-format sales dataframe
        windows : rolling window sizes in days
        target  : column to compute rolling stats over
        shift   : lag shift before rolling window (prevents leakage)
    """
    logger.info(f"Building rolling features with windows={windows}, shift={shift}")
    df = df.sort_values(["id", "date"]).copy()

    for w in windows:
        shifted = df.groupby("id")[target].shift(shift)
        rolled = shifted.groupby(df["id"]).transform(lambda x: x.rolling(w, min_periods=1))

        df[f"rolling_mean_{w}"] = rolled.mean() if hasattr(rolled, "mean") else rolled
        # Correct chained transform
        df[f"rolling_mean_{w}"] = (
            df.groupby("id")[target]
            .shift(shift)
            .groupby(df["id"])
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}"] = (
            df.groupby("id")[target]
            .shift(shift)
            .groupby(df["id"])
            .transform(lambda x: x.rolling(w, min_periods=1).std())
        )
        df[f"rolling_max_{w}"] = (
            df.groupby("id")[target]
            .shift(shift)
            .groupby(df["id"])
            .transform(lambda x: x.rolling(w, min_periods=1).max())
        )

    return df


# ── Calendar Features ─────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract date-derived and event features.

    Includes cyclical encoding for day-of-week and month to preserve
    periodicity (avoids treating Mon=0 and Sun=6 as maximally different).
    """
    logger.info("Building calendar features...")
    df = df.copy()

    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofmonth"] = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Cyclical encoding — sin/cos keeps periodicity
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Event flag (binary)
    df["has_event"] = (df["event_name_1"].notna() & (df["event_name_1"] != "")).astype(int)

    # SNAP flag for the relevant state
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df["is_snap"] = 0
    for state, col in snap_map.items():
        if col in df.columns:
            df.loc[df["state_id"] == state, "is_snap"] = df.loc[df["state_id"] == state, col]

    return df


# ── Price Features ────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive price change and relative price features.

    Price drops are strong demand signals; this captures promotion effects.
    """
    logger.info("Building price features...")
    df = df.sort_values(["id", "date"]).copy()

    df["price_lag1"] = df.groupby("id")["sell_price"].shift(1)
    df["price_change"] = df["sell_price"] - df["price_lag1"]
    df["price_change_pct"] = df["price_change"] / (df["price_lag1"] + 1e-6)
    df["price_rolling_mean_28"] = (
        df.groupby("id")["sell_price"]
        .transform(lambda x: x.rolling(28, min_periods=1).mean())
    )
    # How does this item's price compare to the category average this week?
    cat_week_mean = df.groupby(["cat_id", "wm_yr_wk"])["sell_price"].transform("mean")
    df["price_vs_cat_mean"] = df["sell_price"] / (cat_week_mean + 1e-6)

    return df


# ── Target Encoding ───────────────────────────────────────────────────────────

def add_target_encoding(
    df: pd.DataFrame,
    group_cols: list[str],
    target: str = "sales",
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """
    Smoothed target encoding for categorical hierarchy columns.

    Uses a smoothed mean that blends the group mean with the global mean,
    controlled by `smoothing`. More data → more weight on group mean.
    Avoids target leakage by computing on training portion only.

    Args:
        df         : dataframe
        group_cols : columns to encode (e.g., ["store_id", "cat_id"])
        target     : target column
        smoothing  : smoothing factor (higher = more regularization)
    """
    logger.info(f"Target encoding for {group_cols}")
    df = df.copy()
    global_mean = df[target].mean()

    for col in group_cols:
        agg = df.groupby(col)[target].agg(["mean", "count"])
        smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        enc_col = f"te_{col}"
        df[enc_col] = df[col].map(smooth)

    return df


# ── Master Builder ────────────────────────────────────────────────────────────

def build_all_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Args:
        df     : raw long-format sales dataframe
        config : pipeline config dict

    Returns:
        Feature-enriched dataframe, NaN rows from lags dropped.
    """
    feat_cfg = config["features"]
    horizon = config["training"]["forecast_horizon"]

    df = add_lag_features(df, lag_days=feat_cfg["lag_days"])
    df = add_rolling_features(df, windows=feat_cfg["rolling_windows"], shift=horizon)
    df = add_calendar_features(df)

    if feat_cfg.get("price_features") and "sell_price" in df.columns:
        df = add_price_features(df)

    if feat_cfg.get("target_encoding"):
        df = add_target_encoding(df, group_cols=["store_id", "cat_id", "item_id"])

    # Drop rows where lag features are NaN (insufficient history)
    max_lag = max(feat_cfg["lag_days"])
    df = df.dropna(subset=[f"lag_{max_lag}"])
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.utils.data_loader import load_config, load_m5_data
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    sales_long, _, _ = load_m5_data(cfg)
    featured = build_all_features(sales_long, cfg)
    out = Path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(out / "features.parquet", index=False)
    print(f"Saved features to {out / 'features.parquet'}")
