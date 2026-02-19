"""
lgbm_model.py
-------------
LightGBM global forecasting model — the highest-performing model in this pipeline.

Why LightGBM beats ARIMA/Prophet on M5:
- Learns cross-item patterns (shared knowledge across 3,049 products)
- Handles the engineered lag/rolling features natively
- Captures non-linear feature interactions (price × event × day-of-week)
- Scales to millions of rows efficiently via histogram-based splitting
- Built-in handling of missing values and mixed feature types
- Fast inference — critical for batch prediction over large item catalogues

The model is "global" in the sense that it trains across all items simultaneously,
using item/store/category IDs as features. This is the dominant paradigm in
modern large-scale retail forecasting (see Amazon's DeepAR, N-BEATS, etc.).
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Categorical columns that need label encoding for LightGBM
CATEGORICAL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Feature columns used for training (excludes targets, IDs, raw dates)
EXCLUDE_COLS = {
    "id", "date", "sales", "d", "wm_yr_wk", "weekday",
    "event_name_1", "event_type_1",  # String, replaced by has_event flag
    "snap_CA", "snap_TX", "snap_WI",  # Replaced by is_snap
    "price_lag1",  # Derived, to avoid leakage alongside sell_price
}


class LGBMForecaster:
    """
    LightGBM global demand forecasting model.

    Args:
        params   : LightGBM hyperparameters
        horizon  : forecast horizon in days
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        horizon: int = 28,
    ) -> None:
        self.horizon = horizon
        self.params = params or {
            "objective": "regression_l1",  # MAE loss — robust to outliers (promotions)
            "metric": "mae",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self._model: Optional[lgb.LGBMRegressor] = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._feature_cols: list[str] = []

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Label-encode categorical columns."""
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders.get(col)
                if le:
                    # Handle unseen categories gracefully
                    df[col] = df[col].astype(str).map(
                        lambda x, le=le: le.transform([x])[0]
                        if x in le.classes_
                        else -1
                    )
        return df

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Infer feature columns, excluding non-feature columns."""
        return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype != "object"]

    def fit(self, train_df: pd.DataFrame) -> "LGBMForecaster":
        """
        Fit the global LightGBM model on training data.

        Args:
            train_df : feature-engineered long-format dataframe
        """
        train_enc = self._encode_categoricals(train_df, fit=True)
        self._feature_cols = self._get_feature_cols(train_enc)

        X = train_enc[self._feature_cols]
        y = train_enc["sales"]

        logger.info(
            f"Training LightGBM on {len(X):,} rows × {len(self._feature_cols)} features"
        )

        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(
            X, y,
            callbacks=[lgb.log_evaluation(100)],
        )

        logger.info("LightGBM training complete.")
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for test_df rows.

        Returns:
            np.ndarray of non-negative predictions
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        test_enc = self._encode_categoricals(test_df, fit=False)
        X = test_enc[self._feature_cols]
        preds = self._model.predict(X)
        return np.maximum(0, preds)

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return top-N features by importance (gain-based).

        Useful for interpretability and feature selection.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        importance = self._model.feature_importances_
        df = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False).head(top_n)
        df["importance_pct"] = 100 * df["importance"] / df["importance"].sum()
        return df.reset_index(drop=True)

    def __call__(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        """Backtesting-compatible interface."""
        self.fit(train_df)
        return self.predict(test_df)
