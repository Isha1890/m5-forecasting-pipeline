"""
chronos_model.py
----------------
Amazon Chronos — a pretrained T5-based foundation model for time-series forecasting.

Released in 2024, Chronos was pretrained on 27B time-series tokens from 84 datasets.
It can forecast any time series ZERO-SHOT — no fine-tuning, no feature engineering.

Why this matters for 2026:
- Demonstrates awareness of the foundation model shift in the field
- Zero-shot performance on M5 approaches fine-tuned LightGBM
- Shows ability to benchmark traditional ML vs. modern pretrained approaches

Paper: "Chronos: Learning the Language of Time Series" (Ansari et al., 2024)
GitHub: https://github.com/amazon-science/chronos-forecasting

Install:
    pip install chronos-forecasting
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Use tiny model by default for speed; swap to "amazon/chronos-t5-small" for better accuracy
CHRONOS_MODEL_ID = "amazon/chronos-t5-tiny"


class ChronosForecaster:
    """
    Zero-shot forecaster using Amazon's Chronos foundation model.

    No training required. Feed in a raw sales series and get probabilistic
    forecasts out — including built-in quantile uncertainty estimates.

    Args:
        model_id : HuggingFace model ID for Chronos variant
                   Options: chronos-t5-tiny | small | base | large
        horizon  : forecast horizon in days
        n_samples: number of sample paths for uncertainty quantification
        device   : 'cpu', 'cuda', or 'mps' (Apple Silicon)
    """

    def __init__(
        self,
        model_id: str = CHRONOS_MODEL_ID,
        horizon: int = 28,
        n_samples: int = 20,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.horizon = horizon
        self.n_samples = n_samples
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load Chronos pipeline (downloads ~300MB on first run)."""
        if self._pipeline is not None:
            return
        try:
            from chronos import ChronosPipeline
            logger.info(f"Loading Chronos model: {self.model_id}")
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
            logger.info("Chronos loaded successfully.")
        except ImportError:
            raise ImportError(
                "Install Chronos: pip install chronos-forecasting\n"
                "Docs: https://github.com/amazon-science/chronos-forecasting"
            )

    def forecast_series(
        self,
        series: np.ndarray,
        quantiles: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Forecast a single time series zero-shot.

        Args:
            series    : historical sales values (1D numpy array)
            quantiles : quantiles to return for uncertainty (default: [0.1, 0.5, 0.9])

        Returns:
            dict with keys 'mean', 'median', 'q10', 'q90' — each shape (horizon,)
        """
        self._load_pipeline()
        quantiles = quantiles or [0.1, 0.5, 0.9]

        context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)  # (1, T)

        with torch.no_grad():
            forecast = self._pipeline.predict(
                context=context,
                prediction_length=self.horizon,
                num_samples=self.n_samples,
            )
        # forecast shape: (1, n_samples, horizon)
        samples = forecast[0].numpy()  # (n_samples, horizon)

        return {
            "mean": samples.mean(axis=0),
            "median": np.quantile(samples, 0.5, axis=0),
            "q10": np.quantile(samples, 0.1, axis=0),
            "q90": np.quantile(samples, 0.9, axis=0),
            "samples": samples,
        }

    def __call__(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        """
        Backtesting-compatible interface.

        Chronos forecasts each item independently using only its raw sales history.
        No feature engineering — this is the whole point of foundation models.
        """
        self._load_pipeline()
        all_preds = []

        for item_id in test_df["id"].unique():
            item_train = (
                train_df[train_df["id"] == item_id]
                .sort_values("date")["sales"]
                .values.astype(np.float32)
            )
            item_test = test_df[test_df["id"] == item_id]
            n_steps = len(item_test)

            if len(item_train) < 14:
                # Insufficient history — fall back to mean
                preds = np.full(n_steps, item_train.mean() if len(item_train) else 0.0)
            else:
                try:
                    fc = self.forecast_series(item_train)
                    preds = np.maximum(0, fc["mean"][:n_steps])
                    if len(preds) < n_steps:
                        preds = np.pad(preds, (0, n_steps - len(preds)),
                                       constant_values=preds[-1] if len(preds) else 0)
                except Exception as e:
                    logger.warning(f"Chronos failed for {item_id}: {e}. Using mean.")
                    preds = np.full(n_steps, item_train.mean())

            all_preds.append(pd.Series(preds, index=item_test.index))

        return pd.concat(all_preds).reindex(test_df.index).fillna(0).values
