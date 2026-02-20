"""ensemble_model.py — Weighted ensemble of multiple forecasters."""
from __future__ import annotations
import logging, numpy as np, pandas as pd
logger = logging.getLogger(__name__)

class EnsembleForecaster:
    def __init__(self, models: dict):
        self.models = models
        total = sum(w for _,w in models.values())
        if abs(total-1.0)>1e-3:
            self.models = {n:(f,w/total) for n,(f,w) in models.items()}

    def __call__(self, train_df, test_df):
        pred = np.zeros(len(test_df))
        for name,(fn,w) in self.models.items():
            try:
                pred += w * fn(train_df, test_df)
            except Exception as e:
                logger.error(f"{name} failed: {e}")
        return np.maximum(0, pred)
