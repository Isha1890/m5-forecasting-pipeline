"""arima_model.py — Seasonal ARIMA wrapper for backtesting."""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

class ARIMAForecaster:
    def __init__(self, order=(2,1,2), seasonal_order=(1,1,1,7), horizon=28):
        self.order=order; self.seasonal_order=seasonal_order; self.horizon=horizon
        self._models={}

    def __call__(self, train_df, test_df):
        all_preds=[]
        for item_id in test_df["id"].unique():
            s = train_df[train_df["id"]==item_id].set_index("date")["sales"]
            n = len(test_df[test_df["id"]==item_id])
            try:
                fit = SARIMAX(s, order=self.order, seasonal_order=self.seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                p = np.maximum(0, fit.forecast(n).values)
            except Exception:
                p = np.full(n, s.mean())
            idx = test_df[test_df["id"]==item_id].index
            all_preds.append(pd.Series(p[:n], index=idx))
        return pd.concat(all_preds).reindex(test_df.index).fillna(0).values
