"""prophet_model.py — Prophet wrapper for backtesting."""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

class ProphetForecaster:
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True,
                 changepoint_prior_scale=0.05, horizon=28):
        self.yearly=yearly_seasonality; self.weekly=weekly_seasonality
        self.cps=changepoint_prior_scale; self.horizon=horizon

    def __call__(self, train_df, test_df):
        from prophet import Prophet
        all_preds=[]
        for item_id in test_df["id"].unique():
            s = train_df[train_df["id"]==item_id][["date","sales"]].rename(columns={"date":"ds","sales":"y"})
            n = len(test_df[test_df["id"]==item_id])
            try:
                m = Prophet(yearly_seasonality=self.yearly, weekly_seasonality=self.weekly,
                            daily_seasonality=False, changepoint_prior_scale=self.cps)
                m.fit(s)
                fc = m.predict(m.make_future_dataframe(n))
                p  = np.maximum(0, fc["yhat"].tail(n).values)
            except Exception:
                p = np.full(n, s["y"].mean())
            idx = test_df[test_df["id"]==item_id].index
            all_preds.append(pd.Series(p[:n], index=idx))
        return pd.concat(all_preds).reindex(test_df.index).fillna(0).values
