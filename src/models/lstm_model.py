"""
lstm_model.py
-------------
LSTM (Long Short-Term Memory) neural network for time-series forecasting.

LSTMs capture:
- Long-range temporal dependencies that lag features might miss
- Non-linear sequential patterns
- Cross-timestep interactions (e.g., build-up before events)

Architecture:
    Input sequence (T=28 days) → LSTM layers → Linear head → Forecast (H=28 days)

We use a sequence-to-point architecture: predict the next H steps from
a window of T historical steps. Items are batched for GPU efficiency.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class SalesDataset(Dataset):
    """
    Sliding-window dataset for LSTM training.

    Each sample:
        X : (seq_len, n_features) — historical window
        y : (horizon,)            — future sales values
    """

    def __init__(
        self,
        sales_series: np.ndarray,
        seq_len: int = 28,
        horizon: int = 28,
    ) -> None:
        self.seq_len = seq_len
        self.horizon = horizon

        self.X: list[np.ndarray] = []
        self.y: list[np.ndarray] = []

        for i in range(len(sales_series) - seq_len - horizon + 1):
            self.X.append(sales_series[i : i + seq_len])
            self.y.append(sales_series[i + seq_len : i + seq_len + horizon])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        y = torch.tensor(self.y[idx], dtype=torch.float32)                 # (horizon,)
        return x, y


# ── Model Architecture ────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    """
    Stacked LSTM network with dropout regularization.

    Args:
        input_size  : number of input features (1 for univariate)
        hidden_size : LSTM hidden state dimension
        num_layers  : number of stacked LSTM layers
        dropout     : dropout probability between layers
        horizon     : output forecast horizon
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 28,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last timestep's hidden state
        return self.head(last_hidden)     # (batch, horizon)


# ── Trainer ───────────────────────────────────────────────────────────────────

class LSTMForecaster:
    """
    LSTM forecasting wrapper compatible with walk-forward backtesting.

    Normalizes sales per-item (zero mean, unit variance) before training,
    then denormalizes predictions. This is critical — LSTM is sensitive
    to input scale, and item sales vary wildly (0-100+ units/day).

    Args:
        hidden_size  : LSTM hidden dimension
        num_layers   : number of stacked LSTM layers
        dropout      : dropout regularization
        seq_len      : input sequence length
        horizon      : forecast horizon
        epochs       : training epochs
        batch_size   : mini-batch size
        lr           : learning rate
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 28,
        horizon: int = 28,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self._net: Optional[LSTMNet] = None
        self._item_stats: dict[str, tuple[float, float]] = {}  # (mean, std) per item

    def _normalize(self, series: np.ndarray, item_id: str, fit: bool = True) -> np.ndarray:
        if fit:
            mean, std = series.mean(), series.std() + 1e-8
            self._item_stats[item_id] = (mean, std)
        else:
            mean, std = self._item_stats.get(item_id, (0.0, 1.0))
        return (series - mean) / std

    def _denormalize(self, preds: np.ndarray, item_id: str) -> np.ndarray:
        mean, std = self._item_stats.get(item_id, (0.0, 1.0))
        return preds * std + mean

    def fit(self, train_df: pd.DataFrame) -> "LSTMForecaster":
        """Aggregate all items and train one global LSTM."""
        self._net = LSTMNet(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.HuberLoss()  # Robust to outliers from promotional spikes

        all_X, all_y = [], []
        for item_id, group in train_df.groupby("id"):
            sales = group.sort_values("date")["sales"].values.astype(np.float32)
            if len(sales) < self.seq_len + self.horizon:
                continue
            norm_sales = self._normalize(sales, item_id, fit=True)
            ds = SalesDataset(norm_sales, seq_len=self.seq_len, horizon=self.horizon)
            for x, y in ds:
                all_X.append(x)
                all_y.append(y)

        if not all_X:
            logger.warning("No training samples for LSTM.")
            return self

        loader = DataLoader(
            list(zip(all_X, all_y)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self._net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = self._net(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss/len(loader):.4f}")

        return self

    def predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        """Generate forecasts using the last seq_len days of train as context."""
        if self._net is None:
            raise RuntimeError("Call fit() first.")
        self._net.eval()
        all_preds = []

        for item_id in test_df["id"].unique():
            item_train = train_df[train_df["id"] == item_id].sort_values("date")["sales"].values
            item_test = test_df[test_df["id"] == item_id]
            n_steps = len(item_test)

            if len(item_train) < self.seq_len or item_id not in self._item_stats:
                fallback = np.full(n_steps, item_train.mean() if len(item_train) else 0.0)
                all_preds.append(pd.Series(fallback, index=item_test.index))
                continue

            mean, std = self._item_stats.get(item_id, (0.0, 1.0))
            context = (item_train[-self.seq_len:] - mean) / (std + 1e-8)
            x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

            with torch.no_grad():
                pred_norm = self._net(x).squeeze(0).cpu().numpy()

            preds = pred_norm * std + mean
            preds = np.maximum(0, preds[:n_steps])

            if len(preds) < n_steps:
                preds = np.pad(preds, (0, n_steps - len(preds)), constant_values=preds[-1])

            all_preds.append(pd.Series(preds, index=item_test.index))

        return pd.concat(all_preds).reindex(test_df.index).fillna(0).values

    def __call__(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        self.fit(train_df)
        return self.predict(train_df, test_df)
