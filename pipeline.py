"""
pipeline.py
-----------
Master pipeline: data loading → feature engineering → training → backtesting → reporting.

Usage:
    python src/pipeline.py --config configs/default.yaml
    python src/pipeline.py --config configs/default.yaml --models lgbm prophet
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_config, load_m5_data
from src.features.build_features import build_all_features
from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.lgbm_model import LGBMForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.ensemble_model import EnsembleForecaster
from src.evaluation.backtest import walk_forward_backtest, compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def build_models(config: dict, enabled: list[str] | None = None) -> dict:
    """Instantiate all enabled models from config."""
    model_cfg = config["models"]
    horizon = config["training"]["forecast_horizon"]
    models = {}

    def is_enabled(name: str) -> bool:
        if enabled is not None:
            return name in enabled
        return model_cfg.get(name, {}).get("enabled", False)

    if is_enabled("arima"):
        cfg = model_cfg["arima"]
        models["ARIMA"] = ARIMAForecaster(
            order=tuple(cfg["order"]),
            seasonal_order=tuple(cfg["seasonal_order"]),
            horizon=horizon,
        )

    if is_enabled("prophet"):
        cfg = model_cfg["prophet"]
        models["Prophet"] = ProphetForecaster(
            yearly_seasonality=cfg["yearly_seasonality"],
            weekly_seasonality=cfg["weekly_seasonality"],
            changepoint_prior_scale=cfg["changepoint_prior_scale"],
            horizon=horizon,
        )

    if is_enabled("lgbm"):
        cfg = model_cfg["lgbm"]
        lgbm_params = {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": cfg["n_estimators"],
            "learning_rate": cfg["learning_rate"],
            "num_leaves": cfg["num_leaves"],
            "min_child_samples": cfg["min_child_samples"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "reg_alpha": cfg["reg_alpha"],
            "reg_lambda": cfg["reg_lambda"],
            "random_state": cfg["random_state"],
            "n_jobs": -1,
            "verbose": -1,
        }
        models["LightGBM"] = LGBMForecaster(params=lgbm_params, horizon=horizon)

    if is_enabled("lstm"):
        cfg = model_cfg["lstm"]
        models["LSTM"] = LSTMForecaster(
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            seq_len=cfg["sequence_length"],
            horizon=horizon,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["learning_rate"],
        )

    return models


def run_pipeline(config: dict, enabled_models: list[str] | None = None) -> None:
    """End-to-end pipeline execution."""
    train_cfg = config["training"]
    out_metrics = Path(config["evaluation"]["output_dir"])
    out_metrics.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/4 — Loading data")
    logger.info("=" * 60)
    sales_long, calendar, prices = load_m5_data(config)

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/4 — Feature engineering")
    logger.info("=" * 60)
    featured_df = build_all_features(sales_long, config)

    # ── 3. Run Backtests ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3/4 — Walk-forward backtesting")
    logger.info("=" * 60)
    models = build_models(config, enabled=enabled_models)

    if not models:
        logger.error("No models enabled! Check configs/default.yaml.")
        return

    backtest_results = []
    for model_name, model in models.items():
        logger.info(f"\n▶ Backtesting: {model_name}")
        result = walk_forward_backtest(
            df=featured_df,
            model_fn=model,
            model_name=model_name,
            horizon=train_cfg["forecast_horizon"],
            n_splits=train_cfg["backtest_windows"],
            min_train_days=train_cfg["min_train_days"],
        )
        backtest_results.append(result)

        # Save per-model predictions
        if result.predictions:
            pred_df = pd.concat(result.predictions)
            pred_df.to_parquet(out_metrics / f"{model_name.lower()}_predictions.parquet", index=False)

    # ── 4. Leaderboard ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4/4 — Model comparison")
    logger.info("=" * 60)
    leaderboard = compare_models(backtest_results)
    leaderboard.to_csv(out_metrics / "leaderboard.csv", index=False)

    print("\n" + "=" * 60)
    print("🏆  MODEL LEADERBOARD (sorted by MAE)")
    print("=" * 60)
    print(leaderboard.to_string())
    print("=" * 60)
    print(f"\n✅ Results saved to {out_metrics}/")


def main():
    parser = argparse.ArgumentParser(description="M5 Forecasting Pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--models", nargs="+", help="Models to run (e.g. lgbm prophet)")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, enabled_models=args.models)


if __name__ == "__main__":
    main()
