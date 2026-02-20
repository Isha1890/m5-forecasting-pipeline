"""
pipeline.py
-----------
Master pipeline: data → features → training → walk-forward backtesting → leaderboard.

Includes:
  - ARIMA, Prophet, LightGBM, LSTM (trained models)
  - Chronos (zero-shot foundation model — no training)
  - Conformal prediction intervals on best model
  - Model leaderboard with coverage metrics

Usage:
    python src/pipeline.py
    python src/pipeline.py --models lgbm chronos
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import load_config, load_m5_data
from src.features.build_features import build_all_features
from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.lgbm_model import LGBMForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.chronos_model import ChronosForecaster
from src.models.ensemble_model import EnsembleForecaster
from src.evaluation.backtest import walk_forward_backtest, compare_models
from src.evaluation.conformal import ConformalForecaster, compute_empirical_coverage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_models(config: dict, enabled: list[str] | None = None) -> dict:
    mc = config["models"]
    horizon = config["training"]["forecast_horizon"]

    def on(name):
        if enabled:
            return name in enabled
        return mc.get(name, {}).get("enabled", False)

    models = {}
    if on("arima"):
        cfg = mc["arima"]
        models["ARIMA"] = ARIMAForecaster(
            order=tuple(cfg["order"]),
            seasonal_order=tuple(cfg["seasonal_order"]),
            horizon=horizon,
        )
    if on("prophet"):
        cfg = mc["prophet"]
        models["Prophet"] = ProphetForecaster(
            yearly_seasonality=cfg["yearly_seasonality"],
            weekly_seasonality=cfg["weekly_seasonality"],
            changepoint_prior_scale=cfg["changepoint_prior_scale"],
            horizon=horizon,
        )
    if on("lgbm"):
        cfg = mc["lgbm"]
        models["LightGBM"] = LGBMForecaster(
            params={
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
            },
            horizon=horizon,
        )
    if on("lstm"):
        cfg = mc["lstm"]
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
    if on("chronos"):
        cfg = mc["chronos"]
        models["Chronos"] = ChronosForecaster(
            model_id=cfg["model_id"],
            horizon=horizon,
            n_samples=cfg["n_samples"],
            device=cfg["device"],
        )
    return models


def run_pipeline(config: dict, enabled_models: list[str] | None = None) -> None:
    train_cfg = config["training"]
    out_dir = Path(config["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/5 — Loading M5 data")
    logger.info("=" * 60)
    sales_long, _, _ = load_m5_data(config)

    # ── 2. Features ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/5 — Feature engineering")
    logger.info("=" * 60)
    featured_df = build_all_features(sales_long, config)

    # ── 3. Backtest all models ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3/5 — Walk-forward backtesting")
    logger.info("=" * 60)
    models = build_models(config, enabled=enabled_models)
    if not models:
        logger.error("No models enabled.")
        return

    backtest_results = []
    for name, model in models.items():
        logger.info(f"\n▶ {name}")
        result = walk_forward_backtest(
            df=featured_df,
            model_fn=model,
            model_name=name,
            horizon=train_cfg["forecast_horizon"],
            n_splits=train_cfg["backtest_windows"],
            min_train_days=train_cfg["min_train_days"],
        )
        backtest_results.append(result)
        if result.predictions:
            pd.concat(result.predictions).to_parquet(
                out_dir / f"{name.lower()}_predictions.parquet", index=False
            )

    # ── 4. Conformal intervals on best model ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4/5 — Conformal prediction intervals")
    logger.info("=" * 60)
    # Use LightGBM if available — it has the most accurate point forecasts
    best_model_name = "LightGBM" if "LightGBM" in models else list(models.keys())[0]
    best_model = models[best_model_name]
    target_coverage = config["evaluation"].get("conformal_coverage", 0.95)

    # Use the first fold for calibration
    all_dates = sorted(featured_df["date"].unique())
    min_days = train_cfg["min_train_days"]
    cal_cutoff = all_dates[min_days]
    cal_end = pd.Timestamp(cal_cutoff) + pd.Timedelta(days=train_cfg["forecast_horizon"])

    cal_train = featured_df[featured_df["date"] <= cal_cutoff]
    cal_test = featured_df[
        (featured_df["date"] > cal_cutoff) & (featured_df["date"] <= cal_end)
    ]

    if not cal_test.empty:
        conformal = ConformalForecaster(best_model, coverage=target_coverage)
        conformal.calibrate(cal_train, cal_test)
        lower, point, upper = conformal.predict_with_intervals(cal_train, cal_test)
        coverage_metrics = conformal.evaluate_coverage(
            cal_test["sales"].values, lower, upper
        )
        logger.info(f"Conformal coverage ({best_model_name}): {coverage_metrics}")

        # Save intervals
        ci_df = cal_test[["id", "date", "sales"]].copy()
        ci_df["point"] = point
        ci_df["lower"] = lower
        ci_df["upper"] = upper
        ci_df.to_parquet(out_dir / "conformal_intervals.parquet", index=False)

    # ── 5. Leaderboard ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5/5 — Leaderboard")
    logger.info("=" * 60)
    leaderboard = compare_models(backtest_results)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)

    print("\n" + "=" * 60)
    print("🏆  MODEL LEADERBOARD")
    print("=" * 60)
    print(leaderboard.to_string())
    print(f"\n✅ Outputs → {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="M5 Forecasting Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="+",
                        help="Subset of models to run: arima prophet lgbm lstm chronos")
    args = parser.parse_args()
    run_pipeline(load_config(args.config), enabled_models=args.models)


if __name__ == "__main__":
    main()
