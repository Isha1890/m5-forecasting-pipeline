"""
data_loader.py — Loads and melts M5 sales data into long format.
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

def load_config(path: str = "configs/default.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent.parent / path
    with open(p) as f:
        return yaml.safe_load(f)

def load_m5_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw M5 CSVs and return (sales_long, calendar, prices)."""
    raw_dir = config["data"]["raw_dir"]
    raw = Path(raw_dir)
    if not raw.is_absolute():
        raw = Path(__file__).resolve().parent.parent.parent / raw_dir

    logger.info("Loading sales data...")
    sales_wide = pd.read_csv(raw / "sales_train_evaluation.csv")

    logger.info("Loading calendar...")
    calendar = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])

    logger.info("Loading prices...")
    prices = pd.read_csv(raw / "sell_prices.csv")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols  = [c for c in sales_wide.columns if c.startswith("d_")]

    logger.info(f"Melting {len(d_cols)} day columns for {len(sales_wide)} items...")
    sales_long = sales_wide.melt(id_vars=id_cols, value_vars=d_cols,
                                  var_name="d", value_name="sales")

    sales_long = sales_long.merge(
        calendar[["d","date","wm_yr_wk","wday","month","year",
                   "event_name_1","event_type_1","snap_CA","snap_TX","snap_WI"]],
        on="d", how="left")
    sales_long = sales_long.merge(prices, on=["store_id","item_id","wm_yr_wk"], how="left")
    sales_long.sort_values(["id","date"], inplace=True)
    sales_long.reset_index(drop=True, inplace=True)

    logger.info(f"Long-format shape: {sales_long.shape}")
    return sales_long, calendar, prices
