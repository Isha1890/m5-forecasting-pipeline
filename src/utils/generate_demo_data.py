"""
generate_demo_data.py — Generates synthetic M5-style data (no Kaggle needed).
Usage: python src/utils/generate_demo_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_m5_style_data(n_items=50, n_days=730, seed=42):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    stores     = ["CA_1","CA_2","TX_1","TX_2","WI_1"]
    categories = ["FOODS","HOBBIES","HOUSEHOLD"]
    start_date = pd.Timestamp("2020-01-01")
    dates      = pd.date_range(start_date, periods=n_days, freq="D")

    events = {"SuperBowl":"Sporting","ValentinesDay":"Cultural","Easter":"Religious",
              "MemorialDay":"National","Thanksgiving":"National","Christmas":"Religious"}

    # Calendar
    cal_rows = []
    for i, d in enumerate(dates):
        evt, typ = "", ""
        if rng.random() < 0.03:
            k = rng.choice(list(events.keys())); evt, typ = k, events[k]
        cal_rows.append({"date":d,"wm_yr_wk":d.isocalendar().week+d.year*100,
                         "weekday":d.day_name(),"wday":d.weekday(),"month":d.month,
                         "year":d.year,"event_name_1":evt,"event_type_1":typ,
                         "snap_CA":int(d.day<=10),"snap_TX":int(10<d.day<=20),
                         "snap_WI":int(d.day>20),"d":f"d_{i+1}"})
    calendar_df = pd.DataFrame(cal_rows)

    # Sales
    items, sales_matrix = [], []
    for i in range(n_items):
        store = rng.choice(stores); cat = rng.choice(categories)
        item_id = f"{cat}_{i+1:03d}"; state = store.split("_")[0]
        t = np.arange(n_days)
        sales = np.maximum(0,
            rng.uniform(1.5,4.0) + rng.uniform(-0.001,0.002)*t
            + 1.5*np.sin(2*np.pi*t/7+rng.uniform(0,2*np.pi))
            + 2.0*np.sin(2*np.pi*t/365+rng.uniform(0,2*np.pi))
            + rng.normal(0,0.8,n_days)
        ).astype(int)
        evt_idx = np.where(calendar_df["event_name_1"]!="")[0]
        sales[evt_idx] = (sales[evt_idx]*rng.uniform(1.3,2.5,len(evt_idx))).astype(int)
        sales[rng.random(n_days)<0.05] = 0
        items.append({"id":f"{item_id}_{store}","item_id":item_id,
                      "dept_id":f"{cat}_1","cat_id":cat,"store_id":store,"state_id":state})
        sales_matrix.append(sales)

    matrix_df = pd.DataFrame(sales_matrix, columns=calendar_df["d"].tolist())
    sales_df  = pd.concat([pd.DataFrame(items), matrix_df], axis=1)

    # Prices
    price_rows = []
    for item in items:
        base = rng.uniform(1.5, 15.0)
        for wk in calendar_df["wm_yr_wk"].unique():
            chg = rng.choice([1.0,0.9,0.85,1.05], p=[0.7,0.15,0.1,0.05])
            price_rows.append({"store_id":item["store_id"],"item_id":item["item_id"],
                                "wm_yr_wk":wk,"sell_price":round(base*chg,2)})
    return sales_df, calendar_df, pd.DataFrame(price_rows)

def main():
    root        = Path(__file__).resolve().parent.parent.parent
    config_path = root / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out_dir = root / cfg["data"]["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sales, calendar, prices = generate_m5_style_data(
        n_items=cfg["data"]["n_demo_items"], n_days=cfg["data"]["n_demo_days"])
    sales.to_csv(out_dir/"sales_train_evaluation.csv", index=False)
    calendar.to_csv(out_dir/"calendar.csv", index=False)
    prices.to_csv(out_dir/"sell_prices.csv", index=False)
    logger.info(f"✅ Demo data saved to {out_dir}")
    logger.info(f"   sales: {sales.shape} | calendar: {calendar.shape} | prices: {prices.shape}")

if __name__ == "__main__":
    main()
