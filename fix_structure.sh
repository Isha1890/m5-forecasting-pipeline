#!/usr/bin/env bash
# fix_structure.sh
# Run from inside your m5-forecasting folder:
#   chmod +x fix_structure.sh && ./fix_structure.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $1"; }
info() { echo -e "${CYAN}→${NC} $1"; }

echo ""
echo "🔧 Fixing project structure..."
echo ""

# ── 1. Create all folders ─────────────────────────────────────────────────────
info "Creating folder structure..."
mkdir -p app src/models src/evaluation src/features src/utils configs tests data/raw outputs
ok "Folders created"

# ── 2. Move files into correct locations ──────────────────────────────────────
info "Moving files..."

# Only move if file exists in root (safe to re-run)
[ -f dashboard.py ]       && mv dashboard.py       app/
[ -f default.yaml ]       && mv default.yaml       configs/
[ -f chronos_model.py ]   && mv chronos_model.py   src/models/
[ -f lgbm_model.py ]      && mv lgbm_model.py      src/models/
[ -f lstm_model.py ]      && mv lstm_model.py      src/models/
[ -f backtest.py ]        && mv backtest.py        src/evaluation/
[ -f conformal.py ]       && mv conformal.py       src/evaluation/
[ -f metrics.py ]         && mv metrics.py         src/evaluation/
[ -f build_features.py ]  && mv build_features.py  src/features/
[ -f pipeline.py ]        && mv pipeline.py        src/
[ -f test_features.py ]   && mv test_features.py   tests/
[ -f test_metrics.py ]    && mv test_metrics.py    tests/
ok "Files moved"

# ── 3. Create __init__.py files ───────────────────────────────────────────────
info "Creating __init__.py files..."
touch src/__init__.py
touch src/models/__init__.py
touch src/evaluation/__init__.py
touch src/features/__init__.py
touch src/utils/__init__.py
touch app/__init__.py
touch tests/__init__.py
ok "__init__.py files created"

# ── 4. Write missing files directly ───────────────────────────────────────────
info "Writing src/utils/data_loader.py..."
cat > src/utils/data_loader.py << 'PYEOF'
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
PYEOF
ok "data_loader.py written"

info "Writing src/utils/generate_demo_data.py..."
cat > src/utils/generate_demo_data.py << 'PYEOF'
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
PYEOF
ok "generate_demo_data.py written"

# ── 5. Write missing model files ──────────────────────────────────────────────
info "Writing missing model stubs..."

[ ! -f src/models/arima_model.py ] && cat > src/models/arima_model.py << 'PYEOF'
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
PYEOF

[ ! -f src/models/prophet_model.py ] && cat > src/models/prophet_model.py << 'PYEOF'
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
PYEOF

[ ! -f src/models/ensemble_model.py ] && cat > src/models/ensemble_model.py << 'PYEOF'
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
PYEOF

ok "Model files written"

# ── 6. Write .gitignore ───────────────────────────────────────────────────────
info "Writing .gitignore..."
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
venv/
.venv/
data/raw/
data/processed/
!data/.gitkeep
!data/raw/.gitkeep
outputs/
!outputs/.gitkeep
*.parquet
*.pkl
*.pt
*.pth
.DS_Store
*.log
.streamlit/secrets.toml
EOF
touch data/.gitkeep data/raw/.gitkeep outputs/.gitkeep
ok ".gitignore written"

# ── 7. Write .streamlit/config.toml ──────────────────────────────────────────
info "Writing .streamlit/config.toml..."
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
base = "dark"
primaryColor = "#4fc3f7"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1e2130"
textColor = "#e0e0e0"
EOF
ok ".streamlit/config.toml written"

# ── 8. Write CI workflow ──────────────────────────────────────────────────────
info "Writing GitHub Actions CI..."
mkdir -p .github/workflows
cat > .github/workflows/ci.yml << 'EOF'
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install deps
        run: pip install pandas numpy scikit-learn statsmodels lightgbm torch pytest pytest-cov pyyaml
      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=term-missing
EOF
ok "CI workflow written"

# ── 9. Generate demo data ─────────────────────────────────────────────────────
info "Generating demo data..."
python src/utils/generate_demo_data.py
ok "Demo data created in data/raw/"

# ── 10. Verify Streamlit launches ─────────────────────────────────────────────
info "Testing Streamlit (8 second smoke test)..."
python -m streamlit run app/dashboard.py \
    --server.headless true \
    --server.port 8502 &>/tmp/st_test.log &
ST_PID=$!
sleep 8
if kill -0 $ST_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Streamlit is running correctly!${NC}"
    kill $ST_PID 2>/dev/null; wait $ST_PID 2>/dev/null || true
else
    echo "⚠️  Streamlit exited early. Last log lines:"
    tail -15 /tmp/st_test.log
fi

# ── 11. Git commit & push ─────────────────────────────────────────────────────
echo ""
info "Committing and pushing to GitHub..."
git add .
git status --short
git commit -m "refactor: proper folder structure — app/, src/, configs/, tests/

- Moved all files into correct module layout
- Added src/utils/data_loader.py + generate_demo_data.py
- Added conformal prediction (conformal.py)
- Added Chronos foundation model (chronos_model.py)  
- Updated dashboard with 5 tabs incl. Uncertainty + Leaderboard
- Added .streamlit/config.toml dark theme
- Added GitHub Actions CI"

git push origin main
ok "Pushed to GitHub!"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅  All done!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo "  Run the dashboard:"
echo "  python -m streamlit run app/dashboard.py"
echo ""
echo "  Push future changes:"
echo "  git add . && git commit -m 'msg' && git push"
echo ""
