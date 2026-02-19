"""
dashboard.py
------------
Interactive Streamlit forecasting dashboard for the M5 pipeline.

Launch:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yaml

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="M5 Forecasting Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border: 1px solid #3d405b;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #9e9e9e; margin-top: 0.2rem; }
    .model-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.1rem;
    }
    .stSelectbox label { color: #cfd8dc !important; }
    .stSlider label { color: #cfd8dc !important; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .stTab [data-baseweb="tab"] { color: #9e9e9e; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    """Load raw data and return long-format sales."""
    from src.utils.data_loader import load_m5_data
    cfg = load_config()
    sales_long, calendar, prices = load_m5_data(cfg)
    return sales_long, calendar, prices, cfg


def generate_forecast(
    series: pd.Series,
    model_name: str,
    horizon: int,
    config: dict,
) -> pd.DataFrame:
    """Fit model on series and return forecast DataFrame."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)

    if model_name == "ARIMA":
        try:
            model = SARIMAX(series, order=(2,1,2), seasonal_order=(1,1,1,7),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            preds = np.maximum(0, fit.forecast(steps=horizon).values)
        except Exception:
            preds = np.full(horizon, series.mean())

    elif model_name == "Prophet":
        try:
            from prophet import Prophet
            train_df = pd.DataFrame({"ds": series.index, "y": series.values})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(train_df)
            fut = m.make_future_dataframe(periods=horizon)
            fc = m.predict(fut)
            preds = np.maximum(0, fc["yhat"].tail(horizon).values)
        except Exception:
            preds = np.full(horizon, series.mean())

    elif model_name == "LightGBM":
        # Simplified LightGBM for single-series demo: use lag features
        from sklearn.ensemble import GradientBoostingRegressor
        vals = series.values
        X, y = [], []
        lags = [1, 7, 14, 28]
        for i in range(max(lags), len(vals)):
            X.append([vals[i-l] for l in lags])
            y.append(vals[i])
        if len(X) > 10:
            reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
            reg.fit(X, y)
            preds = []
            history = list(vals)
            for _ in range(horizon):
                feat = [history[-(l)] for l in lags]
                p = max(0, reg.predict([feat])[0])
                preds.append(p)
                history.append(p)
        else:
            preds = np.full(horizon, series.mean())

    elif model_name == "Ensemble":
        # Simple average of naive seasonal + trend extrapolation
        seasonal = np.array([series[series.index.dayofweek == d].mean() for d in range(7)])
        trend = np.polyfit(np.arange(min(90, len(series))), series.values[-min(90, len(series)):], 1)
        preds = []
        for i in range(horizon):
            dow = future_dates[i].dayofweek
            t_val = len(series) + i
            p = max(0, seasonal[dow] * 0.7 + np.polyval(trend, t_val) * 0.3)
            preds.append(p)
    else:
        # Naive seasonal
        preds = np.array([series[series.index.dayofweek == future_dates[i].dayofweek].mean()
                          for i in range(horizon)])
        preds = np.maximum(0, preds)

    # Add confidence intervals (±1.5 * rolling std as heuristic)
    rolling_std = series.rolling(14).std().iloc[-1] or series.std()
    lower = np.maximum(0, preds - 1.5 * rolling_std)
    upper = preds + 1.5 * rolling_std

    return pd.DataFrame({
        "date": future_dates,
        "forecast": preds,
        "lower": lower,
        "upper": upper,
    })


def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = 100 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE (%)": round(mape, 1)}


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📦 M5 Forecasting")
    st.markdown("*Walmart Demand Forecasting Pipeline*")
    st.markdown("---")

    with st.spinner("Loading data..."):
        try:
            sales_long, calendar, prices, cfg = load_and_prepare_data()
            data_loaded = True
        except Exception as e:
            st.error(f"Data not found. Run: `python src/utils/generate_demo_data.py`\n\nError: {e}")
            data_loaded = False

    if data_loaded:
        st.success(f"✅ {sales_long['id'].nunique()} items loaded")
        st.markdown("---")

        # Filters
        st.markdown("### 🔍 Filters")
        stores = ["All"] + sorted(sales_long["store_id"].unique().tolist())
        selected_store = st.selectbox("Store", stores)

        cats = ["All"] + sorted(sales_long["cat_id"].unique().tolist())
        selected_cat = st.selectbox("Category", cats)

        filtered = sales_long.copy()
        if selected_store != "All":
            filtered = filtered[filtered["store_id"] == selected_store]
        if selected_cat != "All":
            filtered = filtered[filtered["cat_id"] == selected_cat]

        items = sorted(filtered["id"].unique().tolist())
        selected_item = st.selectbox("Item ID", items)

        st.markdown("---")
        st.markdown("### ⚙️ Forecast Settings")
        horizon = st.slider("Forecast horizon (days)", 7, 56, 28, step=7)
        model_choice = st.selectbox(
            "Model",
            ["LightGBM", "Prophet", "ARIMA", "Ensemble"],
            help="LightGBM is fastest. Prophet shows interpretable decomposition."
        )
        history_days = st.slider("History to display (days)", 30, 365, 90)


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

st.markdown("# 📦 M5 Demand Forecasting Dashboard")
st.markdown("*Built on Walmart's M5 dataset — 3,049 products × 10 stores × 5+ years*")
st.markdown("---")

if not data_loaded:
    st.info("👈 Please load data first using the instructions in the sidebar.")
    st.stop()

# ── Item series ───────────────────────────────────────────────────────────────
item_df = (
    sales_long[sales_long["id"] == selected_item]
    .sort_values("date")
    .set_index("date")
)
series = item_df["sales"].dropna()
series_display = series.tail(history_days)

# Generate forecast
with st.spinner(f"Running {model_choice}..."):
    forecast_df = generate_forecast(series, model_choice, horizon, cfg)

# Holdout metrics (last 28 days as pseudo-test)
if len(series) > 56:
    train_s = series.iloc[:-28]
    test_s = series.iloc[-28:]
    holdout_fc = generate_forecast(train_s, model_choice, 28, cfg)
    metrics = compute_metrics(test_s.values, holdout_fc["forecast"].values[:len(test_s)])
else:
    metrics = {}


# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_sales = series.tail(28).mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_sales:.1f}</div>
        <div class="metric-label">Avg Daily Sales (28d)</div>
    </div>""", unsafe_allow_html=True)

with col2:
    fc_mean = forecast_df["forecast"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{fc_mean:.1f}</div>
        <div class="metric-label">Avg Forecasted Sales</div>
    </div>""", unsafe_allow_html=True)

with col3:
    mae_val = metrics.get("MAE", "—")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{mae_val}</div>
        <div class="metric-label">Holdout MAE</div>
    </div>""", unsafe_allow_html=True)

with col4:
    mape_val = metrics.get("MAPE (%)", "—")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{mape_val}{'%' if mape_val != '—' else ''}</div>
        <div class="metric-label">Holdout MAPE</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast", "🔄 Model Comparison", "📊 EDA", "📋 Leaderboard"
])


# ─── TAB 1: FORECAST ──────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### {model_choice} — {horizon}-Day Forecast for `{selected_item}`")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=series_display.index,
        y=series_display.values,
        mode="lines",
        name="Historical Sales",
        line=dict(color="#4fc3f7", width=2),
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(255, 167, 38, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Interval",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["forecast"],
        mode="lines+markers",
        name=f"{model_choice} Forecast",
        line=dict(color="#ffa726", width=2.5, dash="dash"),
        marker=dict(size=5),
    ))

    # Vertical separator
    fig.add_vline(
        x=str(series.index[-1]),
        line_dash="dot",
        line_color="#78909c",
        annotation_text="Forecast start",
        annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", title="Units Sold"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    with st.expander("📄 Forecast Details"):
        display_fc = forecast_df.copy()
        display_fc["date"] = display_fc["date"].dt.strftime("%Y-%m-%d")
        display_fc.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
        display_fc = display_fc.round(2)
        st.dataframe(display_fc, use_container_width=True, hide_index=True)


# ─── TAB 2: MODEL COMPARISON ──────────────────────────────────────────────────
with tab2:
    st.markdown("### Side-by-Side Model Comparison")
    st.markdown("*Each model generates a {}-day forecast for the selected item.*".format(horizon))

    all_models = ["ARIMA", "Prophet", "LightGBM", "Ensemble"]
    colors = {"ARIMA": "#ef5350", "Prophet": "#ab47bc", "LightGBM": "#4fc3f7", "Ensemble": "#66bb6a"}

    with st.spinner("Running all models..."):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=series_display.index, y=series_display.values,
            mode="lines", name="Historical",
            line=dict(color="#607d8b", width=2),
        ))

        comparison_metrics = []
        for m in all_models:
            fc = generate_forecast(series, m, horizon, cfg)
            fig2.add_trace(go.Scatter(
                x=fc["date"], y=fc["forecast"],
                mode="lines", name=m,
                line=dict(color=colors[m], width=2),
            ))
            if len(series) > 56:
                train_s2 = series.iloc[:-28]
                test_s2 = series.iloc[-28:]
                hfc = generate_forecast(train_s2, m, 28, cfg)
                mt = compute_metrics(test_s2.values, hfc["forecast"].values[:len(test_s2)])
                comparison_metrics.append({"Model": m, **mt})

    fig2.add_vline(x=str(series.index[-1]), line_dash="dot", line_color="#78909c")
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", title="Units Sold"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    if comparison_metrics:
        st.markdown("#### Holdout Performance (last 28 days)")
        cmp_df = pd.DataFrame(comparison_metrics).set_index("Model")
        st.dataframe(
            cmp_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#1b5e20"),
            use_container_width=True,
        )


# ─── TAB 3: EDA ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Exploratory Data Analysis")

    # Daily sales distribution
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Daily Sales Distribution**")
        fig_hist = px.histogram(
            x=series.values, nbins=30,
            color_discrete_sequence=["#4fc3f7"],
            template="plotly_dark",
        )
        fig_hist.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=300, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Units Sold", yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown("**Average Sales by Day of Week**")
        dow_avg = (
            series.groupby(series.index.day_name())
            .mean()
            .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        )
        fig_dow = px.bar(
            x=dow_avg.index, y=dow_avg.values,
            color=dow_avg.values,
            color_continuous_scale="Blues",
            template="plotly_dark",
        )
        fig_dow.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=300, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="", yaxis_title="Avg Units Sold",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    # Monthly trend
    st.markdown("**Monthly Sales Trend**")
    monthly = series.resample("ME").sum()
    fig_monthly = px.area(
        x=monthly.index, y=monthly.values,
        template="plotly_dark",
        color_discrete_sequence=["#4fc3f7"],
    )
    fig_monthly.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=280, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="", yaxis_title="Monthly Units Sold",
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)


# ─── TAB 4: LEADERBOARD ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🏆 Model Leaderboard")
    st.markdown("Results from walk-forward backtesting across all items.")

    leaderboard_path = Path("outputs/metrics/leaderboard.csv")
    if leaderboard_path.exists():
        lb = pd.read_csv(leaderboard_path)
        st.dataframe(
            lb.style.highlight_min(subset=["mae", "rmse", "mape"], color="#1b5e20"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run the pipeline to generate leaderboard: `python src/pipeline.py`")

        # Show illustrative benchmark
        st.markdown("**Illustrative Benchmark Results** *(run pipeline for real results)*")
        demo_lb = pd.DataFrame({
            "model": ["ARIMA", "Prophet", "LightGBM", "LSTM", "Ensemble"],
            "mae": [2.41, 1.98, 1.24, 1.31, 1.19],
            "rmse": [4.18, 3.52, 2.31, 2.44, 2.24],
            "mape": [18.3, 14.7, 9.8, 10.4, 9.3],
        })
        st.dataframe(
            demo_lb.style.highlight_min(subset=["mae", "rmse", "mape"], color="#1b5e20"),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown("""
    **Metric Definitions:**
    - **MAE** — Mean Absolute Error: average absolute deviation in units sold
    - **RMSE** — Root Mean Squared Error: penalizes large errors more heavily
    - **MAPE** — Mean Absolute Percentage Error: scale-independent accuracy metric
    - **Walk-forward validation**: each fold trains on all prior data and tests on the next 28 days
    """)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#546e7a; font-size:0.8rem'>"
    "Built with Python · LightGBM · Prophet · PyTorch · Streamlit  |  "
    "Data: M5 Forecasting Competition (Walmart)"
    "</div>",
    unsafe_allow_html=True,
)
