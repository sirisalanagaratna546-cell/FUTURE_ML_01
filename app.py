"""
app.py
------
Streamlit Dashboard for Favorita Grocery Sales Forecasting.
Supports uploading new datasets or using the default pre-processed data.

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import pickle
import logging
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Favorita Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Style ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1e293b; }
    .sub-title  { font-size: 1rem;   color: #64748b; margin-bottom: 1.5rem; }
    .kpi-card   { background: #ffffff; border-radius: 12px; padding: 1.2rem 1.5rem;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.10); border-left: 4px solid; }
    .metric-val { font-size: 1.7rem; font-weight: 700; }
    .metric-lbl { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
    div[data-testid="stSidebar"] { background-color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")

COLOR = {
    "primary": "#2563EB", "secondary": "#7C3AED",
    "success": "#059669",  "warning": "#D97706",
    "danger": "#DC2626",   "neutral": "#6B7280",
}


# ─── Data Loaders (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_preprocessed_data(data_dir):
    from src.data_preprocessing import run_preprocessing
    daily_df, family_daily, store_daily, stores = run_preprocessing(data_dir)
    return daily_df, family_daily, store_daily, stores


@st.cache_data(show_spinner=False)
def load_outputs(outputs_dir):
    metrics, forecast_df, test_result = None, None, None

    m_path = os.path.join(outputs_dir, "metrics.json")
    if os.path.exists(m_path):
        with open(m_path) as f:
            m = json.load(f)
        metrics = {
            "all_models": pd.DataFrame(m["all_models"]),
            "best_model": m["best_model"],
            "best_metrics": m["best_metrics"],
        }

    f_path = os.path.join(outputs_dir, "future_predictions.csv")
    if os.path.exists(f_path):
        forecast_df = pd.read_csv(f_path, parse_dates=["date"])

    return metrics, forecast_df


@st.cache_resource(show_spinner=False)
def load_model_bundle(outputs_dir):
    model_path = os.path.join(outputs_dir, "best_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def run_full_pipeline(data_dir, outputs_dir):
    """Run pipeline and cache results."""
    from src.data_preprocessing import run_preprocessing
    from src.feature_engineering import build_feature_matrix
    from src.train_model import train_all_models
    from src.forecast import run_forecast

    with st.spinner("⚙️ Step 1/4 — Preprocessing data..."):
        daily_df, family_daily, store_daily, stores = run_preprocessing(data_dir)

    with st.spinner("⚙️ Step 2/4 — Engineering features..."):
        feature_df = build_feature_matrix(daily_df)

    with st.spinner("⚙️ Step 3/4 — Training models..."):
        best_model, best_scaled, scaler, feature_cols, metrics_df, test_result, best_name = \
            train_all_models(feature_df, output_dir=outputs_dir)

    with st.spinner("⚙️ Step 4/4 — Generating forecast..."):
        forecast_df = run_forecast(daily_df, periods=30, output_dir=outputs_dir)

    st.cache_data.clear()
    st.cache_resource.clear()
    return daily_df, family_daily, metrics_df, test_result, forecast_df, best_name


# ─── KPI Cards ────────────────────────────────────────────────────────────────
def kpi_card(col, label, value, color, icon=""):
    col.markdown(f"""
    <div class="kpi-card" style="border-left-color:{color}">
        <div class="metric-val" style="color:{color}">{icon} {value}</div>
        <div class="metric-lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/shop.png", width=48)
    st.markdown("## 🛒 Sales Forecasting")
    st.markdown("---")

    st.markdown("### ⚙️ Controls")
    forecast_periods = st.slider("Forecast Horizon (days)", 7, 90, 30)
    show_raw = st.checkbox("Show raw data tables", False)

    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    uploaded = st.file_uploader("Upload train CSV", type=["csv"])

    st.markdown("---")
    rerun_btn = st.button("🔄 Re-run Full Pipeline", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("📦 Dataset: Corporación Favorita  \n🗓️ Period: 2015 – 2017  \n🏪 54 Stores · 33 Families")


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🛒 Favorita Grocery Sales Forecasting</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ML-powered demand forecasting across 54 stores & 33 product families in Ecuador</p>', unsafe_allow_html=True)

# Load data
try:
    with st.spinner("Loading data..."):
        data_dir_use = DATA_DIR
        if uploaded is not None:
            import tempfile, shutil
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, "train.csv")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getvalue())
            for fname in ["stores.csv", "holidays_events.csv", "oil.csv", "transactions.csv"]:
                src = os.path.join(DATA_DIR, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(tmp_dir, fname))
            data_dir_use = tmp_dir

        daily_df, family_daily, store_daily, stores = load_preprocessed_data(data_dir_use)
        metrics, forecast_df = load_outputs(OUTPUTS_DIR)

    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

if rerun_btn and data_loaded:
    with st.spinner("Running full pipeline..."):
        daily_df, family_daily, metrics_df, test_result, forecast_df, best_name = \
            run_full_pipeline(data_dir_use, OUTPUTS_DIR)
    metrics, forecast_df = load_outputs(OUTPUTS_DIR)
    st.success("✅ Pipeline complete!")

if data_loaded and metrics is not None and forecast_df is not None:

    # ── KPI Row ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 Key Performance Indicators")
    k1, k2, k3, k4, k5 = st.columns(5)

    kpi_card(k1, "Total Historical Sales", f"${daily_df['total_sales'].sum()/1e9:.2f}B",
             COLOR["primary"], "💰")
    kpi_card(k2, "Avg Daily Sales", f"${daily_df['total_sales'].mean():,.0f}",
             COLOR["secondary"], "📅")
    kpi_card(k3, "Peak Daily Sales",
             f"${daily_df['total_sales'].max():,.0f}",
             COLOR["warning"], "🚀")
    kpi_card(k4, "30-Day Forecast Total",
             f"${forecast_df['forecasted_sales'].sum()/1e6:.1f}M",
             COLOR["success"], "🔮")
    best_r2 = metrics["best_metrics"].get("R2", 0) if isinstance(metrics["best_metrics"], dict) else 0
    kpi_card(k5, f"Best Model R²", f"{best_r2:.3f}",
             COLOR["danger"], "🏆")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Historical Trend", "🗓️ Seasonality",
        "🔮 Forecast", "🎯 Model Performance",
        "🏪 Category Analysis", "📋 Data"
    ])

    # ── Tab 1: Historical Trend ───────────────────────────────────────────────
    with tab1:
        st.markdown("#### Historical Daily Sales Trend")

        resample_opt = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True)
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
        freq = freq_map[resample_opt]

        ts = daily_df.set_index("date")["total_sales"].resample(freq).sum().reset_index()
        ts.columns = ["date", "total_sales"]
        ts["ma"] = ts["total_sales"].rolling(4 if freq != "D" else 30).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["date"], y=ts["total_sales"],
                                  mode="lines", name="Sales",
                                  line=dict(color=COLOR["primary"], width=1.5), opacity=0.6))
        fig.add_trace(go.Scatter(x=ts["date"], y=ts["ma"],
                                  mode="lines", name="Moving Avg",
                                  line=dict(color=COLOR["warning"], width=2.5)))
        fig.update_layout(template="plotly_white", hovermode="x unified",
                          xaxis_title="Date", yaxis_title="Total Sales",
                          legend=dict(orientation="h"), height=420)
        st.plotly_chart(fig, use_container_width=True)

        # YoY comparison
        st.markdown("#### Year-over-Year Comparison")
        daily_df["year"] = daily_df["date"].dt.year
        daily_df["month"] = daily_df["date"].dt.month
        yoy = daily_df.groupby(["year", "month"])["total_sales"].sum().reset_index()
        fig2 = px.line(yoy, x="month", y="total_sales", color="year",
                       color_discrete_sequence=px.colors.qualitative.Set1,
                       labels={"total_sales": "Monthly Sales", "month": "Month"},
                       template="plotly_white")
        fig2.update_layout(height=350, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Seasonality ────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Seasonality Decomposition")

        c1, c2 = st.columns(2)
        daily_df["month_name"] = daily_df["date"].dt.strftime("%b")
        daily_df["dow"] = daily_df["date"].dt.day_name()

        month_agg = daily_df.groupby(["month", "month_name"])["total_sales"].mean().reset_index().sort_values("month")
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_agg = daily_df.groupby("dow")["total_sales"].mean().reset_index()
        dow_agg["dow"] = pd.Categorical(dow_agg["dow"], categories=dow_order, ordered=True)
        dow_agg = dow_agg.sort_values("dow")

        with c1:
            fig = px.bar(month_agg, x="month_name", y="total_sales",
                         title="Avg Sales by Month",
                         color="total_sales", color_continuous_scale="Blues",
                         labels={"total_sales": "Avg Sales", "month_name": "Month"},
                         template="plotly_white")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(dow_agg, x="dow", y="total_sales",
                         title="Avg Sales by Day of Week",
                         color="total_sales", color_continuous_scale="Purples",
                         labels={"total_sales": "Avg Sales", "dow": "Day"},
                         template="plotly_white")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap: month × dow
        st.markdown("#### 🔥 Sales Heatmap (Month × Day of Week)")
        daily_df["dow_num"] = daily_df["date"].dt.dayofweek
        heat = daily_df.groupby(["month_name", "month", "dow", "dow_num"])["total_sales"].mean().reset_index()
        heat = heat.sort_values(["month", "dow_num"])
        pivot = heat.pivot(index="month_name", columns="dow", values="total_sales")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot.reindex([m for m in month_order if m in pivot.index])
        pivot = pivot[dow_order]

        fig = px.imshow(pivot, color_continuous_scale="Blues",
                        labels=dict(color="Avg Sales"),
                        aspect="auto", template="plotly_white")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Forecast ───────────────────────────────────────────────────────
    with tab3:
        st.markdown(f"#### 🔮 {forecast_periods}-Day Sales Forecast")
        st.info(f"**Best Model:** {metrics['best_model']} | "
                f"**RMSE:** {metrics['best_metrics'].get('RMSE', 'N/A'):,.0f} | "
                f"**MAPE:** {metrics['best_metrics'].get('MAPE', 'N/A'):.2f}% | "
                f"**R²:** {metrics['best_metrics'].get('R2', 'N/A'):.4f}")

        last_n = 90
        hist_tail = daily_df.tail(last_n)
        fc = forecast_df.head(forecast_periods)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_tail["date"], y=hist_tail["total_sales"],
                                  mode="lines", name="Historical",
                                  line=dict(color=COLOR["primary"], width=2)))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["date"], fc["date"][::-1]]),
            y=pd.concat([fc["upper_bound"], fc["lower_bound"][::-1]]),
            fill="toself", fillcolor="rgba(5,150,105,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))

        fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecasted_sales"],
                                  mode="lines+markers", name="Forecast",
                                  line=dict(color=COLOR["success"], width=2.5),
                                  marker=dict(size=6)))

        split_date = str(daily_df["date"].max().date())
        fig.add_shape(type="line", x0=split_date, x1=split_date,
                      y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color=COLOR["neutral"], dash="dot", width=1.5))
        fig.add_annotation(x=split_date, y=0.98, xref="x", yref="paper",
                           text="Forecast →", showarrow=False,
                           font=dict(size=11, color=COLOR["neutral"]))

        fig.update_layout(template="plotly_white", hovermode="x unified",
                          xaxis_title="Date", yaxis_title="Forecasted Sales",
                          legend=dict(orientation="h"), height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Forecast Details")
            display_fc = fc[["date", "day_of_week", "forecasted_sales", "lower_bound", "upper_bound"]].copy()
            display_fc.columns = ["Date", "Day", "Forecast", "Lower", "Upper"]
            display_fc["Forecast"] = display_fc["Forecast"].map("${:,.0f}".format)
            display_fc["Lower"]    = display_fc["Lower"].map("${:,.0f}".format)
            display_fc["Upper"]    = display_fc["Upper"].map("${:,.0f}".format)
            st.dataframe(display_fc, use_container_width=True, height=400)

        with c2:
            st.markdown("#### Forecast Summary")
            raw = fc["forecasted_sales"]
            st.metric("Total (30 days)", f"${raw.sum():,.0f}")
            st.metric("Daily Average", f"${raw.mean():,.0f}")
            st.metric("Peak Day", f"${raw.max():,.0f}")
            st.metric("Lowest Day", f"${raw.min():,.0f}")
            st.metric("Weekend vs Weekday",
                      f"{fc[fc['is_weekend']==1]['forecasted_sales'].mean()/fc[fc['is_weekend']==0]['forecasted_sales'].mean():.2f}x")

        # Download
        csv_bytes = fc.to_csv(index=False).encode()
        st.download_button("⬇️ Download Forecast CSV", csv_bytes,
                           file_name="future_predictions.csv", mime="text/csv")

    # ── Tab 4: Model Performance ──────────────────────────────────────────────
    with tab4:
        st.markdown("#### 🏆 Model Comparison")

        metrics_df = metrics["all_models"]
        best_name = metrics["best_model"]

        # Highlight best
        def highlight_best(row):
            is_best = row["model"] == best_name
            return ["background-color: #dcfce7" if is_best else "" for _ in row]

        display_m = metrics_df.copy()
        for col in ["MAE", "RMSE", "MAPE"]:
            if col in display_m.columns:
                display_m[col] = display_m[col].map("{:,.1f}".format)
        if "R2" in display_m.columns:
            display_m["R2"] = display_m["R2"].map("{:.4f}".format)
        st.dataframe(display_m.style.apply(highlight_best, axis=1), use_container_width=True)

        # Charts
        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            "MAE (lower = better)", "RMSE (lower = better)",
            "R² (higher = better)", "MAPE % (lower = better)"
        ])
        m_df = metrics["all_models"]
        colors = [COLOR["success"] if r["model"] == best_name else COLOR["primary"]
                  for _, r in m_df.iterrows()]

        for (metric, row, col) in [("MAE", 1, 1), ("RMSE", 1, 2), ("R2", 2, 1), ("MAPE", 2, 2)]:
            if metric not in m_df.columns:
                continue
            fig.add_trace(go.Bar(x=m_df["model"], y=m_df[metric],
                                  marker_color=colors, showlegend=False,
                                  text=m_df[metric].round(2), textposition="outside"),
                          row=row, col=col)

        fig.update_layout(template="plotly_white", height=550)
        st.plotly_chart(fig, use_container_width=True)

        # Load test results for actual vs predicted
        m_path = os.path.join(OUTPUTS_DIR, "metrics.json")
        if os.path.exists(m_path):
            st.markdown("#### Feature Importance Note")
            st.info(f"✅ **{best_name}** was selected as the best model based on lowest RMSE. "
                    "It used 47 engineered features including lag values, rolling averages, "
                    "calendar features, cyclical encodings, and external signals (oil price, "
                    "transactions, holidays).")

    # ── Tab 5: Category Analysis ──────────────────────────────────────────────
    with tab5:
        st.markdown("#### 🏪 Product Family Sales Analysis")

        family_total = family_daily.groupby("family")["sales"].sum().sort_values(ascending=False)
        top_n = st.slider("Show top N families", 5, 33, 10)
        top_fam = family_total.head(top_n)

        fig = go.Figure(go.Bar(
            x=top_fam.values, y=top_fam.index,
            orientation="h",
            marker_color=px.colors.qualitative.Set3[:top_n],
            text=[f"${v/1e6:.1f}M" for v in top_fam.values],
            textposition="outside",
        ))
        fig.update_layout(template="plotly_white", height=max(400, top_n * 35),
                          xaxis_title="Total Sales (USD)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Monthly trend for selected families
        st.markdown("#### Monthly Trend by Family")
        selected_fams = st.multiselect("Select families", family_total.index.tolist(),
                                        default=family_total.index[:5].tolist())
        if selected_fams:
            sel_df = family_daily[family_daily["family"].isin(selected_fams)].copy()
            sel_df["month"] = sel_df["date"].dt.to_period("M").dt.to_timestamp()
            monthly = sel_df.groupby(["month", "family"])["sales"].sum().reset_index()
            fig2 = px.line(monthly, x="month", y="sales", color="family",
                           template="plotly_white", labels={"sales": "Monthly Sales", "month": "Month"},
                           color_discrete_sequence=px.colors.qualitative.Set1)
            fig2.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

        # Store-level (uses transactions proxy)
        st.markdown("#### Store Cluster Summary")
        if stores is not None:
            stores_agg = store_daily.groupby("store_nbr")["sales"].sum().reset_index()
            stores_merged = stores.merge(stores_agg, on="store_nbr", how="left").fillna(0)
            city_agg = stores_merged.groupby("city")["sales"].sum().sort_values(ascending=False).head(15).reset_index()
            fig3 = px.bar(city_agg, x="city", y="sales", title="Top 15 Cities by Sales",
                          color="sales", color_continuous_scale="Blues",
                          template="plotly_white")
            fig3.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 6: Raw Data ───────────────────────────────────────────────────────
    with tab6:
        st.markdown("#### 📋 Data Explorer")

        data_option = st.selectbox("Select dataset", [
            "Daily Aggregated Sales", "Family-Level Sales", "Forecast Output"
        ])

        if data_option == "Daily Aggregated Sales":
            st.dataframe(daily_df.tail(100), use_container_width=True)
        elif data_option == "Family-Level Sales":
            st.dataframe(family_daily.tail(200), use_container_width=True)
        elif data_option == "Forecast Output":
            st.dataframe(forecast_df, use_container_width=True)

        st.markdown("#### Dataset Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Days", f"{len(daily_df):,}")
        c2.metric("Product Families", f"{family_daily['family'].nunique()}")
        c3.metric("Stores", "54")

elif not data_loaded:
    st.error("❌ Could not load data. Please ensure data files are in the `/data` directory.")

else:
    st.warning("⚠️ Outputs not found. Click **Re-run Full Pipeline** in the sidebar to train models and generate forecasts.")
    if st.button("▶️ Run Pipeline Now", type="primary"):
        daily_df, family_daily, metrics_df, test_result, forecast_df, best_name = \
            run_full_pipeline(data_dir_use, OUTPUTS_DIR)
        st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:0.8rem'>"
    "Favorita Sales Forecasting Dashboard · Built with Streamlit & Plotly · "
    "Data: Corporación Favorita Grocery Sales (2015–2017)"
    "</div>",
    unsafe_allow_html=True,
)
