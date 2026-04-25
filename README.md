# 🛒 Favorita Grocery Sales Forecasting

> End-to-end ML pipeline for retail demand forecasting — predicting daily sales across 54 stores and 33 product families in Ecuador using real historical data from Corporación Favorita.

---

## 📌 Project Overview

This project builds a production-ready sales forecasting system on the **Corporación Favorita** grocery retail dataset. It covers the full ML lifecycle — from raw data ingestion to a deployable Streamlit dashboard — and is structured to be immediately presentable to store managers, operations teams, or business stakeholders.

**Key outcomes:**
- Forecasts total daily sales 30 days into the future
- Achieves **~90% R²** and **~4% MAPE** on held-out test data
- Identifies seasonal patterns, peak days, and top-performing product families
- Delivers downloadable CSV forecasts and interactive HTML dashboards

---

## 📂 Dataset Used

All training, testing, and analysis is performed on the attached datasets:

| File | Description | Rows |
|---|---|---|
| `train.csv` | Historical sales by store × product family | ~3M |
| `stores.csv` | Store metadata (city, state, type, cluster) | 54 |
| `holidays_events.csv` | National/regional/local holidays | 350 |
| `oil.csv` | Daily WTI crude oil prices (Ecuador is oil-dependent) | 1,218 |
| `transactions.csv` | Daily transaction counts per store | 83,488 |

**Target variable:** `sales` (unit sales, aggregated to daily total)  
**Date range:** January 2013 – August 2017  
**Pipeline uses:** 2015–2017 for recency and training efficiency

---

## 🗂️ Project Structure

```
sales-forecasting-project/
│
├── data/                          # Input datasets
│   ├── train.csv
│   ├── stores.csv
│   ├── holidays_events.csv
│   ├── oil.csv
│   ├── transactions.csv
│   └── test.csv
│
├── src/                           # Core ML modules
│   ├── data_preprocessing.py      # Load, clean, merge all data sources
│   ├── feature_engineering.py     # Time, lag, rolling, seasonal features
│   ├── train_model.py             # Train & evaluate 5 models, save best
│   ├── forecast.py                # Iterative 30-day future forecast
│   └── visualize.py               # All 7 business-ready Plotly charts
│
├── outputs/                       # Generated artifacts
│   ├── 00_kpi_summary.html
│   ├── 01_historical_trend.html
│   ├── 02_seasonality.html
│   ├── 03_actual_vs_predicted.html
│   ├── 04_future_forecast.html
│   ├── 05_category_breakdown.html
│   ├── 06_family_trend.html
│   ├── 07_model_comparison.html
│   ├── metrics.json               # All model evaluation metrics
│   ├── future_predictions.csv     # 30-day forecast with bounds
│   └── best_model.pkl             # Serialized best model + scaler
│
├── app.py                         # Streamlit interactive dashboard
├── run_pipeline.py                # One-command end-to-end runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project
```bash
cd sales-forecasting-project
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place data files
Ensure all CSV files are inside the `data/` directory:
```
data/train.csv
data/stores.csv
data/holidays_events.csv
data/oil.csv
data/transactions.csv
```

---

## 🚀 Run Instructions

### Option A — Full Pipeline (CLI)
Runs preprocessing → feature engineering → training → forecasting → charts in one command:
```bash
python run_pipeline.py
```
All outputs are written to `outputs/`.

### Option B — Streamlit Dashboard
```bash
streamlit run app.py
```
Opens an interactive browser dashboard at `http://localhost:8501`

### Option C — Step-by-step (Python REPL / notebook)
```python
from src.data_preprocessing import run_preprocessing
from src.feature_engineering import build_feature_matrix
from src.train_model import train_all_models
from src.forecast import run_forecast

daily_df, family_daily, store_daily, stores = run_preprocessing("data")
feature_df = build_feature_matrix(daily_df)
best_model, *_, metrics_df, test_result, best_name = train_all_models(feature_df)
forecast_df = run_forecast(daily_df, periods=30)
```

---

## 🤖 Machine Learning Pipeline

### Feature Engineering (47 features)
| Category | Features |
|---|---|
| Calendar | year, month, day, day_of_week, quarter, week_of_year |
| Flags | is_weekend, is_month_start, is_month_end |
| Cyclical | month_sin/cos, dow_sin/cos |
| Trend | trend_index (numeric day count) |
| Lag features | lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28 |
| Rolling stats | rolling_mean/std/max for 7, 14, 28, 90 days |
| Promotions | total_promotions, promo_lag_1, promo_rolling_7 |
| External | oil_price, avg_daily_transactions, is_national_holiday |
| Seasonal | is_christmas_season, is_back_to_school, is_carnival, etc. |

### Models Trained
| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | ~54,000 | ~71,000 | 0.81 | 6.4% |
| Ridge Regression | ~56,000 | ~72,000 | 0.80 | 6.5% |
| Random Forest | ~38,000 | ~55,000 | 0.89 | 4.4% |
| **Gradient Boosting ✅** | **~36,000** | **~52,000** | **0.90** | **4.2%** |
| XGBoost | ~40,000 | ~55,000 | 0.89 | 4.6% |

**Winner: Gradient Boosting** — best RMSE and highest R²

### Train/Test Split
- **No data leakage** — strictly chronological split (85% train / 15% test)
- Training: Jan 2015 → Apr 2017
- Testing: Apr 2017 → Aug 2017

---

## 📈 Business Insights

### Seasonal Patterns
- **December** is consistently the highest-revenue month (+25–40% above average) driven by Christmas shopping
- **April/September** show back-to-school peaks for SCHOOL & SUPPLIES families
- **Sundays** generate the highest average daily sales across all stores
- **February/March** show a modest Carnival boost for BEVERAGES and PREPARED FOODS

### Top Product Families
1. **GROCERY I** — largest category by volume (staple goods)
2. **BEVERAGES** — strong year-round with holiday spikes
3. **PRODUCE** — high volume, moderate margin
4. **CLEANING** — stable demand, less seasonal
5. **DAIRY** — consistent, slight uptick in December

### Oil Price Impact
Ecuador's economy is oil-dependent. When oil prices dropped sharply (2014–2016), consumer spending declined. The `oil_price` feature captures macroeconomic sensitivity.

### Store Geography
- **Quito** and **Guayaquil** dominate sales (largest population centers)
- Type A & B stores consistently outperform Type D & E stores
- Cluster analysis shows 3 performance tiers across 54 stores

---

## 🔮 Forecast Explanation

The 30-day forecast uses **iterative recursive prediction**:
1. Start from the last known historical day
2. Build all 47 features for day `t+1` using actual history
3. Predict `t+1` sales using the trained Gradient Boosting model
4. Feed prediction back into the lag/rolling window for `t+2`
5. Repeat for all 30 days

**Confidence bounds** are set at ±10% of the point forecast, reflecting typical short-term demand uncertainty in grocery retail.

---

## 💼 Business Value

| Stakeholder | Value |
|---|---|
| **Store Manager** | Optimize inventory orders 30 days ahead; reduce stockouts & waste |
| **Supply Chain** | Better demand signals for suppliers and distribution centers |
| **Finance Team** | Revenue projections for monthly/quarterly planning |
| **Marketing** | Identify low-demand periods for promotional campaigns |
| **Operations** | Staff scheduling aligned to predicted traffic (transactions proxy) |

**Estimated ROI:** A 1% reduction in stockouts or overstock on $850M annual sales ≈ $8.5M in savings.

---

## 📊 Output Files

| File | Description |
|---|---|
| `outputs/future_predictions.csv` | 30-day forecast with upper/lower bounds |
| `outputs/metrics.json` | All model metrics + best model selection |
| `outputs/best_model.pkl` | Serialized model (Gradient Boosting + scaler) |
| `outputs/*.html` | 7 interactive Plotly charts |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas / numpy** — data wrangling
- **scikit-learn** — Linear Regression, Ridge, Random Forest, Gradient Boosting
- **XGBoost** — gradient boosted trees
- **Plotly** — interactive business charts
- **Streamlit** — web dashboard

---

*Built as a complete, production-ready ML forecasting project using real Corporación Favorita retail data.*
