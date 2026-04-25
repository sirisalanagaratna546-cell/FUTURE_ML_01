"""
run_pipeline.py
---------------
End-to-end pipeline: preprocess → engineer → train → forecast → visualize.
Run this script to reproduce all outputs from scratch.
"""

import os
import sys
import logging

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import run_preprocessing
from src.feature_engineering import build_feature_matrix
from src.train_model import train_all_models
from src.forecast import run_forecast
from src.visualize import generate_all_charts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  FAVORITA SALES FORECASTING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Data Preprocessing ─────────────────────────────
    logger.info("\n[1/5] DATA PREPROCESSING")
    daily_df, family_daily, store_daily, stores = run_preprocessing(DATA_DIR)

    # ── Step 2: Feature Engineering ────────────────────────────
    logger.info("\n[2/5] FEATURE ENGINEERING")
    feature_df = build_feature_matrix(daily_df, target_col="total_sales")

    # ── Step 3: Model Training & Evaluation ────────────────────
    logger.info("\n[3/5] MODEL TRAINING & EVALUATION")
    best_model, best_scaled, scaler, feature_cols, metrics_df, test_result, best_name = \
        train_all_models(feature_df, target_col="total_sales", output_dir=OUTPUTS_DIR)

    # ── Step 4: Future Forecast ─────────────────────────────────
    logger.info("\n[4/5] GENERATING 30-DAY FORECAST")
    forecast_df = run_forecast(daily_df, periods=30, output_dir=OUTPUTS_DIR)

    # ── Step 5: Visualizations ──────────────────────────────────
    logger.info("\n[5/5] GENERATING VISUALIZATIONS")
    generate_all_charts(
        daily_df=daily_df,
        family_daily=family_daily,
        test_result=test_result,
        forecast_df=forecast_df,
        metrics_df=metrics_df,
        best_model_name=best_name,
        output_dir=OUTPUTS_DIR,
    )

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Outputs saved to: {OUTPUTS_DIR}")
    logger.info("=" * 60)

    return {
        "daily_df": daily_df,
        "family_daily": family_daily,
        "store_daily": store_daily,
        "stores": stores,
        "feature_df": feature_df,
        "metrics_df": metrics_df,
        "test_result": test_result,
        "forecast_df": forecast_df,
        "best_name": best_name,
    }


if __name__ == "__main__":
    main()
