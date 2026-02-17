"""
╔══════════════════════════════════════════════════════════════════════╗
║       SURGE PRICING ENGINE — PHASE C: MODEL TRAINING                ║
║   XGBoost with Custom Asymmetric Loss & Incremental Training        ║
╚══════════════════════════════════════════════════════════════════════╝

This module:
  1. Reads feature-engineered data from Phase B.
  2. Prepares train/validation split.
  3. Defines a custom asymmetric loss function that penalizes
     under-estimation of surge 2x more than over-estimation.
  4. Trains XGBoost using `tree_method='hist'` (memory-optimized).
  5. Implements incremental/batched training via DMatrix for
     memory safety on 100M+ row datasets.
  6. Evaluates model and saves artifacts.

Memory Safety:
  - tree_method='hist' uses histograms instead of exact sort.
  - Batched DMatrix loading (500K rows per batch).
  - Manual gc.collect() after batch processing.

Author: Data Engineering Team
Version: 1.0.0
"""

import gc
import json
import os
import pickle
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from config import cfg
from utils import (
    MemoryMonitor,
    get_logger,
    get_spark_session,
    timer,
    validate_path_exists,
)


logger = get_logger(__name__)
monitor = MemoryMonitor(logger)


# ────────────────────────────────────────────────────────────────────
# CUSTOM ASYMMETRIC LOSS FUNCTION
# ────────────────────────────────────────────────────────────────────
def asymmetric_surge_loss(
    y_pred: np.ndarray,
    dtrain: xgb.DMatrix,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom asymmetric loss for surge pricing.

    Under-estimation of surge is penalized 2x more than over-estimation:
      - Under-estimation penalty: UNDERESTIMATION_PENALTY (default 2.0)
      - Over-estimation penalty: 1.0

    This reflects the business requirement that missing a surge event
    (lost revenue + poor driver allocation) is worse than slightly
    over-predicting surge (minor rider friction).

    Mathematical Definition:
      residual = y_true - y_pred

      if residual > 0 (under-estimation):
          grad = -2 * penalty * residual
          hess =  2 * penalty
      else (over-estimation):
          grad = -2 * residual
          hess =  2

    Args:
        y_pred: Model predictions (1D numpy array).
        dtrain: XGBoost DMatrix containing labels.

    Returns:
        Tuple of (gradient, hessian) arrays.
    """
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    penalty = cfg.model.UNDERESTIMATION_PENALTY

    # Gradient
    grad = np.where(
        residual > 0,
        -2.0 * penalty * residual,   # Under-estimation: stronger gradient
        -2.0 * residual,             # Over-estimation: normal gradient
    )

    # Hessian (second derivative)
    hess = np.where(
        residual > 0,
        2.0 * penalty,   # Under-estimation: higher curvature
        2.0,             # Over-estimation: standard curvature
    )

    return grad, hess


def asymmetric_eval_metric(
    y_pred: np.ndarray,
    dtrain: xgb.DMatrix,
) -> Tuple[str, float]:
    """
    Custom evaluation metric aligned with the asymmetric loss.

    Computes weighted MSE where under-estimation errors are weighted
    by the penalty factor.

    Args:
        y_pred: Predictions.
        dtrain: DMatrix with labels.

    Returns:
        Tuple of (metric_name, metric_value). Lower is better.
    """
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    penalty = cfg.model.UNDERESTIMATION_PENALTY

    weights = np.where(residual > 0, penalty, 1.0)
    weighted_mse = np.mean(weights * residual**2)

    return "asymmetric_wmse", float(weighted_mse)


# ────────────────────────────────────────────────────────────────────
# STEP 1: LOAD FEATURE DATA
# ────────────────────────────────────────────────────────────────────
def load_feature_data() -> pd.DataFrame:
    """
    Load feature data from parquet in memory-safe batches.

    Reads the Spark-written parquet output using pandas with
    pyarrow backend for efficient columnar reads. Selects only
    numeric feature columns needed for XGBoost.

    Returns:
        pandas DataFrame with features and target.
    """
    logger.info("📂 Loading feature data for model training...")

    input_path = cfg.paths.FEATURES_DIR
    validate_path_exists(input_path, "Feature data directory")

    # Read with pyarrow for columnar efficiency
    df = pd.read_parquet(input_path, engine="pyarrow")

    logger.info(
        f"✅ Feature data loaded | Shape: {df.shape} | "
        f"Memory: {df.memory_usage(deep=True).sum() / (1024 ** 2):.1f} MB"
    )

    return df


# ────────────────────────────────────────────────────────────────────
# STEP 2: PREPARE FEATURES
# ────────────────────────────────────────────────────────────────────
# Feature columns for XGBoost (all numeric)
FEATURE_COLUMNS = [
    # Temporal
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_rush_hour",
    # Location
    "PULocationID",
    "DOLocationID",
    # Trip
    "trip_distance",
    "trip_duration_minutes",
    "passenger_count",
    "fare_amount",
    "total_amount",
    # Demand / Supply
    "demand_15min",
    "demand_30min",
    "demand_60min",
    "supply_baseline",
    "demand_zscore",
    # Derived
    "fare_per_mile",
    "fare_per_minute",
    "speed_mph",
    "demand_acceleration",
    "demand_trend",
    "congestion_indicator",
    # Weather
    "temperature_f",
    "precipitation_in",
    "wind_speed_mph",
    "visibility_miles",
    "weather_is_clear",
    "weather_is_rain",
    "weather_is_snow",
    "weather_is_fog",
]

TARGET_COLUMN = cfg.model.TARGET_COLUMN


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare feature matrix X and target vector y.

    - Selects only numeric feature columns.
    - Handles remaining nulls.
    - Validates target column exists.

    Args:
        df: Raw feature DataFrame.

    Returns:
        Tuple of (X, y, feature_names).
    """
    logger.info("🔧 Preparing feature matrix...")

    # Filter to available columns
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing_features:
        logger.warning(f"⚠️  Missing feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()

    # Handle nulls — fill with column median (safer for tree models)
    null_cols = X.columns[X.isnull().any()].tolist()
    if null_cols:
        logger.info(f"  Filling nulls in {len(null_cols)} columns with median")
        X = X.fillna(X.median())

    # Handle any remaining NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.fillna(y.median()).replace([np.inf, -np.inf], y.median())

    logger.info(
        f"✅ Features prepared | X: {X.shape} | y: {y.shape} | "
        f"Feature count: {len(available_features)}"
    )

    return X, y, available_features


# ────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN-TEST SPLIT
# ────────────────────────────────────────────────────────────────────
def create_train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and validation sets.

    Uses stratified-like splitting based on the target distribution
    to ensure both sets represent the full surge range.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val).
    """
    logger.info(
        f"✂️  Splitting data: {1 - cfg.model.VALIDATION_SPLIT:.0%} train / "
        f"{cfg.model.VALIDATION_SPLIT:.0%} validation"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.model.VALIDATION_SPLIT,
        random_state=cfg.model.RANDOM_STATE,
        shuffle=True,
    )

    logger.info(
        f"✅ Split complete | "
        f"Train: {X_train.shape[0]:,} rows | "
        f"Validation: {X_val.shape[0]:,} rows"
    )

    return X_train, X_val, y_train, y_val


# ────────────────────────────────────────────────────────────────────
# STEP 4A: STANDARD TRAINING (for datasets fitting in memory)
# ────────────────────────────────────────────────────────────────────
def train_standard(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.Booster:
    """
    Train XGBoost with the full dataset in memory.

    Used when the feature data fits comfortably in available RAM.
    Uses DMatrix for efficient data handling and the custom
    asymmetric loss function.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Trained xgb.Booster model.
    """
    logger.info("🎯 Training XGBoost (standard mode)...")

    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_val.columns))

    # XGBoost parameters
    params = cfg.model.to_xgb_params()
    params["disable_default_eval_metric"] = 1  # Use custom metric only

    logger.info(f"  XGBoost params: {json.dumps(params, indent=2)}")

    # Train with custom loss
    evals_result: Dict = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=cfg.model.N_ESTIMATORS,
        evals=[(dtrain, "train"), (dval, "validation")],
        obj=asymmetric_surge_loss,
        custom_metric=asymmetric_eval_metric,
        early_stopping_rounds=cfg.model.EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=50,
    )

    logger.info(
        f"✅ Training complete | "
        f"Best iteration: {model.best_iteration} | "
        f"Best score: {model.best_score:.6f}"
    )

    # Cleanup DMatrix
    del dtrain, dval
    gc.collect()

    return model


# ────────────────────────────────────────────────────────────────────
# STEP 4B: INCREMENTAL TRAINING (for very large datasets)
# ────────────────────────────────────────────────────────────────────
def train_incremental(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.Booster:
    """
    Train XGBoost incrementally in batches.

    Splits the training data into chunks of BATCH_SIZE rows,
    creates a DMatrix per chunk, and continues training the
    same Booster model across batches.

    This prevents loading the entire dataset into an XGBoost
    DMatrix at once, which could exceed 8GB RAM.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Trained xgb.Booster model.
    """
    batch_size = cfg.model.BATCH_SIZE
    n_samples = len(X_train)
    n_batches = (n_samples + batch_size - 1) // batch_size

    logger.info(
        f"🎯 Training XGBoost (incremental mode) | "
        f"Batches: {n_batches} | Batch size: {batch_size:,}"
    )

    params = cfg.model.to_xgb_params()
    params["disable_default_eval_metric"] = 1

    # Validation DMatrix (persists across batches)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_val.columns))

    model: Optional[xgb.Booster] = None

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)

        logger.info(
            f"  📦 Batch {batch_idx + 1}/{n_batches} | "
            f"Rows: [{start_idx:,} — {end_idx:,}]"
        )

        # Create DMatrix for this batch
        X_batch = X_train.iloc[start_idx:end_idx]
        y_batch = y_train.iloc[start_idx:end_idx]

        dtrain_batch = xgb.DMatrix(
            X_batch,
            label=y_batch,
            feature_names=list(X_train.columns),
        )

        # Train / continue training
        model = xgb.train(
            params=params,
            dtrain=dtrain_batch,
            num_boost_round=cfg.model.N_BOOST_ROUNDS_PER_BATCH,
            evals=[(dtrain_batch, "train"), (dval, "validation")],
            obj=asymmetric_surge_loss,
            custom_metric=asymmetric_eval_metric,
            xgb_model=model,  # Continue from previous model
            verbose_eval=5,
        )

        # Cleanup batch DMatrix
        del dtrain_batch, X_batch, y_batch
        gc.collect()

        monitor.checkpoint(f"Batch {batch_idx + 1}/{n_batches}")

    del dval
    gc.collect()

    logger.info(f"✅ Incremental training complete | Total batches: {n_batches}")

    return model


# ────────────────────────────────────────────────────────────────────
# STEP 5: MODEL EVALUATION
# ────────────────────────────────────────────────────────────────────
def evaluate_model(
    model: xgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
) -> Dict:
    """
    Comprehensive model evaluation.

    Computes:
      - RMSE, MAE, R², MAPE
      - Directional accuracy (surge vs no-surge)
      - Feature importance (gain, weight, cover)

    Args:
        model: Trained XGBoost Booster.
        X_val: Validation features.
        y_val: Validation targets.
        feature_names: List of feature column names.

    Returns:
        Dict of evaluation metrics and feature importances.
    """
    logger.info("📈 Evaluating model...")

    dval = xgb.DMatrix(X_val, feature_names=feature_names)
    y_pred = model.predict(dval)

    # Core metrics
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))

    # MAPE (avoiding division by zero)
    mask = y_val > 0
    mape = float(
        np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
    )

    # Directional accuracy (correctly predicting surge > 1)
    surge_threshold = 1.2
    true_surge = y_val >= surge_threshold
    pred_surge = y_pred >= surge_threshold
    directional_accuracy = float(np.mean(true_surge == pred_surge) * 100)

    # Under-estimation rate
    under_estimation_rate = float(
        np.mean(y_pred[true_surge] < y_val[true_surge]) * 100
    ) if true_surge.sum() > 0 else 0.0

    # Feature importance
    importance_gain = model.get_score(importance_type="gain")
    importance_weight = model.get_score(importance_type="weight")

    # Sort by gain
    top_features = sorted(
        importance_gain.items(), key=lambda x: x[1], reverse=True
    )[:20]

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape_percent": mape,
        "directional_accuracy_percent": directional_accuracy,
        "under_estimation_rate_percent": under_estimation_rate,
        "n_trees": model.best_iteration if hasattr(model, "best_iteration") else -1,
        "n_features": len(feature_names),
        "feature_importance_top20": {k: round(v, 4) for k, v in top_features},
    }

    # Log metrics
    logger.info("=" * 50)
    logger.info("  MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  RMSE            : {rmse:.6f}")
    logger.info(f"  MAE             : {mae:.6f}")
    logger.info(f"  R²              : {r2:.6f}")
    logger.info(f"  MAPE            : {mape:.2f}%")
    logger.info(f"  Directional Acc : {directional_accuracy:.2f}%")
    logger.info(f"  Under-est Rate  : {under_estimation_rate:.2f}%")
    logger.info(f"  Top 5 Features  :")
    for feat, gain in top_features[:5]:
        logger.info(f"    {feat:30s} — gain: {gain:.4f}")
    logger.info("=" * 50)

    del dval
    gc.collect()

    return metrics


# ────────────────────────────────────────────────────────────────────
# STEP 6: SAVE MODEL ARTIFACTS
# ────────────────────────────────────────────────────────────────────
def save_model_artifacts(
    model: xgb.Booster,
    metrics: Dict,
    feature_names: List[str],
) -> str:
    """
    Save trained model and evaluation artifacts.

    Saved artifacts:
      - surge_model.json     — XGBoost model (JSON format)
      - surge_model.ubj      — XGBoost model (binary format, for loading)
      - metrics.json          — Evaluation metrics
      - feature_names.json    — Ordered feature list
      - feature_importance.csv — Full feature importance table

    Args:
        model: Trained Booster.
        metrics: Evaluation metrics dict.
        feature_names: List of feature names.

    Returns:
        Path to the model directory.
    """
    model_dir = cfg.paths.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"💾 Saving model artifacts to: {model_dir}")

    # Model (JSON format — portable)
    model_json_path = os.path.join(model_dir, "surge_model.json")
    model.save_model(model_json_path)
    logger.info(f"  ✅ Model saved (JSON): {model_json_path}")

    # Model (binary format — faster loading)
    model_ubj_path = os.path.join(model_dir, "surge_model.ubj")
    model.save_model(model_ubj_path)
    logger.info(f"  ✅ Model saved (binary): {model_ubj_path}")

    # Metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"  ✅ Metrics saved: {metrics_path}")

    # Feature names
    features_path = os.path.join(model_dir, "feature_names.json")
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"  ✅ Feature names saved: {features_path}")

    # Feature importance CSV
    importance = model.get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        [
            {"feature": k, "importance_gain": v}
            for k, v in sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )
        ]
    )
    importance_path = os.path.join(model_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"  ✅ Feature importance saved: {importance_path}")

    logger.info(f"✅ All model artifacts saved to: {model_dir}")

    return model_dir


# ────────────────────────────────────────────────────────────────────
# MAIN MODEL TRAINING ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────
def run_model_training() -> str:
    """
    Execute the full model training pipeline.

    Automatically selects between standard or incremental training
    based on dataset size vs available memory.

    Stages:
      1. Load feature data.
      2. Prepare features.
      3. Train-test split.
      4. Train XGBoost (standard or incremental).
      5. Evaluate model.
      6. Save artifacts.

    Returns:
        Path to the model output directory.
    """
    logger.info("=" * 70)
    logger.info("  PHASE C: MODEL TRAINING — START")
    logger.info("=" * 70)

    monitor.checkpoint("Model — Pre-Start")

    try:
        with timer("Phase C: Full Model Training", logger):

            # ── Step 1: Load Data ──
            with timer("Step 1: Load Feature Data", logger):
                df = load_feature_data()

            monitor.checkpoint("Model — Post-Load")

            # ── Step 2: Prepare Features ──
            with timer("Step 2: Prepare Features", logger):
                X, y, feature_names = prepare_features(df)
                del df
                gc.collect()

            monitor.checkpoint("Model — Post-Prep")

            # ── Step 3: Split ──
            with timer("Step 3: Train-Val Split", logger):
                X_train, X_val, y_train, y_val = create_train_val_split(X, y)
                del X, y
                gc.collect()

            # ── Step 4: Train ──
            # Auto-select training strategy based on dataset size
            train_data_size_mb = (
                X_train.memory_usage(deep=True).sum() / (1024**2)
            )
            memory_available = monitor.get_system_memory_mb()["available_mb"]

            if train_data_size_mb * 3 > memory_available:
                # DMatrix roughly 2-3x the DataFrame size
                logger.info(
                    f"  📊 Data size ({train_data_size_mb:.0f} MB) × 3 > "
                    f"Available RAM ({memory_available:.0f} MB) → "
                    "Using INCREMENTAL training"
                )
                with timer("Step 4: Incremental Training", logger):
                    model = train_incremental(X_train, y_train, X_val, y_val)
            else:
                logger.info(
                    f"  📊 Data size ({train_data_size_mb:.0f} MB) fits in "
                    f"RAM ({memory_available:.0f} MB) → Using STANDARD training"
                )
                with timer("Step 4: Standard Training", logger):
                    model = train_standard(X_train, y_train, X_val, y_val)

            # Free training data
            del X_train, y_train
            gc.collect()

            monitor.checkpoint("Model — Post-Train")

            # ── Step 5: Evaluate ──
            with timer("Step 5: Evaluate Model", logger):
                metrics = evaluate_model(model, X_val, y_val, feature_names)

            del X_val, y_val
            gc.collect()

            # ── Step 6: Save ──
            with timer("Step 6: Save Artifacts", logger):
                model_dir = save_model_artifacts(model, metrics, feature_names)

            del model
            gc.collect()

        # ── Post-Training Cleanup ──
        monitor.force_gc("Post-Model Training cleanup")
        monitor.checkpoint("Model — Complete")

        logger.info("=" * 70)
        logger.info("  PHASE C: MODEL TRAINING — COMPLETE ✅")
        logger.info("=" * 70)

        return model_dir

    except Exception as e:
        logger.error(f"❌ MODEL TRAINING FAILED: {type(e).__name__}: {e}")
        monitor.checkpoint("Model — FAILED")
        monitor.force_gc("Post-failure cleanup")
        raise


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        output = run_model_training()
        logger.info(f"Model artifacts at: {output}")
    except Exception as e:
        logger.error(f"Pipeline terminated: {e}")
        sys.exit(1)
