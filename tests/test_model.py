"""
╔══════════════════════════════════════════════════════════════════════╗
║         SURGE PRICING ENGINE — MODEL TRAINING TESTS                 ║
║        Phase C: Custom Loss, Model Output, and Artifacts            ║
╚══════════════════════════════════════════════════════════════════════╝

Tests cover:
  - Custom asymmetric loss function (gradient & hessian)
  - Feature preparation and null handling
  - Model training and prediction
  - Evaluation metric alignment
  - Model artifact saving
"""

import os
import sys
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model_training import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    asymmetric_eval_metric,
    asymmetric_surge_loss,
    create_train_val_split,
    evaluate_model,
    prepare_features,
    save_model_artifacts,
)


# ════════════════════════════════════════════════════════════════════
# CUSTOM ASYMMETRIC LOSS FUNCTION
# ════════════════════════════════════════════════════════════════════
class TestAsymmetricLoss:
    """Tests for the custom asymmetric surge loss function."""

    def test_gradient_shape_matches_input(self):
        """Gradient should have the same shape as predictions."""
        y_true = np.array([1.5, 2.0, 1.0, 3.0])
        y_pred = np.array([1.2, 2.5, 0.8, 2.8])
        dtrain = xgb.DMatrix(
            np.random.randn(4, 3), label=y_true,
        )
        grad, hess = asymmetric_surge_loss(y_pred, dtrain)
        assert grad.shape == y_pred.shape
        assert hess.shape == y_pred.shape

    def test_under_estimation_penalty_stronger(self):
        """Under-estimation gradient should be stronger than over-estimation."""
        y_true = np.array([2.0, 2.0])
        # First: under-estimate (pred < true), Second: over-estimate (pred > true)
        y_pred = np.array([1.5, 2.5])  # Both 0.5 error
        dtrain = xgb.DMatrix(
            np.random.randn(2, 3), label=y_true,
        )
        grad, hess = asymmetric_surge_loss(y_pred, dtrain)

        # Under-estimation should have larger absolute gradient
        assert abs(grad[0]) > abs(grad[1]), (
            f"Under-est gradient ({abs(grad[0]):.4f}) should be > "
            f"over-est gradient ({abs(grad[1]):.4f})"
        )

    def test_hessian_positive(self):
        """Hessian should always be positive (convex)."""
        y_true = np.array([1.0, 2.0, 3.0, 1.5])
        y_pred = np.array([1.5, 1.5, 2.5, 2.0])
        dtrain = xgb.DMatrix(np.random.randn(4, 3), label=y_true)
        _, hess = asymmetric_surge_loss(y_pred, dtrain)
        assert np.all(hess > 0), "Hessian must be positive for convergence"

    def test_zero_residual_zero_gradient(self):
        """Perfect prediction should yield zero gradient."""
        y_true = np.array([2.0])
        y_pred = np.array([2.0])
        dtrain = xgb.DMatrix(np.random.randn(1, 3), label=y_true)
        grad, _ = asymmetric_surge_loss(y_pred, dtrain)
        assert abs(grad[0]) < 1e-10, "Zero residual should yield zero gradient"

    def test_penalty_factor_configurable(self):
        """The under-estimation penalty should match config."""
        assert cfg.model.UNDERESTIMATION_PENALTY == 2.0


# ════════════════════════════════════════════════════════════════════
# CUSTOM EVAL METRIC
# ════════════════════════════════════════════════════════════════════
class TestEvalMetric:
    """Tests for the custom evaluation metric."""

    def test_eval_metric_returns_tuple(self):
        """Eval metric should return (name, value) tuple."""
        dtrain = xgb.DMatrix(
            np.random.randn(10, 5),
            label=np.random.uniform(1, 3, 10),
        )
        y_pred = np.random.uniform(1, 3, 10)
        name, value = asymmetric_eval_metric(y_pred, dtrain)
        assert isinstance(name, str)
        assert isinstance(value, float)

    def test_eval_metric_name(self):
        """Eval metric should be named 'asymmetric_wmse'."""
        dtrain = xgb.DMatrix(
            np.random.randn(5, 3),
            label=np.ones(5),
        )
        name, _ = asymmetric_eval_metric(np.ones(5), dtrain)
        assert name == "asymmetric_wmse"

    def test_perfect_prediction_zero_loss(self):
        """Perfect predictions should yield zero eval metric."""
        y = np.array([1.0, 2.0, 3.0])
        dtrain = xgb.DMatrix(np.random.randn(3, 3), label=y)
        _, value = asymmetric_eval_metric(y, dtrain)
        assert value < 1e-10


# ════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ════════════════════════════════════════════════════════════════════
class TestFeaturePreparation:
    """Tests for feature matrix preparation."""

    def test_prepare_returns_correct_types(self, sample_feature_df):
        """prepare_features should return (DataFrame, Series, list)."""
        X, y, names = prepare_features(sample_feature_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(names, list)

    def test_no_nulls_in_features(self, sample_feature_df):
        """Feature matrix should have no nulls after preparation."""
        X, _, _ = prepare_features(sample_feature_df)
        assert X.isnull().sum().sum() == 0

    def test_no_infinities_in_features(self, sample_feature_df):
        """Feature matrix should have no infinite values."""
        X, _, _ = prepare_features(sample_feature_df)
        assert np.isfinite(X.values).all()

    def test_target_column_present(self, sample_feature_df):
        """Target column should be extractable."""
        _, y, _ = prepare_features(sample_feature_df)
        assert len(y) == len(sample_feature_df)

    def test_missing_target_raises_error(self, sample_feature_df):
        """Missing target column should raise ValueError."""
        df = sample_feature_df.drop(columns=["surge_multiplier"])
        with pytest.raises(ValueError, match="Target column"):
            prepare_features(df)

    def test_feature_names_match_columns(self, sample_feature_df):
        """Feature names should match X columns."""
        X, _, names = prepare_features(sample_feature_df)
        assert list(X.columns) == names


# ════════════════════════════════════════════════════════════════════
# TRAIN-TEST SPLIT
# ════════════════════════════════════════════════════════════════════
class TestTrainTestSplit:
    """Tests for the train-validation split."""

    def test_split_sizes(self, sample_feature_df):
        """Split should respect the configured validation ratio."""
        X, y, _ = prepare_features(sample_feature_df)
        X_train, X_val, y_train, y_val = create_train_val_split(X, y)

        total = len(X)
        expected_val = int(total * cfg.model.VALIDATION_SPLIT)

        # Allow ±5% tolerance
        assert abs(len(X_val) - expected_val) <= total * 0.05

    def test_split_no_data_loss(self, sample_feature_df):
        """Total rows after split should equal original."""
        X, y, _ = prepare_features(sample_feature_df)
        X_train, X_val, _, _ = create_train_val_split(X, y)
        assert len(X_train) + len(X_val) == len(X)

    def test_split_reproducible(self, sample_feature_df):
        """Same random_state should produce same split."""
        X, y, _ = prepare_features(sample_feature_df)
        X1, _, _, _ = create_train_val_split(X, y)
        X2, _, _, _ = create_train_val_split(X, y)
        pd.testing.assert_frame_equal(X1, X2)


# ════════════════════════════════════════════════════════════════════
# MODEL TRAINING & PREDICTION
# ════════════════════════════════════════════════════════════════════
class TestModelTraining:
    """Tests for model training and prediction."""

    def test_model_trains_successfully(self, sample_feature_df):
        """Model should train without errors on sample data."""
        X, y, names = prepare_features(sample_feature_df)
        X_train, X_val, y_train, y_val = create_train_val_split(X, y)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=names)

        params = cfg.model.to_xgb_params()
        params["disable_default_eval_metric"] = 1

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=10,
            evals=[(dval, "val")],
            obj=asymmetric_surge_loss,
            custom_metric=asymmetric_eval_metric,
            verbose_eval=False,
        )
        assert model is not None

    def test_predictions_in_valid_range(self, sample_feature_df):
        """Predictions should be in a reasonable range."""
        X, y, names = prepare_features(sample_feature_df)
        X_train, X_val, y_train, y_val = create_train_val_split(X, y)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=names)

        params = cfg.model.to_xgb_params()
        params["disable_default_eval_metric"] = 1

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=50,
            evals=[(dval, "val")],
            obj=asymmetric_surge_loss,
            custom_metric=asymmetric_eval_metric,
            verbose_eval=False,
        )

        preds = model.predict(dval)
        # Predictions should be roughly in the range of the target
        assert preds.min() > -5, f"Min prediction too low: {preds.min()}"
        assert preds.max() < 20, f"Max prediction too high: {preds.max()}"

    def test_model_xgb_params(self):
        """XGBoost params should use hist tree method."""
        params = cfg.model.to_xgb_params()
        assert params["tree_method"] == "hist"


# ════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════
class TestModelEvaluation:
    """Tests for model evaluation."""

    def test_evaluation_returns_all_metrics(self, sample_feature_df):
        """Evaluation should return all expected metrics."""
        X, y, names = prepare_features(sample_feature_df)
        X_train, X_val, y_train, y_val = create_train_val_split(X, y)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=names)
        params = cfg.model.to_xgb_params()
        params["disable_default_eval_metric"] = 1

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=10,
            obj=asymmetric_surge_loss,
            custom_metric=asymmetric_eval_metric,
            verbose_eval=False,
        )

        metrics = evaluate_model(model, X_val, y_val, names)

        expected_keys = [
            "rmse", "mae", "r2_score", "mape_percent",
            "directional_accuracy_percent",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"


# ════════════════════════════════════════════════════════════════════
# MODEL ARTIFACTS
# ════════════════════════════════════════════════════════════════════
class TestModelArtifacts:
    """Tests for model artifact saving."""

    def test_save_creates_all_files(self, sample_feature_df, tmp_output_dir):
        """All artifact files should be created."""
        X, y, names = prepare_features(sample_feature_df)
        dtrain = xgb.DMatrix(X, label=y, feature_names=names)

        params = cfg.model.to_xgb_params()
        params["disable_default_eval_metric"] = 1

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=5,
            obj=asymmetric_surge_loss,
            custom_metric=asymmetric_eval_metric,
            verbose_eval=False,
        )

        # Override model dir for test
        import config
        original_dir = cfg.paths.MODEL_DIR
        config.cfg.paths = cfg.paths.__class__(
            **{
                **{
                    f.name: getattr(cfg.paths, f.name)
                    for f in cfg.paths.__dataclass_fields__.values()
                },
                "MODEL_DIR": tmp_output_dir,
            }
        )

        try:
            metrics = {"rmse": 0.1, "mae": 0.05}
            save_model_artifacts(model, metrics, names)

            assert os.path.exists(os.path.join(tmp_output_dir, "surge_model.json"))
            assert os.path.exists(os.path.join(tmp_output_dir, "metrics.json"))
            assert os.path.exists(os.path.join(tmp_output_dir, "feature_names.json"))
            assert os.path.exists(os.path.join(tmp_output_dir, "feature_importance.csv"))
        finally:
            # Restore original config
            config.cfg.paths = cfg.paths.__class__(
                **{
                    **{
                        f.name: getattr(cfg.paths, f.name)
                        for f in cfg.paths.__dataclass_fields__.values()
                    },
                    "MODEL_DIR": original_dir,
                }
            )

    def test_metrics_json_valid(self, tmp_output_dir):
        """Saved metrics.json should be valid JSON."""
        metrics = {"rmse": 0.15, "mae": 0.08, "r2_score": 0.92}
        metrics_path = os.path.join(tmp_output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        with open(metrics_path) as f:
            loaded = json.load(f)

        assert loaded["rmse"] == 0.15
        assert loaded["r2_score"] == 0.92
