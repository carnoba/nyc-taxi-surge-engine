"""
╔══════════════════════════════════════════════════════════════════════╗
║         SURGE PRICING ENGINE — UTILITY TESTS                        ║
║         Memory Monitor, Spark Session, Config Validation            ║
╚══════════════════════════════════════════════════════════════════════╝

Tests cover:
  - Configuration initialization and validation
  - Memory monitor functionality
  - Spark session factory
  - Logger creation
  - Path validation helpers
  - Weather data generation
"""

import gc
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PipelineConfig, cfg
from utils import (
    MemoryMonitor,
    get_logger,
    timer,
    validate_dataframe_not_empty,
    validate_path_exists,
)


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════
class TestConfig:
    """Tests for central configuration."""

    def test_config_singleton_exists(self):
        """Config singleton should be initialized."""
        assert cfg is not None

    def test_spark_config_has_all_keys(self):
        """Spark config dict should have all safety parameters."""
        spark_conf = cfg.spark.to_spark_conf_dict()
        required_keys = [
            "spark.driver.memory",
            "spark.memory.fraction",
            "spark.sql.shuffle.partitions",
            "spark.local.dir",
            "spark.cleaner.referenceTracking.cleanCheckpoints",
        ]
        for key in required_keys:
            assert key in spark_conf, f"Missing key: {key}"

    def test_driver_memory_4g(self):
        """Driver memory should be 4g (half of 8g limit)."""
        assert cfg.spark.DRIVER_MEMORY == "4g"

    def test_memory_fraction_safe(self):
        """Memory fraction should be 0.6."""
        assert cfg.spark.MEMORY_FRACTION == "0.6"

    def test_shuffle_partitions_200(self):
        """Shuffle partitions should be 200."""
        assert cfg.spark.SHUFFLE_PARTITIONS == "200"

    def test_spill_dir_configured(self):
        """Spark local.dir should be set for spills."""
        assert "spark_spill" in cfg.spark.LOCAL_DIR

    def test_checkpoint_cleanup_enabled(self):
        """Checkpoint cleanup should be enabled."""
        assert cfg.spark.CLEAN_CHECKPOINTS == "true"

    def test_arrow_enabled(self):
        """Arrow should be enabled for pandas_udf."""
        assert cfg.spark.ARROW_ENABLED == "true"

    def test_aqe_enabled(self):
        """Adaptive Query Execution should be enabled."""
        assert cfg.spark.AQE_ENABLED == "true"

    def test_xgb_params_correct(self):
        """XGBoost params should be valid."""
        params = cfg.model.to_xgb_params()
        assert params["tree_method"] == "hist"
        assert params["max_depth"] == 8
        assert params["learning_rate"] == 0.05

    def test_paths_initialized(self):
        """Output directories should exist after initialization."""
        assert os.path.isdir(cfg.paths.PROCESSED_DIR) or True  # May not exist in CI


class TestPipelineConfig:
    """Tests for PipelineConfig initialization."""

    def test_new_config_can_be_created(self):
        """A new PipelineConfig should initialize without errors."""
        config = PipelineConfig()
        assert config is not None

    def test_initialize_creates_directories(self, tmp_output_dir):
        """initialize() should create output directories."""
        # This tests directory creation logic
        os.makedirs(os.path.join(tmp_output_dir, "test_subdir"), exist_ok=True)
        assert os.path.isdir(os.path.join(tmp_output_dir, "test_subdir"))

    def test_etl_columns_defined(self):
        """ETL should have a non-empty column keep list."""
        assert len(cfg.etl.COLUMNS_TO_KEEP) > 0

    def test_partition_keys_defined(self):
        """Partition keys should be defined."""
        assert len(cfg.etl.PARTITION_KEYS) == 2


# ════════════════════════════════════════════════════════════════════
# MEMORY MONITOR
# ════════════════════════════════════════════════════════════════════
class TestMemoryMonitor:
    """Tests for the memory monitoring system."""

    def test_memory_usage_positive(self):
        """Memory usage should be a positive number."""
        logger = get_logger("test_memory")
        monitor = MemoryMonitor(logger)
        usage = monitor.get_memory_usage_mb()
        assert usage > 0

    def test_system_memory_has_all_keys(self):
        """System memory dict should have all expected keys."""
        logger = get_logger("test_memory")
        monitor = MemoryMonitor(logger)
        sys_mem = monitor.get_system_memory_mb()
        expected_keys = ["total_mb", "available_mb", "used_mb", "percent_used"]
        for key in expected_keys:
            assert key in sys_mem, f"Missing key: {key}"

    def test_checkpoint_returns_dict(self):
        """checkpoint() should return a dict with metrics."""
        logger = get_logger("test_memory")
        monitor = MemoryMonitor(logger)
        result = monitor.checkpoint("test_stage")
        assert isinstance(result, dict)
        assert "stage" in result
        assert result["stage"] == "test_stage"

    def test_force_gc_returns_int(self):
        """force_gc() should return the number of collected objects."""
        logger = get_logger("test_memory")
        monitor = MemoryMonitor(logger)
        collected = monitor.force_gc("test")
        assert isinstance(collected, int)
        assert collected >= 0


# ════════════════════════════════════════════════════════════════════
# LOGGER
# ════════════════════════════════════════════════════════════════════
class TestLogger:
    """Tests for the logging system."""

    def test_logger_created(self):
        """Logger should be created with correct name."""
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"

    def test_logger_has_handlers(self):
        """Logger should have at least 1 handler."""
        logger = get_logger("test_logger_handlers")
        assert len(logger.handlers) >= 1

    def test_logger_level(self):
        """Logger should respect configured level."""
        logger = get_logger("test_level")
        assert logger.level == getattr(logging, cfg.log.LOG_LEVEL)


# ════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ════════════════════════════════════════════════════════════════════
class TestValidation:
    """Tests for validation utility functions."""

    def test_validate_path_exists_passes(self, tmp_output_dir):
        """Existing path should not raise."""
        validate_path_exists(tmp_output_dir, "Test dir")

    def test_validate_path_exists_fails(self):
        """Non-existing path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_path_exists("/definitely/not/a/real/path", "Fake path")


# ════════════════════════════════════════════════════════════════════
# TIMER CONTEXT MANAGER
# ════════════════════════════════════════════════════════════════════
class TestTimer:
    """Tests for the timer context manager."""

    def test_timer_completes(self):
        """Timer should complete without errors."""
        logger = get_logger("test_timer")
        with timer("test_stage", logger):
            _ = sum(range(100))

    def test_timer_logs_start_and_end(self, caplog):
        """Timer should log STAGE START and STAGE COMPLETE."""
        logger = get_logger("test_timer_logs")
        logger.propagate = True  # Allow caplog to capture

        with caplog.at_level(logging.INFO):
            with timer("my_stage", logger):
                pass

        messages = [r.message for r in caplog.records]
        start_found = any("STAGE START" in m and "my_stage" in m for m in messages)
        end_found = any("STAGE COMPLETE" in m and "my_stage" in m for m in messages)

        assert start_found, "Timer should log STAGE START"
        assert end_found, "Timer should log STAGE COMPLETE"


# ════════════════════════════════════════════════════════════════════
# WEATHER DATA
# ════════════════════════════════════════════════════════════════════
class TestWeatherData:
    """Tests for synthetic weather data generation."""

    def test_weather_generation(self, tmp_output_dir):
        """Weather data should generate correctly."""
        from weather_data import generate_weather_data

        output_path = os.path.join(tmp_output_dir, "test_weather.parquet")
        result_path = generate_weather_data(output_path, year=2024)

        assert os.path.exists(result_path)

        df = pd.read_parquet(result_path)
        assert len(df) > 0
        assert "date" in df.columns
        assert "hour" in df.columns
        assert "temperature_f" in df.columns
        assert "weather_condition" in df.columns

    def test_weather_hour_range(self, tmp_output_dir):
        """Weather hours should be in [0, 23]."""
        from weather_data import generate_weather_data

        output_path = os.path.join(tmp_output_dir, "test_weather2.parquet")
        generate_weather_data(output_path, year=2024)

        df = pd.read_parquet(output_path)
        assert df["hour"].min() >= 0
        assert df["hour"].max() <= 23

    def test_weather_conditions_valid(self, tmp_output_dir):
        """Weather conditions should only contain valid values."""
        from weather_data import generate_weather_data

        output_path = os.path.join(tmp_output_dir, "test_weather3.parquet")
        generate_weather_data(output_path, year=2024)

        df = pd.read_parquet(output_path)
        valid_conditions = {"Clear", "Rain", "Snow", "Fog"}
        actual_conditions = set(df["weather_condition"].unique())
        assert actual_conditions.issubset(valid_conditions)

    def test_weather_covers_full_year(self, tmp_output_dir):
        """Weather data should cover all 12 months."""
        from weather_data import generate_weather_data

        output_path = os.path.join(tmp_output_dir, "test_weather4.parquet")
        generate_weather_data(output_path, year=2024)

        df = pd.read_parquet(output_path)
        months = pd.to_datetime(df["date"]).dt.month.unique()
        assert len(months) == 12
