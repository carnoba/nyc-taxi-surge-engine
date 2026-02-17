"""
╔══════════════════════════════════════════════════════════════════════╗
║              SURGE PRICING ENGINE — CONFIGURATION                   ║
║         Memory-Safe, Production-Grade (8GB RAM Limit)               ║
╚══════════════════════════════════════════════════════════════════════╝

Central configuration for the NYC TLC Surge Pricing Engine.
All tunable parameters, paths, and Spark safety settings live here.

Author: Data Engineering Team
Version: 1.0.0
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration for all pipeline I/O."""

    # Project root — resolves relative to this config file
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

    # Raw data
    RAW_PARQUET_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "2024"
    )
    ZONE_LOOKUP_CSV: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "taxi+_zone_lookup.csv"
    )

    # Processed data output
    PROCESSED_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "processed"
    )
    FEATURES_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "features"
    )
    MODEL_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "models"
    )
    REPORTS_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "reports"
    )

    # Spark spill directory (HDD-backed to prevent OOM)
    # Production: "D:/Careem_Task/tmp" — change if D:/ drive is available
    SPARK_TMP_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tmp", "spark_spill"
    )

    # Weather data
    WEATHER_DATA_PATH: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weather_data.parquet"
    )

    def ensure_directories(self) -> None:
        """Create all output directories if they don't exist."""
        for attr_name in [
            "PROCESSED_DIR",
            "FEATURES_DIR",
            "MODEL_DIR",
            "REPORTS_DIR",
        ]:
            path = getattr(self, attr_name)
            os.makedirs(path, exist_ok=True)

        # Spark tmp dir
        os.makedirs(self.SPARK_TMP_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# SPARK SAFETY CONFIGURATION — THE "8GB SHIELD"
# ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SparkConfig:
    """
    Memory-Safe Spark Configuration.

    Designed for machines with ONLY 8GB RAM:
      - 4GB for Spark driver (leaves 4GB for OS/background).
      - 0.6 memory fraction (balanced storage vs execution).
      - 200 shuffle partitions (smaller partitions prevent OOM on joins).
      - HDD-backed spill directory (D:/Careem_Task/tmp).
      - Reference tracking cleanup to prevent memory leaks.
    """

    APP_NAME: str = "NYC_Surge_Pricing_Engine"

    # ── Core Memory Constraints ──
    DRIVER_MEMORY: str = "4g"
    MEMORY_FRACTION: str = "0.6"
    SHUFFLE_PARTITIONS: str = "10"

    # ── Off-Heap Memory (helps with OOM in Python 3.13) ──
    OFF_HEAP_ENABLED: str = "true"
    OFF_HEAP_SIZE: str = "2g"

    # ── Spill & Cleanup ──
    # Production: "D:/Careem_Task/tmp" — change if D:/ drive is available
    LOCAL_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tmp", "spark_spill"
    )
    CLEAN_CHECKPOINTS: str = "true"

    # ── Serialization (Kryo is faster + more compact) ──
    SERIALIZER: str = "org.apache.spark.serializer.KryoSerializer"

    # ── Broadcast Join Threshold (10MB — safe for zone lookup) ──
    BROADCAST_THRESHOLD: str = "10485760"  # 10 MB in bytes

    # ── Adaptive Query Execution (Spark 3.x) ──
    AQE_ENABLED: str = "true"
    AQE_COALESCE_PARTITIONS: str = "true"

    # ── Arrow Optimization for pandas_udf ──
    ARROW_ENABLED: str = "true"
    ARROW_FALLBACK: str = "true"

    def to_spark_conf_dict(self) -> Dict[str, str]:
        """Convert to a dict suitable for SparkSession.builder.config()."""
        return {
            "spark.app.name": self.APP_NAME,
            "spark.driver.memory": self.DRIVER_MEMORY,
            "spark.memory.fraction": self.MEMORY_FRACTION,
            "spark.sql.shuffle.partitions": self.SHUFFLE_PARTITIONS,
            "spark.memory.offHeap.enabled": self.OFF_HEAP_ENABLED,
            "spark.memory.offHeap.size": self.OFF_HEAP_SIZE,
            "spark.local.dir": self.LOCAL_DIR,
            "spark.cleaner.referenceTracking.cleanCheckpoints": self.CLEAN_CHECKPOINTS,
            "spark.serializer": self.SERIALIZER,
            "spark.sql.autoBroadcastJoinThreshold": self.BROADCAST_THRESHOLD,
            "spark.sql.adaptive.enabled": self.AQE_ENABLED,
            "spark.sql.adaptive.coalescePartitions.enabled": self.AQE_COALESCE_PARTITIONS,
            "spark.sql.execution.arrow.pyspark.enabled": self.ARROW_ENABLED,
            "spark.sql.execution.arrow.pyspark.fallback.enabled": self.ARROW_FALLBACK,
        }


# ────────────────────────────────────────────────────────────────────
# ETL CONFIGURATION
# ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ETLConfig:
    """Configuration for the ETL / data ingestion phase."""

    # Parquet file glob pattern (Filtered to January only for memory safety)
    PARQUET_GLOB: str = "yellow_tripdata_2024-01.parquet"

    # Columns to retain from raw data (reduces memory footprint)
    COLUMNS_TO_KEEP: tuple = (
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "RatecodeID",
        "payment_type",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
    )

    # Output partitioning keys
    PARTITION_KEYS: tuple = ("pickup_date", "hour")

    # Data quality thresholds
    MIN_TRIP_DISTANCE: float = 0.1        # miles
    MAX_TRIP_DISTANCE: float = 200.0      # miles
    MIN_FARE: float = 1.0                 # dollars
    MAX_FARE: float = 5000.0              # dollars
    MIN_PASSENGER_COUNT: int = 1
    MAX_PASSENGER_COUNT: int = 9

    # Output format
    OUTPUT_FORMAT: str = "parquet"
    COMPRESSION: str = "snappy"


# ────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING CONFIGURATION
# ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for geospatial and temporal feature engineering."""

    # H3 Resolution (8 ≈ ~460m edge length — city-block level)
    H3_RESOLUTION: int = 8

    # Rolling window sizes (in minutes)
    ROLLING_WINDOWS_MINUTES: tuple = (15, 30, 60)

    # Demand/Supply ratio configuration
    SUPPLY_HISTORICAL_DAYS: int = 28  # 4-week lookback for supply baseline

    # Surge multiplier bounds
    MIN_SURGE: float = 1.0
    MAX_SURGE: float = 5.0

    # Time features to extract
    TIME_FEATURES: tuple = (
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "is_rush_hour",
    )

    # Peak hour ranges
    MORNING_RUSH: tuple = (7, 10)   # 7 AM — 10 AM
    EVENING_RUSH: tuple = (16, 20)  # 4 PM — 8 PM

    # pandas_udf batch size (rows per Arrow batch)
    ARROW_BATCH_SIZE: int = 10000


# ────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """Configuration for XGBoost surge prediction model."""

    # ── XGBoost Core Params ──
    TREE_METHOD: str = "hist"               # Memory-optimized histogram method
    MAX_DEPTH: int = 8
    LEARNING_RATE: float = 0.05
    N_ESTIMATORS: int = 500
    SUBSAMPLE: float = 0.8
    COLSAMPLE_BYTREE: float = 0.8
    REG_ALPHA: float = 0.1                 # L1 regularization
    REG_LAMBDA: float = 1.0                # L2 regularization
    MIN_CHILD_WEIGHT: int = 5
    GAMMA: float = 0.1

    # ── Custom Loss Params ──
    UNDERESTIMATION_PENALTY: float = 2.0   # 2x penalty for under-predicting surge

    # ── Training Strategy ──
    EARLY_STOPPING_ROUNDS: int = 50
    EVAL_METRIC: str = "rmse"
    VALIDATION_SPLIT: float = 0.2
    RANDOM_STATE: int = 42

    # ── Incremental Training ──
    BATCH_SIZE: int = 500_000              # Rows per DMatrix batch
    N_BOOST_ROUNDS_PER_BATCH: int = 10

    # ── Feature columns (populated dynamically) ──
    # These are set at runtime based on the engineered features
    TARGET_COLUMN: str = "surge_multiplier"

    def to_xgb_params(self) -> Dict:
        """Convert to an XGBoost-compatible parameter dict."""
        return {
            "tree_method": self.TREE_METHOD,
            "max_depth": self.MAX_DEPTH,
            "learning_rate": self.LEARNING_RATE,
            "subsample": self.SUBSAMPLE,
            "colsample_bytree": self.COLSAMPLE_BYTREE,
            "reg_alpha": self.REG_ALPHA,
            "reg_lambda": self.REG_LAMBDA,
            "min_child_weight": self.MIN_CHILD_WEIGHT,
            "gamma": self.GAMMA,
            "eval_metric": self.EVAL_METRIC,
            "random_state": self.RANDOM_STATE,
            "verbosity": 1,
        }


# ────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LogConfig:
    """Logging and monitoring configuration."""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = (
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    )
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Memory reporting thresholds (MB)
    MEMORY_WARNING_THRESHOLD_MB: int = 6144   # 6 GB — 75% of 8 GB
    MEMORY_CRITICAL_THRESHOLD_MB: int = 7168  # 7 GB — 87.5% of 8 GB


# ────────────────────────────────────────────────────────────────────
# MASTER CONFIG — Single access point
# ────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """
    Master configuration object.

    Usage:
        from config import cfg
        spark_params = cfg.spark.to_spark_conf_dict()
        raw_path = cfg.paths.RAW_PARQUET_DIR
    """

    paths: PathConfig = field(default_factory=PathConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    etl: ETLConfig = field(default_factory=ETLConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def initialize(self) -> "PipelineConfig":
        """Run startup checks and create directories."""
        self.paths.ensure_directories()
        return self


# ── Singleton instance ──
cfg = PipelineConfig().initialize()


if __name__ == "__main__":
    # Quick sanity check
    print("=" * 60)
    print("  SURGE PRICING ENGINE — CONFIG SUMMARY")
    print("=" * 60)
    print(f"  Project Root     : {cfg.paths.PROJECT_ROOT}")
    print(f"  Raw Data Dir     : {cfg.paths.RAW_PARQUET_DIR}")
    print(f"  Spark Driver Mem : {cfg.spark.DRIVER_MEMORY}")
    print(f"  Shuffle Parts    : {cfg.spark.SHUFFLE_PARTITIONS}")
    print(f"  Spill Dir        : {cfg.spark.LOCAL_DIR}")
    print(f"  H3 Resolution    : {cfg.features.H3_RESOLUTION}")
    print(f"  XGBoost Method   : {cfg.model.TREE_METHOD}")
    print(f"  Under-est Penalty: {cfg.model.UNDERESTIMATION_PENALTY}x")
    print("=" * 60)
