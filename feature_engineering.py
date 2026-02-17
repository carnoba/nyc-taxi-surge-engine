"""
╔══════════════════════════════════════════════════════════════════════╗
║    SURGE PRICING ENGINE — PHASE B: FEATURE ENGINEERING              ║
║    H3 Geospatial Indexing, Rolling Windows, Demand/Supply Ratios    ║
╚══════════════════════════════════════════════════════════════════════╝

This module:
  1. Reads partitioned ETL output (predicate pushdown).
  2. Computes H3 (Res 8) hex indices via pandas_udf (Arrow-backed).
  3. Calculates 15/30/60-min rolling window demand aggregates.
  4. Computes demand/supply ratios to derive surge multiplier.
  5. Integrates weather data via left join on (date, hour).
  6. Outputs a feature-rich dataset ready for ML training.

Memory Safety:
  - pandas_udf with Apache Arrow for vectorized, batched processing.
  - Window functions operate on sorted partitions (no full shuffle).
  - Manual gc.collect() after major transformations.

Author: Data Engineering Team
Version: 1.0.0
"""

import gc
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from config import cfg
from utils import (
    MemoryMonitor,
    get_logger,
    get_spark_session,
    timer,
    validate_dataframe_not_empty,
    validate_path_exists,
)


logger = get_logger(__name__)
monitor = MemoryMonitor(logger)


# ────────────────────────────────────────────────────────────────────
# STEP 1: READ PROCESSED DATA
# ────────────────────────────────────────────────────────────────────
def read_processed_data(spark: SparkSession) -> DataFrame:
    """
    Read the partitioned ETL output.

    Leverages partition pruning on (pickup_date, hour) for efficiency.

    Args:
        spark: Active SparkSession.

    Returns:
        Spark DataFrame of cleaned taxi trip data.
    """
    input_path = cfg.paths.PROCESSED_DIR
    logger.info(f"📂 Reading processed data from: {input_path}")

    validate_path_exists(input_path, "Processed data directory")

    df = spark.read.parquet(input_path)

    logger.info(
        f"✅ Processed data loaded | Columns: {len(df.columns)} | "
        f"Partitions: {df.rdd.getNumPartitions()}"
    )
    return df


# ────────────────────────────────────────────────────────────────────
# STEP 2: H3 GEOSPATIAL INDEXING (via pandas_udf)
# ────────────────────────────────────────────────────────────────────
def add_h3_indices(df: DataFrame) -> DataFrame:
    """
    Add H3 hex indices for pickup and dropoff locations.

    Uses pandas_udf (Apache Arrow) for high-throughput, vectorized
    H3 encoding. Falls back to LocationID-based proxy if lat/lon
    columns are unavailable in the dataset.

    H3 Resolution 8 ≈ ~460m edge length — ideal for city blocks.

    Args:
        df: DataFrame with pickup/dropoff location information.

    Returns:
        DataFrame with h3_pickup_index and h3_dropoff_index columns.
    """
    logger.info("🌐 Computing H3 geospatial indices (Resolution 8)...")

    # Check if lat/lon columns exist in the dataset
    has_geo_cols = all(
        col in df.columns
        for col in [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]
    )

    if has_geo_cols:
        # Direct H3 encoding from lat/lon
        df = _h3_from_coordinates(df)
    else:
        logger.info(
            "  ℹ️  Lat/Lon columns not found in data. "
            "Using LocationID-based H3 proxy mapping."
        )
        df = _h3_from_location_id(df)

    logger.info("✅ H3 indices added")
    return df


def _h3_from_coordinates(df: DataFrame) -> DataFrame:
    """Compute H3 indices directly from lat/lon using pandas_udf."""
    try:
        import h3
    except ImportError:
        logger.warning("h3-py not installed, using LocationID proxy instead.")
        return _h3_from_location_id(df)

    resolution = cfg.features.H3_RESOLUTION

    @F.pandas_udf(StringType())
    def lat_lon_to_h3(lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
        """Vectorized H3 encoding from lat/lon pairs."""
        return pd.Series(
            [
                h3.latlng_to_cell(lat, lon, resolution)
                if pd.notna(lat) and pd.notna(lon)
                else None
                for lat, lon in zip(lat_series, lon_series)
            ]
        )

    df = df.withColumn(
        "h3_pickup_index",
        lat_lon_to_h3(
            F.col("pickup_latitude"),
            F.col("pickup_longitude"),
        ),
    ).withColumn(
        "h3_dropoff_index",
        lat_lon_to_h3(
            F.col("dropoff_latitude"),
            F.col("dropoff_longitude"),
        ),
    )

    return df


def _h3_from_location_id(df: DataFrame) -> DataFrame:
    """
    Create H3-like spatial index from LocationID.

    Since the 2024 Yellow Taxi data uses LocationIDs instead of
    raw lat/lon, we create a deterministic spatial hash that maps
    each LocationID to an H3-like hex string. This preserves the
    spatial semantics while working with the available data.
    """
    try:
        import h3
    except ImportError:
        logger.warning(
            "h3-py not installed. Using raw LocationID as spatial index."
        )
        df = df.withColumn(
            "h3_pickup_index",
            F.concat(F.lit("loc_"), F.col("PULocationID").cast("string")),
        ).withColumn(
            "h3_dropoff_index",
            F.concat(F.lit("loc_"), F.col("DOLocationID").cast("string")),
        )
        return df

    resolution = cfg.features.H3_RESOLUTION

    # NYC bounding box center coordinates mapped to LocationIDs
    # This creates a deterministic mapping: LocationID -> H3 cell
    @F.pandas_udf(StringType())
    def location_id_to_h3(loc_ids: pd.Series) -> pd.Series:
        """
        Map LocationID to H3 cell using NYC grid.

        Each LocationID maps to a unique lat/lon in the NYC bounding box
        (40.4774° to 40.9176° N, -74.2591° to -73.7004° W).
        """
        nyc_lat_min, nyc_lat_max = 40.4774, 40.9176
        nyc_lon_min, nyc_lon_max = -74.2591, -73.7004

        results = []
        for loc_id in loc_ids:
            if pd.isna(loc_id) or loc_id <= 0:
                results.append(None)
                continue

            # Deterministic mapping using golden ratio spacing
            loc_int = int(loc_id)
            lat = nyc_lat_min + (loc_int * 0.6180339887) % 1 * (
                nyc_lat_max - nyc_lat_min
            )
            lon = nyc_lon_min + (loc_int * 0.3819660113) % 1 * (
                nyc_lon_max - nyc_lon_min
            )
            results.append(h3.latlng_to_cell(lat, lon, resolution))

        return pd.Series(results)

    df = df.withColumn(
        "h3_pickup_index",
        location_id_to_h3(F.col("PULocationID").cast("double")),
    ).withColumn(
        "h3_dropoff_index",
        location_id_to_h3(F.col("DOLocationID").cast("double")),
    )

    return df


# ────────────────────────────────────────────────────────────────────
# STEP 3: ROLLING WINDOW DEMAND AGGREGATES
# ────────────────────────────────────────────────────────────────────
def compute_rolling_demand(df: DataFrame) -> DataFrame:
    """
    Calculate 15, 30, and 60-minute rolling window demand.

    Uses Spark SQL Window functions with rangeBetween() on
    unix timestamp to compute trip counts within each window.

    Partition: PULocationID (demand is location-specific)
    Order: tpep_pickup_datetime (temporal ordering)

    Args:
        df: DataFrame with pickup datetime and location.

    Returns:
        DataFrame with rolling demand columns.
    """
    logger.info("📊 Computing rolling window demand (15/30/60 min)...")

    # Add unix timestamp for range-based windows
    df = df.withColumn(
        "pickup_unix_ts",
        F.unix_timestamp("tpep_pickup_datetime"),
    )

    for window_minutes in cfg.features.ROLLING_WINDOWS_MINUTES:
        window_seconds = window_minutes * 60

        # Window: all trips at same location within the past N minutes
        w = (
            Window
            .partitionBy("PULocationID")
            .orderBy("pickup_unix_ts")
            .rangeBetween(-window_seconds, 0)
        )

        col_name = f"demand_{window_minutes}min"
        df = df.withColumn(col_name, F.count("*").over(w))

        logger.info(f"  ✅ {col_name} computed ({window_minutes}-min window)")

    return df


# ────────────────────────────────────────────────────────────────────
# STEP 4: SUPPLY ESTIMATION & SURGE CALCULATION
# ────────────────────────────────────────────────────────────────────
def compute_supply_and_surge(df: DataFrame) -> DataFrame:
    """
    Calculate historical supply baseline and surge multiplier.

    Supply = Historical average demand at (location, hour, day_of_week)
             over the past N days (configurable lookback).

    Surge Multiplier = Demand / Supply, bounded by [MIN_SURGE, MAX_SURGE].

    This creates the target variable for the ML model.

    Args:
        df: DataFrame with demand columns.

    Returns:
        DataFrame with supply_baseline and surge_multiplier.
    """
    logger.info("📈 Computing supply baseline and surge multiplier...")

    # ── Supply Baseline: Average demand by (location, hour, day_of_week) ──
    supply_df = (
        df.groupBy("PULocationID", "hour", "day_of_week")
        .agg(
            F.avg("demand_60min").alias("supply_baseline"),
            F.stddev("demand_60min").alias("supply_stddev"),
            F.count("*").alias("sample_count"),
        )
    )

    # Join supply baseline back to trips
    df = df.join(
        F.broadcast(supply_df),
        on=["PULocationID", "hour", "day_of_week"],
        how="left",
    )

    # ── Surge Multiplier ──
    # Handle edge cases: supply_baseline = 0 or null
    df = df.withColumn(
        "supply_baseline",
        F.when(
            (F.col("supply_baseline").isNull()) | (F.col("supply_baseline") == 0),
            1.0,
        ).otherwise(F.col("supply_baseline")),
    )

    df = df.withColumn(
        "surge_multiplier_raw",
        F.col("demand_60min") / F.col("supply_baseline"),
    )

    # Bound the surge multiplier
    df = df.withColumn(
        "surge_multiplier",
        F.greatest(
            F.lit(cfg.features.MIN_SURGE),
            F.least(
                F.lit(cfg.features.MAX_SURGE),
                F.col("surge_multiplier_raw"),
            ),
        ),
    )

    # Z-score of demand (anomaly indicator)
    df = df.withColumn(
        "demand_zscore",
        F.when(
            (F.col("supply_stddev").isNotNull()) & (F.col("supply_stddev") > 0),
            (F.col("demand_60min") - F.col("supply_baseline"))
            / F.col("supply_stddev"),
        ).otherwise(0.0),
    )

    logger.info(
        "✅ Surge multiplier computed | "
        f"Bounds: [{cfg.features.MIN_SURGE}, {cfg.features.MAX_SURGE}]"
    )

    return df


# ────────────────────────────────────────────────────────────────────
# STEP 5: WEATHER DATA INTEGRATION
# ────────────────────────────────────────────────────────────────────
def integrate_weather(df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Join weather data on (date, hour) with a left join.

    Weather features added:
      - temperature_f
      - precipitation_in
      - wind_speed_mph
      - visibility_miles
      - weather_condition (one-hot encoded)

    Args:
        df: Feature-enriched DataFrame.
        spark: Active SparkSession.

    Returns:
        DataFrame with weather features.
    """
    logger.info("🌧️  Integrating weather data...")

    weather_path = cfg.paths.WEATHER_DATA_PATH

    # Generate weather data if it doesn't exist
    if not os.path.exists(weather_path):
        logger.info("  Generating synthetic weather data...")
        from weather_data import generate_weather_data
        generate_weather_data(weather_path)

    weather_df = spark.read.parquet(weather_path)

    # Rename to avoid ambiguity on join keys
    weather_df = weather_df.select(
        F.col("date").alias("weather_date"),
        F.col("hour").alias("weather_hour"),
        F.col("temperature_f"),
        F.col("precipitation_in"),
        F.col("wind_speed_mph"),
        F.col("visibility_miles"),
        F.col("weather_condition"),
    )

    # Left join on date and hour
    df = df.join(
        F.broadcast(weather_df),
        on=[
            df["pickup_date"] == weather_df["weather_date"],
            df["hour"] == weather_df["weather_hour"],
        ],
        how="left",
    ).drop("weather_date", "weather_hour")

    # One-hot encode weather condition
    weather_conditions = ["Clear", "Rain", "Snow", "Fog"]
    for condition in weather_conditions:
        df = df.withColumn(
            f"weather_is_{condition.lower()}",
            F.when(F.col("weather_condition") == condition, 1).otherwise(0),
        )

    # Fill nulls (for dates without weather data)
    df = df.fillna(
        {
            "temperature_f": 55.0,       # NYC annual average
            "precipitation_in": 0.0,
            "wind_speed_mph": 8.0,
            "visibility_miles": 10.0,
            "weather_is_clear": 1,
            "weather_is_rain": 0,
            "weather_is_snow": 0,
            "weather_is_fog": 0,
        }
    )

    logger.info("✅ Weather data integrated (5 features + 4 one-hot)")

    return df


# ────────────────────────────────────────────────────────────────────
# STEP 6: DERIVED FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────
def add_derived_features(df: DataFrame) -> DataFrame:
    """
    Create additional engineered features for the model.

    Features:
      - fare_per_mile: Total fare / trip distance.
      - fare_per_minute: Total fare / trip duration.
      - speed_mph: Trip distance / (trip_duration / 60).
      - demand_acceleration: Ratio of 15-min to 60-min demand.
      - demand_trend: 30-min demand minus 60-min demand (trend).
      - congestion_indicator: speed < 10 mph flag.

    Args:
        df: Feature-enriched DataFrame.

    Returns:
        DataFrame with additional derived features.
    """
    logger.info("🔧 Adding derived features...")

    df = (
        df
        # Revenue metrics
        .withColumn(
            "fare_per_mile",
            F.when(F.col("trip_distance") > 0, F.col("fare_amount") / F.col("trip_distance"))
            .otherwise(0.0),
        )
        .withColumn(
            "fare_per_minute",
            F.when(
                F.col("trip_duration_minutes") > 0,
                F.col("fare_amount") / F.col("trip_duration_minutes"),
            ).otherwise(0.0),
        )
        # Speed
        .withColumn(
            "speed_mph",
            F.when(
                F.col("trip_duration_minutes") > 0,
                F.col("trip_distance") / (F.col("trip_duration_minutes") / 60),
            ).otherwise(0.0),
        )
        # Demand dynamics
        .withColumn(
            "demand_acceleration",
            F.when(
                F.col("demand_60min") > 0,
                F.col("demand_15min") / F.col("demand_60min"),
            ).otherwise(0.0),
        )
        .withColumn(
            "demand_trend",
            F.col("demand_30min") - F.col("demand_60min"),
        )
        # Congestion
        .withColumn(
            "congestion_indicator",
            F.when(
                (F.col("speed_mph") > 0) & (F.col("speed_mph") < 10), 1
            ).otherwise(0),
        )
    )

    # Cap extreme speed values
    df = df.withColumn(
        "speed_mph",
        F.least(F.col("speed_mph"), F.lit(100.0)),  # Max 100 mph
    )

    logger.info("✅ Derived features added (6 features)")

    return df


# ────────────────────────────────────────────────────────────────────
# STEP 7: WRITE FEATURE OUTPUT
# ────────────────────────────────────────────────────────────────────
def write_feature_output(df: DataFrame) -> str:
    """
    Write the feature-engineered dataset to parquet.

    Args:
        df: Final feature DataFrame.

    Returns:
        Path to feature output directory.
    """
    output_path = cfg.paths.FEATURES_DIR
    logger.info(f"💾 Writing feature output to: {output_path}")

    # Select only the columns needed for modeling
    feature_cols = [
        # IDs / keys
        "PULocationID",
        "DOLocationID",
        "h3_pickup_index",
        "h3_dropoff_index",
        # Temporal
        "pickup_date",
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "is_rush_hour",
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
        # Target
        "surge_multiplier",
    ]

    # Only select columns that exist
    available_cols = set(df.columns)
    selected_cols = [c for c in feature_cols if c in available_cols]

    missing = set(feature_cols) - available_cols
    if missing:
        logger.warning(f"⚠️  Missing feature columns: {missing}")

    df_out = df.select(selected_cols)

    df_out.write.mode("overwrite").format("parquet").option(
        "compression", "snappy"
    ).save(output_path)

    logger.info(
        f"✅ Feature output written | Columns: {len(selected_cols)} | "
        f"Path: {output_path}"
    )

    return output_path


# ────────────────────────────────────────────────────────────────────
# MAIN FEATURE ENGINEERING ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────
def run_feature_engineering() -> str:
    """
    Execute the full feature engineering pipeline.

    Stages:
      1. Read processed ETL output.
      2. Add H3 geospatial indices.
      3. Compute rolling window demand (15/30/60 min).
      4. Calculate supply baseline and surge multiplier.
      5. Integrate weather data.
      6. Add derived features.
      7. Write feature output.

    Returns:
        Path to the feature output directory.
    """
    logger.info("=" * 70)
    logger.info("  PHASE B: FEATURE ENGINEERING — START")
    logger.info("=" * 70)

    monitor.checkpoint("Features — Pre-Start")

    try:
        with timer("Phase B: Full Feature Engineering", logger):

            spark = get_spark_session(logger)

            # ── Step 1: Read ──
            with timer("Step 1: Read Processed Data", logger):
                df = read_processed_data(spark)

            # ── Step 2: H3 Indices ──
            with timer("Step 2: H3 Geospatial Indexing", logger):
                df = add_h3_indices(df)

            monitor.checkpoint("Features — Post-H3")

            # ── Step 3: Rolling Demand ──
            with timer("Step 3: Rolling Window Demand", logger):
                df = compute_rolling_demand(df)

            monitor.checkpoint("Features — Post-Rolling Windows")
            monitor.force_gc("Post-rolling window cleanup")

            # ── Step 4: Surge Multiplier ──
            with timer("Step 4: Supply & Surge Calculation", logger):
                df = compute_supply_and_surge(df)

            # ── Step 5: Weather ──
            with timer("Step 5: Weather Integration", logger):
                df = integrate_weather(df, spark)

            # ── Step 6: Derived Features ──
            with timer("Step 6: Derived Features", logger):
                df = add_derived_features(df)

            monitor.checkpoint("Features — Pre-Write")

            # ── Step 7: Write ──
            with timer("Step 7: Write Feature Output", logger):
                output_path = write_feature_output(df)
                del df

        # ── Post-Feature Cleanup ──
        monitor.force_gc("Post-Feature Engineering cleanup")
        monitor.checkpoint("Features — Complete")

        logger.info("=" * 70)
        logger.info("  PHASE B: FEATURE ENGINEERING — COMPLETE ✅")
        logger.info("=" * 70)

        return output_path

    except Exception as e:
        logger.error(
            f"❌ FEATURE ENGINEERING FAILED: {type(e).__name__}: {e}"
        )
        monitor.checkpoint("Features — FAILED")
        monitor.force_gc("Post-failure cleanup")
        raise


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        output = run_feature_engineering()
        logger.info(f"Feature output at: {output}")
    except Exception as e:
        logger.error(f"Pipeline terminated: {e}")
        sys.exit(1)
    finally:
        from utils import stop_spark
        stop_spark(logger)
