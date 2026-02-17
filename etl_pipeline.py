"""
╔══════════════════════════════════════════════════════════════════════╗
║         SURGE PRICING ENGINE — PHASE A: ETL PIPELINE                ║
║    Memory-Safe Ingestion, Cleaning, Zone Join, Partitioned Output   ║
╚══════════════════════════════════════════════════════════════════════╝

This module:
  1. Reads raw Yellow Taxi parquet files (100M+ rows).
  2. Performs data quality filtering (distance, fare, passenger bounds).
  3. Joins with taxi+_zone_lookup.csv via broadcast join (< 10 MB).
  4. Extracts temporal features (date, hour, day_of_week).
  5. Writes cleaned data partitioned by (pickup_date, hour) for
     predicate pushdown in downstream stages.

Memory Safety:
  - Column pruning at read time (only keeps necessary columns).
  - Broadcast join for small dimension table.
  - Manual gc.collect() after completion.
  - All operations wrapped in try-except for fault tolerance.

Author: Data Engineering Team
Version: 1.0.0
"""

import gc
import os
import sys
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
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
# ZONE LOOKUP SCHEMA (explicit to avoid inference overhead)
# ────────────────────────────────────────────────────────────────────
ZONE_LOOKUP_SCHEMA = StructType(
    [
        StructField("LocationID", IntegerType(), True),
        StructField("Borough", StringType(), True),
        StructField("Zone", StringType(), True),
        StructField("service_zone", StringType(), True),
    ]
)


# ────────────────────────────────────────────────────────────────────
# STEP 1: READ RAW PARQUET FILES
# ────────────────────────────────────────────────────────────────────
def read_raw_parquet(spark: SparkSession) -> DataFrame:
    """
    Read raw Yellow Taxi parquet files with column pruning.

    Only loads columns specified in ETLConfig.COLUMNS_TO_KEEP
    to minimize memory footprint from the start.

    Args:
        spark: Active SparkSession.

    Returns:
        Raw Spark DataFrame with selected columns.
    """
    parquet_path = os.path.join(
        cfg.paths.RAW_PARQUET_DIR, cfg.etl.PARQUET_GLOB
    )
    logger.info(f"📂 Reading parquet files from: {cfg.paths.RAW_PARQUET_DIR}")

    validate_path_exists(cfg.paths.RAW_PARQUET_DIR, "Raw parquet directory")

    # Read with glob pattern
    df = spark.read.parquet(parquet_path)

    # Column pruning — only select what we need
    available_cols = set(df.columns)
    selected_cols = [
        col for col in cfg.etl.COLUMNS_TO_KEEP if col in available_cols
    ]

    missing_cols = set(cfg.etl.COLUMNS_TO_KEEP) - available_cols
    if missing_cols:
        logger.warning(
            f"⚠️  Columns not found in data (will be skipped): {missing_cols}"
        )

    df = df.select(selected_cols)

    logger.info(
        f"✅ Raw data loaded | Columns: {len(selected_cols)} | "
        f"Schema: {[f'{c.name}:{c.dataType.simpleString()}' for c in df.schema]}"
    )
    return df


# ────────────────────────────────────────────────────────────────────
# STEP 2: DATA QUALITY FILTERING
# ────────────────────────────────────────────────────────────────────
def apply_quality_filters(df: DataFrame) -> DataFrame:
    """
    Apply data quality rules to remove invalid/outlier records.

    Filters:
      - Trip distance within [MIN, MAX] range.
      - Fare amount within [MIN, MAX] range.
      - Passenger count within [MIN, MAX] range.
      - Non-null pickup/dropoff datetimes.
      - Pickup datetime within 2024 calendar year.

    Args:
        df: Raw Spark DataFrame.

    Returns:
        Cleaned Spark DataFrame.
    """
    logger.info("🧹 Applying data quality filters...")

    # Cache initial count is expensive on 100M rows—skip for memory safety
    # Instead, we log the filters applied

    df_clean = df.filter(
        # Valid temporal data
        F.col("tpep_pickup_datetime").isNotNull()
        & F.col("tpep_dropoff_datetime").isNotNull()
        # Year boundary (prevents data from other years leaking in)
        & (F.year("tpep_pickup_datetime") == 2024)
        # Trip distance bounds
        & (F.col("trip_distance") >= cfg.etl.MIN_TRIP_DISTANCE)
        & (F.col("trip_distance") <= cfg.etl.MAX_TRIP_DISTANCE)
        # Fare bounds
        & (F.col("fare_amount") >= cfg.etl.MIN_FARE)
        & (F.col("fare_amount") <= cfg.etl.MAX_FARE)
        # Passenger count bounds (with null handling)
        & (
            F.col("passenger_count").isNull()
            | (
                (F.col("passenger_count") >= cfg.etl.MIN_PASSENGER_COUNT)
                & (F.col("passenger_count") <= cfg.etl.MAX_PASSENGER_COUNT)
            )
        )
        # Positive total amount
        & (F.col("total_amount") > 0)
    )

    logger.info(
        "✅ Quality filters applied: "
        f"distance=[{cfg.etl.MIN_TRIP_DISTANCE}, {cfg.etl.MAX_TRIP_DISTANCE}], "
        f"fare=[{cfg.etl.MIN_FARE}, {cfg.etl.MAX_FARE}], "
        f"passengers=[{cfg.etl.MIN_PASSENGER_COUNT}, {cfg.etl.MAX_PASSENGER_COUNT}]"
    )

    return df_clean


# ────────────────────────────────────────────────────────────────────
# STEP 3: EXTRACT TEMPORAL FEATURES
# ────────────────────────────────────────────────────────────────────
def extract_temporal_features(df: DataFrame) -> DataFrame:
    """
    Extract date/time features from pickup datetime.

    Creates:
      - pickup_date (DATE): For partitioning.
      - hour (INT): Hour of day [0-23].
      - day_of_week (INT): Day of week [1=Mon, 7=Sun].
      - day_of_month (INT): Day of month [1-31].
      - month (INT): Month [1-12].
      - is_weekend (INT): 1 if Saturday/Sunday, 0 otherwise.
      - is_rush_hour (INT): 1 if morning or evening rush, 0 otherwise.
      - trip_duration_minutes (DOUBLE): Dropoff - Pickup in minutes.

    Args:
        df: Cleaned Spark DataFrame with pickup/dropoff datetimes.

    Returns:
        DataFrame with added temporal features.
    """
    logger.info("🕐 Extracting temporal features...")

    morning_start, morning_end = cfg.features.MORNING_RUSH
    evening_start, evening_end = cfg.features.EVENING_RUSH

    df = (
        df
        .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
        .withColumn("hour", F.hour("tpep_pickup_datetime"))
        .withColumn("day_of_week", F.dayofweek("tpep_pickup_datetime"))
        .withColumn("day_of_month", F.dayofmonth("tpep_pickup_datetime"))
        .withColumn("month", F.month("tpep_pickup_datetime"))
        .withColumn(
            "is_weekend",
            F.when(
                F.dayofweek("tpep_pickup_datetime").isin([1, 7]), 1
            ).otherwise(0),
        )
        .withColumn(
            "is_rush_hour",
            F.when(
                (F.hour("tpep_pickup_datetime").between(morning_start, morning_end - 1))
                | (F.hour("tpep_pickup_datetime").between(evening_start, evening_end - 1)),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "trip_duration_minutes",
            F.round(
                (
                    F.unix_timestamp("tpep_dropoff_datetime")
                    - F.unix_timestamp("tpep_pickup_datetime")
                )
                / 60.0,
                2,
            ),
        )
    )

    # Filter out unrealistic trip durations
    df = df.filter(
        (F.col("trip_duration_minutes") > 0.5)  # At least 30 seconds
        & (F.col("trip_duration_minutes") < 360)  # Less than 6 hours
    )

    logger.info("✅ Temporal features extracted (7 features + trip_duration)")

    return df


# ────────────────────────────────────────────────────────────────────
# STEP 4: BROADCAST JOIN WITH ZONE LOOKUP
# ────────────────────────────────────────────────────────────────────
def join_zone_lookup(df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Join taxi trips with zone lookup data using broadcast join.

    The zone lookup CSV is ~30 KB (265 rows), making it an ideal
    candidate for broadcast join to avoid shuffle on the large table.

    Joins on both PULocationID (pickup) and DOLocationID (dropoff)
    to get borough and zone names for both endpoints.

    Args:
        df: Taxi trip DataFrame.
        spark: Active SparkSession.

    Returns:
        DataFrame enriched with zone information.
    """
    logger.info("🗺️  Joining with zone lookup (broadcast join)...")

    zone_csv_path = cfg.paths.ZONE_LOOKUP_CSV

    # If zone lookup CSV doesn't exist, create a minimal one
    if not os.path.exists(zone_csv_path):
        logger.warning(
            f"⚠️  Zone lookup CSV not found at {zone_csv_path}. "
            "Creating a default zone mapping from LocationIDs..."
        )
        return _add_placeholder_zones(df)

    # Read the zone lookup with explicit schema (no inference overhead)
    zones_df = (
        spark.read.format("csv")
        .option("header", "true")
        .schema(ZONE_LOOKUP_SCHEMA)
        .load(zone_csv_path)
    )

    logger.info(
        f"  Zone lookup loaded: {zones_df.count()} zones | "
        f"Size: {os.path.getsize(zone_csv_path) / 1024:.1f} KB"
    )

    # ── Pickup zone join (broadcast) ──
    pu_zones = zones_df.select(
        F.col("LocationID").alias("PU_LocationID"),
        F.col("Borough").alias("pickup_borough"),
        F.col("Zone").alias("pickup_zone"),
        F.col("service_zone").alias("pickup_service_zone"),
    )

    df = df.join(
        F.broadcast(pu_zones),
        df["PULocationID"] == pu_zones["PU_LocationID"],
        "left",
    ).drop("PU_LocationID")

    # ── Dropoff zone join (broadcast) ──
    do_zones = zones_df.select(
        F.col("LocationID").alias("DO_LocationID"),
        F.col("Borough").alias("dropoff_borough"),
        F.col("Zone").alias("dropoff_zone"),
        F.col("service_zone").alias("dropoff_service_zone"),
    )

    df = df.join(
        F.broadcast(do_zones),
        df["DOLocationID"] == do_zones["DO_LocationID"],
        "left",
    ).drop("DO_LocationID")

    logger.info("✅ Zone lookup joined (pickup + dropoff)")

    return df


def _add_placeholder_zones(df: DataFrame) -> DataFrame:
    """Add placeholder zone columns when lookup CSV is unavailable."""
    return (
        df
        .withColumn("pickup_borough", F.lit("Unknown"))
        .withColumn("pickup_zone", F.concat(F.lit("Zone_"), F.col("PULocationID")))
        .withColumn("pickup_service_zone", F.lit("Unknown"))
        .withColumn("dropoff_borough", F.lit("Unknown"))
        .withColumn("dropoff_zone", F.concat(F.lit("Zone_"), F.col("DOLocationID")))
        .withColumn("dropoff_service_zone", F.lit("Unknown"))
    )


# ────────────────────────────────────────────────────────────────────
# STEP 5: WRITE PARTITIONED OUTPUT
# ────────────────────────────────────────────────────────────────────
def write_partitioned_output(df: DataFrame) -> str:
    """
    Write cleaned data partitioned by (pickup_date, hour).

    Partitioning enables predicate pushdown in Phase B,
    allowing Spark to skip irrelevant partitions during reads.

    Args:
        df: Final cleaned and enriched DataFrame.

    Returns:
        Path to the output directory.
    """
    output_path = cfg.paths.PROCESSED_DIR
    logger.info(f"💾 Writing partitioned output to: {output_path}")

    df.write.mode("overwrite").partitionBy(
        *cfg.etl.PARTITION_KEYS
    ).format(
        cfg.etl.OUTPUT_FORMAT
    ).option(
        "compression", cfg.etl.COMPRESSION
    ).save(output_path)

    # Log output size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(output_path)
        for f in filenames
        if f.endswith(".parquet")
    )
    logger.info(
        f"✅ Output written | Path: {output_path} | "
        f"Size: {total_size / (1024 * 1024):.1f} MB | "
        f"Partition keys: {cfg.etl.PARTITION_KEYS}"
    )

    return output_path


# ────────────────────────────────────────────────────────────────────
# MAIN ETL ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────
def run_etl() -> str:
    """
    Execute the full ETL pipeline.

    Stages:
      1. Read raw parquet files (column-pruned).
      2. Apply data quality filters.
      3. Extract temporal features.
      4. Broadcast join with zone lookup.
      5. Write partitioned output.

    Returns:
        Path to the processed output directory.

    Raises:
        Exception: Re-raised after logging for upstream handling.
    """
    logger.info("=" * 70)
    logger.info("  PHASE A: ETL PIPELINE — START")
    logger.info("=" * 70)

    monitor.checkpoint("ETL — Pre-Start")

    try:
        with timer("Phase A: Full ETL Pipeline", logger):

            # ── Initialize Spark ──
            spark = get_spark_session(logger)

            # ── Step 1: Read raw data ──
            with timer("Step 1: Read Raw Parquet", logger):
                raw_df = read_raw_parquet(spark)

            # ── Step 2: Quality filters ──
            with timer("Step 2: Data Quality Filters", logger):
                clean_df = apply_quality_filters(raw_df)
                # Release reference to raw
                del raw_df

            # ── Step 3: Temporal features ──
            with timer("Step 3: Temporal Features", logger):
                temporal_df = extract_temporal_features(clean_df)
                del clean_df

            # ── Step 4: Zone join ──
            with timer("Step 4: Zone Lookup Join", logger):
                enriched_df = join_zone_lookup(temporal_df, spark)
                del temporal_df

            monitor.checkpoint("ETL — Pre-Write")

            # ── Step 5: Write output ──
            with timer("Step 5: Write Partitioned Output", logger):
                output_path = write_partitioned_output(enriched_df)
                del enriched_df

        # ── Post-ETL Cleanup ──
        monitor.force_gc("Post-ETL cleanup")
        monitor.checkpoint("ETL — Complete")

        logger.info("=" * 70)
        logger.info("  PHASE A: ETL PIPELINE — COMPLETE ✅")
        logger.info("=" * 70)

        return output_path

    except Exception as e:
        logger.error(f"❌ ETL PIPELINE FAILED: {type(e).__name__}: {e}")
        monitor.checkpoint("ETL — FAILED")
        monitor.force_gc("Post-failure cleanup")
        raise


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        output = run_etl()
        logger.info(f"ETL output at: {output}")
    except Exception as e:
        logger.error(f"Pipeline terminated: {e}")
        sys.exit(1)
    finally:
        from utils import stop_spark
        stop_spark(logger)
