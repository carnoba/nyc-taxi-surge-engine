"""
Shared pytest fixtures for the Surge Pricing Engine test suite.

Provides:
  - Spark session (shared across tests)
  - Sample DataFrames (trips, zones, weather, features)
  - Temporary directories for test output
"""

import os
import sys
import gc
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Spark Session (module-scoped for speed) ────────────────────────
@pytest.fixture(scope="module")
def spark():
    """Create a lightweight SparkSession for testing."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("SurgePricing_Tests")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    yield spark

    spark.stop()
    gc.collect()


# ─── Temporary output directory ─────────────────────────────────────
@pytest.fixture
def tmp_output_dir():
    """Create a temporary directory for test output."""
    tmpdir = tempfile.mkdtemp(prefix="surge_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Sample Trip Data ──────────────────────────────────────────────
@pytest.fixture
def sample_trip_data(spark):
    """Create a small sample of Yellow Taxi trip data."""
    data = [
        # (VendorID, pickup_dt, dropoff_dt, passengers, distance,
        #  PULID, DOLID, RateCode, payment, fare, extra, mta, tip,
        #  tolls, improvement, total, congestion)
        (1, "2024-03-15 08:30:00", "2024-03-15 08:45:00", 2, 3.5,
         161, 237, 1, 1, 15.00, 1.0, 0.5, 3.00, 0.0, 0.3, 19.80, 2.5),
        (2, "2024-03-15 09:00:00", "2024-03-15 09:20:00", 1, 5.2,
         236, 141, 1, 1, 22.00, 0.0, 0.5, 4.50, 0.0, 0.3, 27.30, 2.5),
        (1, "2024-03-15 17:30:00", "2024-03-15 18:00:00", 3, 8.1,
         161, 48, 1, 2, 35.00, 2.5, 0.5, 0.0, 6.55, 0.3, 44.85, 2.5),
        (2, "2024-06-20 12:00:00", "2024-06-20 12:10:00", 1, 1.2,
         170, 162, 1, 1, 8.50, 0.0, 0.5, 2.00, 0.0, 0.3, 11.30, 2.5),
        (1, "2024-06-20 23:30:00", "2024-06-20 23:55:00", 2, 6.7,
         132, 265, 1, 1, 28.00, 1.0, 0.5, 5.50, 0.0, 0.3, 35.30, 2.5),
        # Invalid records (should be filtered)
        (1, "2024-01-01 00:00:00", "2024-01-01 00:05:00", 0, 0.01,
         1, 1, 1, 1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),   # too short + 0 passengers
        (2, "2024-01-01 01:00:00", "2024-01-01 01:30:00", 1, 250.0,
         1, 1, 1, 1, 6000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6000.0, 0.0),  # too far + too expensive
    ]

    from pyspark.sql.types import (
        DoubleType, IntegerType, StringType, StructField, StructType, TimestampType,
    )

    schema = StructType([
        StructField("VendorID", IntegerType()),
        StructField("tpep_pickup_datetime", StringType()),
        StructField("tpep_dropoff_datetime", StringType()),
        StructField("passenger_count", IntegerType()),
        StructField("trip_distance", DoubleType()),
        StructField("PULocationID", IntegerType()),
        StructField("DOLocationID", IntegerType()),
        StructField("RatecodeID", IntegerType()),
        StructField("payment_type", IntegerType()),
        StructField("fare_amount", DoubleType()),
        StructField("extra", DoubleType()),
        StructField("mta_tax", DoubleType()),
        StructField("tip_amount", DoubleType()),
        StructField("tolls_amount", DoubleType()),
        StructField("improvement_surcharge", DoubleType()),
        StructField("total_amount", DoubleType()),
        StructField("congestion_surcharge", DoubleType()),
    ])

    df = spark.createDataFrame(data, schema)
    df = df.withColumn(
        "tpep_pickup_datetime",
        df["tpep_pickup_datetime"].cast(TimestampType()),
    ).withColumn(
        "tpep_dropoff_datetime",
        df["tpep_dropoff_datetime"].cast(TimestampType()),
    )

    return df


# ─── Sample Zone Lookup Data ───────────────────────────────────────
@pytest.fixture
def sample_zone_data(spark):
    """Create sample zone lookup data."""
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    schema = StructType([
        StructField("LocationID", IntegerType()),
        StructField("Borough", StringType()),
        StructField("Zone", StringType()),
        StructField("service_zone", StringType()),
    ])

    data = [
        (161, "Manhattan", "Midtown Center", "Yellow Zone"),
        (237, "Manhattan", "Upper East Side South", "Yellow Zone"),
        (236, "Manhattan", "Upper East Side North", "Yellow Zone"),
        (141, "Manhattan", "Lenox Hill West", "Yellow Zone"),
        (48, "Manhattan", "Clinton East", "Yellow Zone"),
        (170, "Manhattan", "Murray Hill", "Yellow Zone"),
        (162, "Manhattan", "Midtown East", "Yellow Zone"),
        (132, "Queens", "JFK Airport", "Airports"),
        (265, "Brooklyn", "Williamsburg (South Side)", "Boro Zone"),
    ]

    return spark.createDataFrame(data, schema)


# ─── Sample Feature Data (pandas) ──────────────────────────────────
@pytest.fixture
def sample_feature_df():
    """Create sample feature DataFrame for model testing."""
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "hour": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(1, 8, n),
        "day_of_month": np.random.randint(1, 32, n),
        "month": np.random.randint(1, 13, n),
        "is_weekend": np.random.randint(0, 2, n),
        "is_rush_hour": np.random.randint(0, 2, n),
        "PULocationID": np.random.randint(1, 265, n),
        "DOLocationID": np.random.randint(1, 265, n),
        "trip_distance": np.random.exponential(3, n) + 0.5,
        "trip_duration_minutes": np.random.exponential(15, n) + 1,
        "passenger_count": np.random.randint(1, 5, n),
        "fare_amount": np.random.exponential(15, n) + 2.5,
        "total_amount": np.random.exponential(20, n) + 3,
        "demand_15min": np.random.poisson(10, n),
        "demand_30min": np.random.poisson(20, n),
        "demand_60min": np.random.poisson(40, n),
        "supply_baseline": np.random.exponential(35, n) + 5,
        "demand_zscore": np.random.normal(0, 1, n),
        "fare_per_mile": np.random.exponential(5, n),
        "fare_per_minute": np.random.exponential(1, n),
        "speed_mph": np.random.exponential(15, n) + 2,
        "demand_acceleration": np.abs(np.random.normal(0.25, 0.1, n)),
        "demand_trend": np.random.normal(-5, 10, n),
        "congestion_indicator": np.random.randint(0, 2, n),
        "temperature_f": np.random.normal(55, 15, n),
        "precipitation_in": np.random.exponential(0.1, n),
        "wind_speed_mph": np.random.exponential(8, n) + 2,
        "visibility_miles": np.random.uniform(1, 10, n),
        "weather_is_clear": np.random.randint(0, 2, n),
        "weather_is_rain": np.random.randint(0, 2, n),
        "weather_is_snow": np.random.randint(0, 2, n),
        "weather_is_fog": np.random.randint(0, 2, n),
    })

    # Create realistic surge target: base surge + demand influence + weather
    base_surge = 1.0
    demand_effect = 0.3 * (df["demand_60min"] / df["supply_baseline"])
    rush_effect = 0.2 * df["is_rush_hour"]
    weather_effect = 0.15 * df["weather_is_rain"] + 0.25 * df["weather_is_snow"]
    noise = np.random.normal(0, 0.1, n)

    df["surge_multiplier"] = np.clip(
        base_surge + demand_effect + rush_effect + weather_effect + noise,
        1.0,
        5.0,
    )

    return df
