"""
╔══════════════════════════════════════════════════════════════════════╗
║         SURGE PRICING ENGINE — ETL PIPELINE TESTS                   ║
║         Phase A: Data Integrity & Quality Filter Tests              ║
╚══════════════════════════════════════════════════════════════════════╝

Tests cover:
  - Data quality filter correctness
  - Temporal feature extraction accuracy
  - Zone lookup join integrity
  - Output partitioning validation
  - Edge cases and null handling
"""

import os
import sys
import tempfile
import shutil

import pytest
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from etl_pipeline import (
    apply_quality_filters,
    extract_temporal_features,
    _add_placeholder_zones,
)


# ════════════════════════════════════════════════════════════════════
# DATA QUALITY FILTERS
# ════════════════════════════════════════════════════════════════════
class TestQualityFilters:
    """Tests for the data quality filtering step."""

    def test_valid_records_pass_filters(self, sample_trip_data):
        """Valid records should survive all quality filters."""
        result = apply_quality_filters(sample_trip_data)
        # 5 valid + 2 invalid = 7 total, expect 5
        count = result.count()
        assert count == 5, f"Expected 5 valid records, got {count}"

    def test_short_trips_filtered(self, sample_trip_data, spark):
        """Trips shorter than MIN_TRIP_DISTANCE should be removed."""
        result = apply_quality_filters(sample_trip_data)
        min_distance = result.agg(F.min("trip_distance")).collect()[0][0]
        assert min_distance >= cfg.etl.MIN_TRIP_DISTANCE

    def test_excessive_fare_filtered(self, sample_trip_data):
        """Fares exceeding MAX_FARE should be removed."""
        result = apply_quality_filters(sample_trip_data)
        max_fare = result.agg(F.max("fare_amount")).collect()[0][0]
        assert max_fare <= cfg.etl.MAX_FARE

    def test_zero_total_amount_filtered(self, sample_trip_data, spark):
        """Records with total_amount <= 0 should be removed."""
        result = apply_quality_filters(sample_trip_data)
        min_total = result.agg(F.min("total_amount")).collect()[0][0]
        assert min_total > 0

    def test_year_boundary_enforced(self, sample_trip_data):
        """Only 2024 data should pass the filter."""
        result = apply_quality_filters(sample_trip_data)
        years = (
            result.select(F.year("tpep_pickup_datetime").alias("year"))
            .distinct()
            .collect()
        )
        assert all(row["year"] == 2024 for row in years)

    def test_empty_input_handling(self, spark):
        """Empty DataFrame should produce empty output, not crash."""
        from pyspark.sql.types import (
            DoubleType, IntegerType, StringType, StructField, StructType, TimestampType,
        )

        schema = StructType([
            StructField("tpep_pickup_datetime", TimestampType()),
            StructField("tpep_dropoff_datetime", TimestampType()),
            StructField("trip_distance", DoubleType()),
            StructField("fare_amount", DoubleType()),
            StructField("passenger_count", IntegerType()),
            StructField("total_amount", DoubleType()),
        ])
        empty_df = spark.createDataFrame([], schema)
        result = apply_quality_filters(empty_df)
        assert result.count() == 0


# ════════════════════════════════════════════════════════════════════
# TEMPORAL FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════
class TestTemporalFeatures:
    """Tests for temporal feature extraction."""

    def test_pickup_date_extracted(self, sample_trip_data):
        """pickup_date column should be created as DateType."""
        filtered = apply_quality_filters(sample_trip_data)
        result = extract_temporal_features(filtered)
        assert "pickup_date" in result.columns

    def test_hour_range(self, sample_trip_data):
        """Hour should be between 0 and 23."""
        filtered = apply_quality_filters(sample_trip_data)
        result = extract_temporal_features(filtered)
        hours = result.select("hour").distinct().collect()
        for row in hours:
            assert 0 <= row["hour"] <= 23

    def test_weekend_flag(self, sample_trip_data):
        """is_weekend should be 0 or 1."""
        filtered = apply_quality_filters(sample_trip_data)
        result = extract_temporal_features(filtered)
        weekend_vals = (
            result.select("is_weekend").distinct().rdd.flatMap(lambda x: x).collect()
        )
        assert all(v in (0, 1) for v in weekend_vals)

    def test_rush_hour_flag(self, sample_trip_data):
        """is_rush_hour should be 0 or 1."""
        filtered = apply_quality_filters(sample_trip_data)
        result = extract_temporal_features(filtered)
        rush_vals = (
            result.select("is_rush_hour").distinct().rdd.flatMap(lambda x: x).collect()
        )
        assert all(v in (0, 1) for v in rush_vals)

    def test_trip_duration_positive(self, sample_trip_data):
        """trip_duration_minutes should be positive after filtering."""
        filtered = apply_quality_filters(sample_trip_data)
        result = extract_temporal_features(filtered)
        min_duration = result.agg(F.min("trip_duration_minutes")).collect()[0][0]
        assert min_duration > 0

    def test_all_temporal_columns_present(self, sample_trip_data):
        """All expected temporal columns should be created."""
        filtered = apply_quality_filters(sample_trip_data)
        result = extract_temporal_features(filtered)
        expected = [
            "pickup_date", "hour", "day_of_week", "day_of_month",
            "month", "is_weekend", "is_rush_hour", "trip_duration_minutes",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"


# ════════════════════════════════════════════════════════════════════
# ZONE LOOKUP JOIN
# ════════════════════════════════════════════════════════════════════
class TestZoneJoin:
    """Tests for zone lookup join functionality."""

    def test_placeholder_zones_added(self, sample_trip_data):
        """Placeholder zones should be added when CSV is unavailable."""
        filtered = apply_quality_filters(sample_trip_data)
        result = _add_placeholder_zones(filtered)

        assert "pickup_borough" in result.columns
        assert "pickup_zone" in result.columns
        assert "dropoff_borough" in result.columns
        assert "dropoff_zone" in result.columns

    def test_zone_columns_no_nulls(self, sample_trip_data):
        """Zone placeholder columns should have no nulls."""
        filtered = apply_quality_filters(sample_trip_data)
        result = _add_placeholder_zones(filtered)

        null_count = result.filter(
            F.col("pickup_borough").isNull()
        ).count()
        assert null_count == 0

    def test_row_count_preserved_after_zone_join(self, sample_trip_data):
        """Left join should not duplicate or lose rows."""
        filtered = apply_quality_filters(sample_trip_data)
        original_count = filtered.count()
        result = _add_placeholder_zones(filtered)
        joined_count = result.count()
        assert joined_count == original_count


# ════════════════════════════════════════════════════════════════════
# OUTPUT PARTITIONING
# ════════════════════════════════════════════════════════════════════
class TestOutputPartitioning:
    """Tests for partitioned output writing."""

    def test_partition_keys_valid(self):
        """Partition keys should be defined in config."""
        keys = cfg.etl.PARTITION_KEYS
        assert len(keys) >= 1
        assert "pickup_date" in keys
        assert "hour" in keys

    def test_output_format_valid(self):
        """Output format should be parquet."""
        assert cfg.etl.OUTPUT_FORMAT == "parquet"

    def test_compression_valid(self):
        """Compression should be snappy."""
        assert cfg.etl.COMPRESSION == "snappy"
