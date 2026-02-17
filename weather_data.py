"""
╔══════════════════════════════════════════════════════════════════════╗
║            SURGE PRICING ENGINE — WEATHER DATA GENERATOR            ║
║         Synthetic Weather for NYC (2024) — Used for Joins           ║
╚══════════════════════════════════════════════════════════════════════╝

Generates realistic synthetic hourly weather data for NYC, 2024.
In production, this would be replaced by NOAA API / Open-Meteo data.

Author: Data Engineering Team
Version: 1.0.0
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg
from utils import get_logger, MemoryMonitor


logger = get_logger(__name__)


def generate_weather_data(
    output_path: Optional[str] = None,
    year: int = 2024,
) -> str:
    """
    Generate synthetic hourly weather data for NYC.

    Creates a parquet file with columns:
      - date (DATE)
      - hour (INT)
      - temperature_f (FLOAT)
      - precipitation_in (FLOAT)
      - wind_speed_mph (FLOAT)
      - visibility_miles (FLOAT)
      - weather_condition (STRING): Clear / Rain / Snow / Fog

    Args:
        output_path: Where to save the parquet file. Defaults to config path.
        year: Calendar year to generate.

    Returns:
        Path to the generated weather parquet file.
    """
    output_path = output_path or cfg.paths.WEATHER_DATA_PATH
    logger.info(f"🌤️  Generating synthetic weather data for {year}...")

    # Seed for reproducibility
    rng = np.random.default_rng(seed=42)

    # Generate hourly timestamps for the full year
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31 23:00:00")
    timestamps = pd.date_range(start=start, end=end, freq="h")

    n_hours = len(timestamps)
    logger.info(f"  Generating {n_hours:,} hourly records...")

    # ── Temperature: seasonal sinusoidal pattern + daily variation ──
    day_of_year = timestamps.dayofyear.values.astype(float)
    hour_of_day = timestamps.hour.values.astype(float)

    # NYC avg temps: Jan ~33°F, Jul ~77°F
    seasonal_temp = 55 + 22 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    noise = rng.normal(0, 3, n_hours)
    temperature_f = seasonal_temp + daily_variation + noise

    # ── Precipitation: higher in spring/fall, correlated with temp drops ──
    precip_prob = 0.15 + 0.1 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    has_precip = rng.random(n_hours) < precip_prob
    precipitation_in = np.where(
        has_precip,
        rng.exponential(0.15, n_hours),
        0.0,
    )
    precipitation_in = np.round(precipitation_in, 2)

    # ── Wind Speed: gustier in winter/spring ──
    base_wind = 8 + 4 * np.sin(2 * np.pi * (day_of_year + 30) / 365)
    wind_speed_mph = base_wind + rng.gamma(2, 1.5, n_hours)
    wind_speed_mph = np.round(wind_speed_mph, 1)

    # ── Visibility: reduced during precip ──
    visibility_miles = np.where(
        precipitation_in > 0.5,
        rng.uniform(1, 5, n_hours),
        np.where(
            precipitation_in > 0,
            rng.uniform(5, 8, n_hours),
            rng.uniform(8, 10, n_hours),
        ),
    )
    visibility_miles = np.round(visibility_miles, 1)

    # ── Weather Condition ──
    conditions = np.full(n_hours, "Clear", dtype=object)
    conditions[precipitation_in > 0] = "Rain"
    conditions[(precipitation_in > 0) & (temperature_f < 35)] = "Snow"
    conditions[(visibility_miles < 3) & (precipitation_in == 0)] = "Fog"

    # ── Assemble DataFrame ──
    weather_df = pd.DataFrame(
        {
            "date": timestamps.date,
            "hour": timestamps.hour,
            "temperature_f": np.round(temperature_f, 1),
            "precipitation_in": precipitation_in,
            "wind_speed_mph": wind_speed_mph,
            "visibility_miles": visibility_miles,
            "weather_condition": conditions,
        }
    )

    # Convert date to proper type
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    # Save
    weather_df.to_parquet(output_path, index=False, engine="pyarrow")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        f"✅ Weather data saved: {output_path} "
        f"({n_hours:,} rows, {file_size_mb:.1f} MB)"
    )

    return output_path


if __name__ == "__main__":
    generate_weather_data()
