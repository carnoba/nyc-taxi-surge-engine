"""
╔══════════════════════════════════════════════════════════════════════╗
║              SURGE PRICING ENGINE — UTILITIES                       ║
║         Logging, Memory Monitoring, Spark Session Factory           ║
╚══════════════════════════════════════════════════════════════════════╝

Shared utilities used across all pipeline phases.

Author: Data Engineering Team
Version: 1.0.0
"""

import gc
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional

import psutil
from pyspark.sql import SparkSession

from config import cfg


# ────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ────────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Create a consistently formatted logger.

    Args:
        name: Logger name (typically __name__ of calling module).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, cfg.log.LOG_LEVEL))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, cfg.log.LOG_LEVEL))

        formatter = logging.Formatter(
            fmt=cfg.log.LOG_FORMAT,
            datefmt=cfg.log.DATE_FORMAT,
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (rotated)
        log_file = os.path.join(cfg.paths.REPORTS_DIR, "pipeline.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(getattr(logging, cfg.log.LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ────────────────────────────────────────────────────────────────────
# MEMORY MONITORING
# ────────────────────────────────────────────────────────────────────
class MemoryMonitor:
    """
    Monitors system memory and logs warnings when thresholds are breached.

    Usage:
        monitor = MemoryMonitor(logger)
        monitor.checkpoint("After ETL stage")
        monitor.force_gc("Post-ETL cleanup")
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._process = psutil.Process(os.getpid())

    def get_memory_usage_mb(self) -> float:
        """Get current process RSS memory in MB."""
        return self._process.memory_info().rss / (1024 * 1024)

    def get_system_memory_mb(self) -> dict:
        """Get system-wide memory stats in MB."""
        mem = psutil.virtual_memory()
        return {
            "total_mb": round(mem.total / (1024 * 1024), 1),
            "available_mb": round(mem.available / (1024 * 1024), 1),
            "used_mb": round(mem.used / (1024 * 1024), 1),
            "percent_used": mem.percent,
        }

    def checkpoint(self, stage_name: str) -> dict:
        """
        Log memory usage at a pipeline checkpoint.

        Args:
            stage_name: Human-readable stage identifier.

        Returns:
            Dict with memory metrics.
        """
        process_mb = self.get_memory_usage_mb()
        system = self.get_system_memory_mb()

        self.logger.info(
            f"📊 MEMORY CHECKPOINT [{stage_name}] | "
            f"Process: {process_mb:.1f} MB | "
            f"System: {system['used_mb']:.1f}/{system['total_mb']:.1f} MB "
            f"({system['percent_used']}% used) | "
            f"Available: {system['available_mb']:.1f} MB"
        )

        # Warnings
        if system["used_mb"] >= cfg.log.MEMORY_CRITICAL_THRESHOLD_MB:
            self.logger.critical(
                f"🚨 CRITICAL MEMORY [{stage_name}]: "
                f"{system['used_mb']:.0f} MB used — OOM risk is HIGH!"
            )
        elif system["used_mb"] >= cfg.log.MEMORY_WARNING_THRESHOLD_MB:
            self.logger.warning(
                f"⚠️  HIGH MEMORY [{stage_name}]: "
                f"{system['used_mb']:.0f} MB used — approaching limit"
            )

        return {
            "stage": stage_name,
            "process_mb": round(process_mb, 1),
            **system,
        }

    def force_gc(self, reason: str = "routine") -> int:
        """
        Force garbage collection and log result.

        Args:
            reason: Why GC is being triggered.

        Returns:
            Number of objects collected.
        """
        before_mb = self.get_memory_usage_mb()
        collected = gc.collect()
        after_mb = self.get_memory_usage_mb()

        freed = before_mb - after_mb
        self.logger.info(
            f"🧹 GC [{reason}] | Collected: {collected} objects | "
            f"Before: {before_mb:.1f} MB → After: {after_mb:.1f} MB | "
            f"Freed: {freed:.1f} MB"
        )
        return collected


# ────────────────────────────────────────────────────────────────────
# SPARK SESSION FACTORY
# ────────────────────────────────────────────────────────────────────
_spark_session: Optional[SparkSession] = None


def get_spark_session(logger: Optional[logging.Logger] = None) -> SparkSession:
    """
    Create or retrieve the singleton SparkSession with the 8GB safety shield.

    The session is configured with all parameters from SparkConfig
    to ensure memory safety on constrained hardware.

    Args:
        logger: Optional logger for startup messages.

    Returns:
        Configured SparkSession.
    """
    global _spark_session

    if _spark_session is not None and not _spark_session._jsc.sc().isStopped():
        return _spark_session

    if logger:
        logger.info("🚀 Initializing SparkSession with 8GB Safety Shield...")

    builder = SparkSession.builder

    # Apply all safety configurations
    spark_conf = cfg.spark.to_spark_conf_dict()
    for key, value in spark_conf.items():
        builder = builder.config(key, value)

    _spark_session = builder.getOrCreate()

    # Reduce Spark's own logging noise
    _spark_session.sparkContext.setLogLevel("WARN")

    if logger:
        logger.info(
            f"✅ SparkSession ready | "
            f"Driver Memory: {cfg.spark.DRIVER_MEMORY} | "
            f"Shuffle Partitions: {cfg.spark.SHUFFLE_PARTITIONS} | "
            f"AQE: {cfg.spark.AQE_ENABLED}"
        )

    return _spark_session


def stop_spark(logger: Optional[logging.Logger] = None) -> None:
    """Gracefully stop the SparkSession."""
    global _spark_session
    if _spark_session is not None:
        _spark_session.stop()
        _spark_session = None
        if logger:
            logger.info("🛑 SparkSession stopped.")


# ────────────────────────────────────────────────────────────────────
# TIMING CONTEXT MANAGER
# ────────────────────────────────────────────────────────────────────
@contextmanager
def timer(stage_name: str, logger: logging.Logger):
    """
    Context manager that logs execution time for a pipeline stage.

    Usage:
        with timer("ETL Phase", logger):
            run_etl()
    """
    logger.info(f"⏱️  STAGE START: {stage_name}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        minutes, seconds = divmod(elapsed, 60)
        logger.info(
            f"⏱️  STAGE COMPLETE: {stage_name} | "
            f"Duration: {int(minutes)}m {seconds:.1f}s"
        )


# ────────────────────────────────────────────────────────────────────
# DATA VALIDATION HELPERS
# ────────────────────────────────────────────────────────────────────
def validate_path_exists(path: str, description: str = "Path") -> None:
    """Raise FileNotFoundError if path does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found: {path}. "
            f"Please verify the path in config.py."
        )


def validate_dataframe_not_empty(df, name: str = "DataFrame") -> None:
    """Raise ValueError if a Spark DataFrame is empty."""
    # Use limit(1) to avoid full count — memory safe
    if df.head(1) is None:
        raise ValueError(f"{name} is empty after processing. Aborting stage.")
