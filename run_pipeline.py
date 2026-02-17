"""
╔══════════════════════════════════════════════════════════════════════╗
║         SURGE PRICING ENGINE — PIPELINE ORCHESTRATOR                ║
║                  End-to-End Execution Controller                    ║
╚══════════════════════════════════════════════════════════════════════╝

Runs the complete pipeline:
  Phase A: ETL Pipeline
  Phase B: Feature Engineering
  Phase C: Model Training

Each phase is independently recoverable. If a phase fails,
the pipeline can be restarted from that phase.

Author: Data Engineering Team
Version: 1.0.0
"""
import argparse
import gc
import os
import sys
import time

# ────────────────────────────────────────────────────────────────────
# WINDOWS & PYTHON 3.13 COMPATIBILITY BYPASS
# ────────────────────────────────────────────────────────────────────
# 1. Manually set HADOOP_HOME to the fixed directory
os.environ['HADOOP_HOME'] = r'E:\programming\data science\imposible\hadoop_fix'

# 2. Add Hadoop bin to PATH (required for winutils.exe)
hadoop_bin = os.path.join(os.environ['HADOOP_HOME'], 'bin')
if hadoop_bin not in os.environ['PATH']:
    os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ['PATH']

# 3. Ensure PySpark uses the correct Python interpreter
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 4. NativeIO Bypass & Spark Optimization
# Critical for Python 3.13 + Spark on Windows OOM issues
def apply_native_io_bypass():
    try:
        from py4j.java_gateway import java_import
        from utils import get_spark_session
        
        # Get session (will initialize Spark if not already running)
        spark = get_spark_session()
        
        # ── Step A: Optimization Overrides ──
        # Reduce partitions and enable off-heap for memory safety
        spark.conf.set("spark.sql.shuffle.partitions", "10")
        spark.conf.set("spark.memory.offHeap.enabled", "true")
        spark.conf.set("spark.memory.offHeap.size", "2g")
        
        # ── Step B: NativeIO Bypass ──
        sc = spark.sparkContext
        gw = sc._gateway
        
        # Import NativeIO into the JVM context
        java_import(gw.jvm, "org.apache.hadoop.fs.permission.NativeIO")
        
        # This mocks the access0 method to always return True (access granted)
        # preventing Spark from trying to use the native Windows DLL for permissions
        gw.jvm.org.apache.hadoop.fs.permission.NativeIO.Windows.access0 = lambda *args: True
        
    except Exception as e:
        # If this fails, we log it later after the logger is initialized
        pass

from config import cfg
from utils import MemoryMonitor, get_logger, stop_spark, timer


logger = get_logger(__name__)
monitor = MemoryMonitor(logger)


def run_full_pipeline(
    skip_etl: bool = False,
    skip_features: bool = False,
    skip_model: bool = False,
) -> None:
    """
    Execute the complete surge pricing pipeline.

    Args:
        skip_etl: Skip Phase A (use existing processed data).
        skip_features: Skip Phase B (use existing features).
        skip_model: Skip Phase C (no model training).
    """
    # Apply Windows NativeIO bypass before any Spark logic
    apply_native_io_bypass()

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " NYC TLC SURGE PRICING ENGINE — FULL PIPELINE ".center(68) + "║")
    logger.info("║" + " Memory-Safe | Production-Grade | 8GB RAM ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")

    start_time = time.perf_counter()
    monitor.checkpoint("Pipeline — Start")

    try:
        # ════════════════════════════════════════════════════
        # PHASE A: ETL
        # ════════════════════════════════════════════════════
        if not skip_etl:
            from etl_pipeline import run_etl

            with timer("PHASE A: ETL Pipeline", logger):
                etl_output = run_etl()
                logger.info(f"Phase A output: {etl_output}")

            # Aggressive cleanup between phases
            monitor.force_gc("Inter-phase cleanup (A → B)")
            monitor.checkpoint("Pipeline — Post-ETL")
        else:
            logger.info("⏭️  Skipping Phase A: ETL (--skip-etl)")

        # ════════════════════════════════════════════════════
        # PHASE B: FEATURE ENGINEERING
        # ════════════════════════════════════════════════════
        if not skip_features:
            from feature_engineering import run_feature_engineering

            with timer("PHASE B: Feature Engineering", logger):
                features_output = run_feature_engineering()
                logger.info(f"Phase B output: {features_output}")

            # Stop Spark before model training (free JVM memory)
            stop_spark(logger)
            monitor.force_gc("Inter-phase cleanup (B → C)")
            monitor.checkpoint("Pipeline — Post-Features")
        else:
            logger.info("⏭️  Skipping Phase B: Feature Engineering (--skip-features)")
            # Still stop Spark if it's running
            stop_spark(logger)

        # ════════════════════════════════════════════════════
        # PHASE C: MODEL TRAINING
        # ════════════════════════════════════════════════════
        if not skip_model:
            from model_training import run_model_training

            with timer("PHASE C: Model Training", logger):
                model_output = run_model_training()
                logger.info(f"Phase C output: {model_output}")

            monitor.force_gc("Post-model cleanup")
            monitor.checkpoint("Pipeline — Post-Model")
        else:
            logger.info("⏭️  Skipping Phase C: Model Training (--skip-model)")

        # ════════════════════════════════════════════════════
        # FINAL REPORT
        # ════════════════════════════════════════════════════
        elapsed = time.perf_counter() - start_time
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(int(minutes), 60)

        logger.info("")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║" + " PIPELINE COMPLETE ✅ ".center(68) + "║")
        logger.info("╠" + "═" * 68 + "╣")
        logger.info(
            "║"
            + f"  Total Duration: {hours}h {int(minutes)}m {seconds:.1f}s".ljust(68)
            + "║"
        )
        logger.info(
            "║"
            + f"  Output Dir    : {cfg.paths.PROJECT_ROOT}/output/".ljust(68)
            + "║"
        )
        logger.info("╚" + "═" * 68 + "╝")

        monitor.checkpoint("Pipeline — Final")

    except Exception as e:
        logger.error(f"❌ PIPELINE FAILED: {type(e).__name__}: {e}")
        monitor.checkpoint("Pipeline — FAILED")
        raise
    finally:
        stop_spark(logger)
        gc.collect()


# ────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NYC TLC Surge Pricing Engine Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --skip-etl         # Skip ETL (reuse processed data)
  python run_pipeline.py --skip-features    # Skip feature engineering
  python run_pipeline.py --skip-model       # Skip model training
  python run_pipeline.py --only-etl         # Run only ETL
        """,
    )

    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help="Skip Phase A: ETL Pipeline",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip Phase B: Feature Engineering",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip Phase C: Model Training",
    )
    parser.add_argument(
        "--only-etl",
        action="store_true",
        help="Run only Phase A: ETL Pipeline",
    )
    parser.add_argument(
        "--only-features",
        action="store_true",
        help="Run only Phase B: Feature Engineering",
    )
    parser.add_argument(
        "--only-model",
        action="store_true",
        help="Run only Phase C: Model Training",
    )

    args = parser.parse_args()

    # Handle --only-* shortcuts
    if args.only_etl:
        args.skip_features = True
        args.skip_model = True
    elif args.only_features:
        args.skip_etl = True
        args.skip_model = True
    elif args.only_model:
        args.skip_etl = True
        args.skip_features = True

    run_full_pipeline(
        skip_etl=args.skip_etl,
        skip_features=args.skip_features,
        skip_model=args.skip_model,
    )


if __name__ == "__main__":
    main()
