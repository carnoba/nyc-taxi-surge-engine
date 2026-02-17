# ══════════════════════════════════════════════════════════════════════
# SURGE PRICING ENGINE — Dockerfile
# ══════════════════════════════════════════════════════════════════════
# Multi-stage build for a memory-safe PySpark + XGBoost environment.
# Designed to run within the 8GB RAM constraint.

FROM python:3.11-slim AS base

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Java environment ──
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ── Working directory ──
WORKDIR /app

# ── Install Python dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ──
COPY config.py .
COPY utils.py .
COPY weather_data.py .
COPY etl_pipeline.py .
COPY feature_engineering.py .
COPY model_training.py .
COPY run_pipeline.py .
COPY tests/ tests/

# ── Create spill directory ──
RUN mkdir -p /tmp/spark_spill

# ── Override spill dir for container ──
ENV SPARK_LOCAL_DIR=/tmp/spark_spill

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import config; print('Config OK')" || exit 1

# ── Default entrypoint ──
ENTRYPOINT ["python", "run_pipeline.py"]
