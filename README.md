# 🚖 NYC Taxi Surge Engine: High-Concurrency Predictive Modeling

[![Apache Spark](https://img.shields.io/badge/Apache-Spark-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Gradient Boosting](https://img.shields.io/badge/Model-XGBoost/LightGBM-orange)](https://github.com/carnoba)
[![Data Format](https://img.shields.io/badge/Data-Parquet-003B57)](https://parquet.apache.org/)

**NYC Taxi Surge Engine** is a professional-grade Big Data and Machine Learning system architected to solve the dynamic pricing challenges in urban mobility. This engine handles the massive NYC TLC dataset, processing millions of rows via Spark to predict real-time surge multipliers based on hyper-local demand/supply imbalances.

## 🚀 The Data Science Lifecycle

This project demonstrates a complete engineering lifecycle for high-volume data:
1. **Ingestion**: Raw Parquet data ingestion using Apache Spark for distributed processing.
2. **ETL & Engineering**: Advanced feature engineering including rolling window demand metrics and geospatial binning.
3. **Predictive Modeling**: High-precision gradient boosting models trained to estimate surge pricing with minimal latency.
4. **Interactive Simulation**: A real-time dashboard to visualize and interact with predictive surge scenarios.

## ✨ Premium Features

- **Scalable Spark Pipeline**: Efficiently processes multi-gigabyte Parquet files without memory bottlenecks.
- **Geospatial Intelligence**: Analyzes demand patterns across NYC's complex grid to identify "Saturation Zones."
- **Dynamic Pricing Algorithm**: Implements an asymmetric loss function to prioritize pricing accuracy during peak demand surges.
- **Visual Analytics Hub**: Interactive graphs and maps showcasing historical vs. predicted surge trends.

## 🛠 Tech Stack

- **Data Engineering**: Apache Spark, PySpark, Pandas
- **Machine Learning**: XGBoost, Scikit-learn
- **Visualization**: Streamlit / Plotly
- **Infrastructure**: Parquet Columnar Storage

## 📁 Repository Structure

```
├── spark_pipeline/       # Spark ETL and feature engineering logic
├── models/               # Trained surge prediction models
├── dashboard/            # Interactive Streamlit application
├── data/                 # Sample Parquet datasets (schema only)
└── README.md             # Documentation
```

## ⚙️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/carnoba/nyc-taxi-surge-engine.git
   ```
2. **Environment Setup**:
   Ensure you have a Java environment for Spark and install Python requirements:
   ```bash
   pip install pyspark xgboost streamlit pandas
   ```
3. **Trigger the Pipeline**:
   ```bash
   python spark_pipeline/orchestrate.py
   ```

## 🤝 Contributing

We welcome optimizations in the Spark execution plan or improvements to the surge pricing model. Feel free to fork and open a PR!

## ⭐ Star the Project!

If this engine helps your understanding of Big Data or Surge Pricing, please give it a **Star**! 🌟

---
**Engineered by [Carnoba](https://github.com/carnoba)**

#Tags
#DataEngineering #ApacheSpark #NYC #MachineLearning #SurgePricing #Python #BigData #PySpark #PredictiveModeling #UrbanMobility
