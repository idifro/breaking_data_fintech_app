# Inference Pipeline Documentation

> **Production-Ready Batch Inference System**  
> **Daily Stock Price Predictions with Apache Airflow Orchestration**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Airflow DAG Workflow](#airflow-dag-workflow)
- [Inference Process](#inference-process)
- [Data Flow](#data-flow)
- [Performance Metrics](#performance-metrics)
- [Output Schema](#output-schema)
- [Usage Guide](#usage-guide)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Overview

The **Inference Pipeline** is a production-grade system that generates daily stock price predictions for **29 stocks** using the trained Spark RF model. It runs as part of a **4-stage Apache Airflow DAG** scheduled to execute daily at **23:00 UTC**.

### **Key Features**
✅ **Automated Daily Predictions**: Scheduled Airflow workflow  
✅ **Batch Processing**: 29 stocks predicted in ~120 seconds  
✅ **MLflow Integration**: Loads versioned production model (v1)  
✅ **Delta Lake Storage**: ACID-compliant prediction storage  
✅ **Feature Consistency**: Same feature engineering as training  
✅ **Scalable Architecture**: Distributed Spark processing  

### **Business Impact**
- **Next-day close price forecasts** for portfolio optimization
- **Model confidence scores** for risk assessment
- **Historical prediction tracking** for backtesting
- **Real-time integration** with FinPulse WebApp

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                   DAILY AIRFLOW DAG WORKFLOW                          │
│                   Schedule: 23:00 UTC (Daily)                         │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  STAGE 1        │
│  Extract        │  Extract latest stock prices (OHLCV)
│  YFinance       │  for 29 stocks from Yahoo Finance
│                 │
│  📥 Source:     │
│  • Yahoo API    │
│  • 29 stocks    │
│  • Auto_adjust  │
│                 │
│  📤 Output:     │
│  tmp/yfinance_  │
│  raw_*.parquet  │
└────────┬────────┘
         │
         │ 1-3 minutes (API calls)
         ▼
┌─────────────────────────────────┐
│  STAGE 2                        │
│  Enrich with Sentiment          │  Add sentiment scores from AstraDB
│                                 │  (pre-computed by Streaming Pipeline)
│  📥 Input:                      │
│  • yfinance_raw.parquet         │
│  • AstraDB (news_sentiment_1)   │
│                                 │
│  🔧 Process:                    │
│  • Join stock data + sentiment  │
│  • Calculate scaled_sentiment   │
│  • Add news_flag indicator      │
│                                 │
│  📤 Output:                     │
│  tmp/yfinance_enriched.parquet  │
│  (11 columns with sentiment)    │
└────────┬────────────────────────┘
         │
         │ ~30 seconds (DB query + join)
         ▼
┌───────────────────────────────────┐
│  STAGE 3                          │
│  Load into Delta                  │  Append to stock-specific tables
│                                   │
│  📥 Input:                        │
│  • yfinance_enriched.parquet      │
│                                   │
│  🔧 Process:                      │
│  • Validate schema (11 columns)   │
│  • Type casting to match Delta    │
│  • Per-stock table append         │
│                                   │
│  📤 Output:                       │
│  delta_tables/stock_<SYMBOL>/    │
│  (29 separate Delta tables)       │
└────────┬──────────────────────────┘
         │
         │ ~15 seconds (Delta append)
         ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 4 ⭐ INFERENCE BATCH                                  │
│  Predict Next-Day Close Prices                               │
│                                                               │
│  📥 Input:                                                    │
│  • delta_tables/stock_<SYMBOL>/ (29 tables)                  │
│  • MLflow Model: stock_predictor_spark_rf_unified_29stocks  │
│                                                               │
│  🔧 Process:                                                  │
│  1️⃣  Load Model & Scaler from MLflow                         │
│  2️⃣  Prepare Data (last 50 rows/stock)                       │
│  3️⃣  Feature Engineering (20+ features)                      │
│  4️⃣  Feature Scaling (StandardScaler)                        │
│  5️⃣  Make Predictions (Spark RF)                             │
│  6️⃣  Calculate Predicted Close Prices                        │
│  7️⃣  Save to Delta + CSV                                     │
│                                                               │
│  📤 Output:                                                   │
│  • delta_tables/stock_predictions/ (Delta table)             │
│  • results/batch_predictions_*.csv (optional)                │
│                                                               │
│  ⏱️  Performance: ~120 seconds for 29 stocks                 │
└───────────────────────────────────────────────────────────────┘
```

---

## Airflow DAG Workflow

### **DAG Configuration**

**File**: `scripts/stock_pipeline_dag.py`

```python
dag = DAG(
    dag_id="daily_stock_pipeline",
    schedule_interval="0 23 * * *",  # Daily at 23:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "owner": "data_eng",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }
)
```

### **Task Dependencies**

```
extract_task >> enrich_task >> load_task >> inference_task
```

| Task | Duration | Purpose | Retry Strategy |
|------|----------|---------|----------------|
| **extract_yfinance** | 1-3 min | Fetch OHLCV from Yahoo Finance | 2 retries, 5 min delay |
| **enrich_with_sentiment** | ~30 sec | Add GPT sentiment from AstraDB | 2 retries, 5 min delay |
| **load_into_delta** | ~15 sec | Append to Delta tables | 2 retries, 5 min delay |
| **inference_batch** | ~120 sec | Generate predictions | 2 retries, 5 min delay |

**Total Pipeline Duration**: ~5-7 minutes

### **Stock List** (29 Stocks)

```python
STOCK_LIST = [
    "AAL", "AAPL", "ABBV", "AMD", "AMGN", "BABA", "BIIB",
    "CMCSA", "CMG", "COP", "COST", "CRM", "CVX", "EBAY",
    "GE", "GOOG", "GSK", "MRK", "NKE", "NVDA", "ORCL",
    "PEP", "PYPL", "QCOM", "QQQ", "TSLA", "TSM", "USO", "WFC"
]
```

---

## Inference Process

### **Step-by-Step Flow**

#### **Step 1: Load Model & Scaler from MLflow**

```python
# Load production model from MLflow Model Registry
model_uri = "models:/stock_predictor_spark_rf_unified_29stocks/1"
model = mlflow.spark.load_model(model_uri)

# Load feature scaler (StandardScaler) from model artifacts
feature_scaler = StandardScalerModel.load(scaler_path)
```

**Model Metadata**:
- **Name**: `stock_predictor_spark_rf_unified_29stocks`
- **Version**: `1` (Production)
- **Algorithm**: Spark RandomForestRegressor
- **Artifact**: Feature scaler (StandardScalerModel)

#### **Step 2: Prepare Inference Data**

```python
# For each stock, get last 50 rows from Delta table
lookback_days = 50

# Filter to inference stocks (29 stocks)
df = load_data_from_delta()
df_filtered = df.filter(col("stock_symbol").isin(STOCK_LIST))

# Get latest rows per stock (for feature lags)
window_spec = Window.partitionBy("stock_symbol").orderBy(desc("Date"))
latest_df = df_filtered.withColumn("row_num", row_number().over(window_spec)) \
                       .filter(col("row_num") <= lookback_days)

# Keep only the most recent row for prediction
final_df = latest_df.filter(col("row_num") == 1)
```

**Data Source**: `delta_tables/stock_<SYMBOL>/`  
**Records Retrieved**: 29 (one per stock)

#### **Step 3: Feature Engineering**

Apply **identical feature engineering** as training:

```python
# Price-based features
- Close Lags: Close_lag_1, Close_lag_2, Close_lag_3, Close_lag_5
- Open Lags: Open_lag_1, Open_lag_2, Open_lag_3, Open_lag_5
- Moving Averages: MA_5, MA_10, MA_20
- Price Range: High - Low
- Daily Return: (Close - Close_lag_1) / Close_lag_1

# Volume features
- Volume Lags: Volume_lag_1, Volume_lag_2, Volume_lag_3, Volume_lag_5
- Volume MA: Volume_MA_5, Volume_MA_10, Volume_MA_20
- Volume Change: (Volume - Volume_lag_1) / Volume_lag_1

# Technical indicators
- RSI (Relative Strength Index)
- Volatility (20-day standard deviation)
- Bollinger Bands (upper, lower)

# Sentiment features
- Sentiment_gpt_lag_1, Sentiment_gpt_lag_3, Sentiment_gpt_lag_5
- Scaled_sentiment
- News_flag (binary)

# Interaction features
- Close × Volume
- Sentiment_gpt × Close
- MA_5 × Volume

Total: 20+ features
```

**Critical**: Feature engineering must match training exactly to ensure prediction accuracy.

#### **Step 4: Feature Scaling**

```python
# Assemble features into vector
assembler = VectorAssembler(
    inputCols=feature_names,  # 20+ features
    outputCol="features_raw"
)

# Apply StandardScaler (same scaler from training)
scaled_df = feature_scaler.transform(df_assembled)
```

**Scaler Type**: StandardScaler  
**Fitted On**: Training data (81K+ records)  
**Applied To**: Inference data (29 records)

#### **Step 5: Make Predictions**

```python
# Predict close gain (target: (Close_t+1 - Close_t) / Close_t)
predictions_df = model.transform(scaled_df)

# Output columns:
# - stock_symbol
# - last_date
# - last_close
# - predicted_gain (model output)
# - model_confidence (from config)
# - model_version (1)
# - features_used (comma-separated)
# - created_at (timestamp)
# - days_ahead (1)
```

**Prediction**: Gain ratio for next trading day

#### **Step 6: Calculate Predicted Close Prices**

```python
# Formula: predicted_close = last_close × (1 + predicted_gain)
predicted_close = col("last_close") * (lit(1) + col("predicted_gain"))

# Calculate next business day (skip weekends)
def get_next_business_day(date_val):
    next_day = date_val + timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip Sat/Sun
        next_day += timedelta(days=1)
    return next_day

prediction_date = get_next_business_day(last_date)
```

**Output**: Absolute price prediction for next trading day

#### **Step 7: Save Predictions**

```python
# Save to Delta table (append mode)
predictions_df.write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("delta_tables/stock_predictions/")

# Optional: Save to CSV for reporting
predictions_df.toPandas().to_csv(
    f"results/batch_predictions_{timestamp}.csv",
    index=False
)
```

**Storage**:
- **Primary**: Delta table (ACID transactions, time-travel)
- **Secondary**: CSV files (for FinPulse WebApp integration)

---

## Data Flow

### **Input Data Schema**

**Delta Table**: `delta_tables/stock_<SYMBOL>/`

| Column | Type | Description |
|--------|------|-------------|
| `Date` | Date | Trading date |
| `Open` | Double | Opening price |
| `High` | Double | Highest price |
| `Low` | Double | Lowest price |
| `Close` | Double | Closing price |
| `Volume` | Long | Trading volume |
| `Adj_close` | Double | Adjusted close price |
| `Sentiment_gpt` | Double | GPT sentiment score |
| `News_flag` | Integer | News availability (0/1) |
| `Scaled_sentiment` | Double | Normalized sentiment |
| `stock_symbol` | String | Stock ticker |

**Total Columns**: 11  
**Records per Stock**: ~2,800 (daily since 2015)

### **Output Data Schema**

**Delta Table**: `delta_tables/stock_predictions/`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `prediction_date` | Date | Target trading date | 2025-11-27 |
| `prediction_timestamp` | Timestamp | Prediction time (9:30 AM EST) | 2025-11-27 09:30:00 |
| `predicted_close` | Double | Predicted close price | 189.45 |
| `model_confidence` | Double | Confidence score (0-1) | 0.85 |
| `model_version` | String | Model version | "1" |
| `features_used` | String | Comma-separated feature list | "Close_lag_1,MA_5,..." |
| `created_at` | Timestamp | Prediction creation time | 2025-11-26 23:05:12 |
| `days_ahead` | Integer | Forecast horizon (always 1) | 1 |
| `stock_symbol` | String | Stock ticker | AAPL |

**Total Columns**: 9  
**Daily Inserts**: 29 rows (one per stock)

---

## Performance Metrics

### **Inference Performance (29 Stocks)**

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Inference Time** | ~120 seconds | Model load + prediction + save |
| **Throughput** | 0.24 stocks/sec | 29 stocks / 120 sec |
| **Average Time per Stock** | 4.14 seconds | Including feature engineering |
| **Peak Memory** | ~8 GB | Driver memory usage |
| **CPU Utilization** | ~40% | Local[8] cluster |

### **Breakdown by Stage**

| Stage | Duration | Percentage |
|-------|----------|-----------|
| **Model Loading** | ~15 sec | 12.5% |
| **Data Preparation** | ~30 sec | 25.0% |
| **Feature Engineering** | ~25 sec | 20.8% |
| **Feature Scaling** | ~10 sec | 8.3% |
| **Prediction** | ~20 sec | 16.7% |
| **Save to Delta** | ~15 sec | 12.5% |
| **CSV Export** | ~5 sec | 4.2% |

### **Resource Utilization**

**Spark Configuration**:
```yaml
spark.driver.memory: 8g
spark.executor.memory: 6g
spark.master: local[8]
spark.sql.adaptive.enabled: true
spark.ml.cache.enabled: true
```

**System Resources**:
- **Memory**: 8 GB driver, 6 GB executors
- **Cores**: 8 cores (local mode)
- **Disk I/O**: Delta table reads/writes

### **Scalability Estimates**

| Stock Count | Estimated Time | Throughput |
|-------------|----------------|------------|
| 10 stocks | ~45 sec | 0.22 stocks/sec |
| 29 stocks | ~120 sec | 0.24 stocks/sec |
| 50 stocks | ~205 sec | 0.24 stocks/sec |
| 100 stocks | ~415 sec | 0.24 stocks/sec |

**Observation**: Near-linear scalability due to distributed processing.

---

## Output Schema

### **Delta Table Structure**

```
delta_tables/
└── stock_predictions/
    ├── _delta_log/
    │   ├── 00000000000000000000.json
    │   ├── 00000000000000000001.json
    │   └── ...
    ├── part-00000-*.parquet
    ├── part-00001-*.parquet
    └── ...
```

### **CSV Output** (Optional)

**File**: `results/batch_predictions_YYYYMMDD_HHMMSS.csv`

**Example**:
```csv
prediction_date,prediction_timestamp,predicted_close,model_confidence,model_version,features_used,created_at,days_ahead,stock_symbol
2025-11-27,2025-11-27 09:30:00,189.45,0.85,1,"Close_lag_1,MA_5,...",2025-11-26 23:05:12,1,AAPL
2025-11-27,2025-11-27 09:30:00,142.78,0.85,1,"Close_lag_1,MA_5,...",2025-11-26 23:05:12,1,GOOG
2025-11-27,2025-11-27 09:30:00,287.33,0.85,1,"Close_lag_1,MA_5,...",2025-11-26 23:05:12,1,TSLA
...
```

### **Query Examples**

```sql
-- Read latest predictions from Delta table
SELECT *
FROM delta.`delta_tables/stock_predictions/`
WHERE prediction_date = '2025-11-27'
ORDER BY stock_symbol;

-- Time-travel: view predictions from specific date
SELECT *
FROM delta.`delta_tables/stock_predictions/`
VERSION AS OF 5
WHERE stock_symbol = 'AAPL';

-- Aggregate statistics
SELECT 
    stock_symbol,
    AVG(predicted_close) as avg_prediction,
    COUNT(*) as prediction_count
FROM delta.`delta_tables/stock_predictions/`
GROUP BY stock_symbol;
```

---

## Usage Guide

### **Manual Execution**

```bash
# Activate environment
conda activate breaking_data

# Run inference batch script directly
python scripts/inference_batch.py

# Expected output:
# 🚀 Starting Batch Inference for Stock Price Prediction
# 🔧 Configuration loaded successfully
# 📊 Inference stocks: 29
# 🔧 Spark session created successfully
# 📥 Loading model: models:/stock_predictor_spark_rf_unified_29stocks/1
# ✅ Model loaded successfully: stock_predictor_spark_rf_unified_29stocks v1
# 📊 Preparing data for 29 stocks, lookback: 50 days
# ✅ Prepared 29 records for prediction
# 🎯 Making predictions...
# ✅ Predictions generated successfully
# ✅ Predictions saved to Delta table: delta_tables/stock_predictions/
# 🎉 Batch Inference Completed Successfully!
```

### **Airflow Execution**

```bash
# Initialize Airflow (first time only)
export AIRFLOW_HOME=~/airflow
airflow db init
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Copy DAG file
cp scripts/stock_pipeline_dag.py ~/airflow/dags/

# Start Airflow services
airflow scheduler &
airflow webserver --port 8080 &

# Trigger DAG manually
airflow dags trigger daily_stock_pipeline

# Check task status
airflow tasks list daily_stock_pipeline
airflow tasks test daily_stock_pipeline inference_batch 2025-11-26
```

### **Access Airflow UI**

```bash
# Open browser
http://localhost:8080

# Default credentials: admin / admin

# Navigate to:
# - DAGs → daily_stock_pipeline
# - Graph View: See task dependencies
# - Tree View: See historical runs
# - Logs: Click on tasks to view logs
```

### **View Predictions**

```python
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# Create Spark session
builder = SparkSession.builder \
    .appName("ViewPredictions") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", 
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Read predictions
predictions = spark.read.format("delta").load("delta_tables/stock_predictions/")

# Show latest predictions
predictions.filter(col("prediction_date") == "2025-11-27") \
           .select("stock_symbol", "predicted_close", "model_confidence") \
           .orderBy("stock_symbol") \
           .show(29, truncate=False)

# Convert to Pandas for analysis
import pandas as pd
pdf = predictions.toPandas()
print(pdf.head())
```

### **Integration with FinPulse WebApp**

```python
# FinPulse reads predictions from Delta table
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/predictions/{stock_symbol}")
def get_stock_prediction(stock_symbol: str):
    """Get latest prediction for a stock"""
    predictions = spark.read.format("delta").load("delta_tables/stock_predictions/")
    
    latest = predictions.filter(col("stock_symbol") == stock_symbol) \
                       .orderBy(desc("created_at")) \
                       .limit(1) \
                       .toPandas()
    
    return {
        "stock": stock_symbol,
        "predicted_close": float(latest["predicted_close"].iloc[0]),
        "confidence": float(latest["model_confidence"].iloc[0]),
        "prediction_date": str(latest["prediction_date"].iloc[0])
    }
```

---

## Monitoring & Troubleshooting

### **MLflow Tracking**

```bash
# Start MLflow UI
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri ./mlflow_tracking \
    --default-artifact-root ./mlflow_tracking

# View:
# - Registered Models → stock_predictor_spark_rf_unified_29stocks
# - Model Version: 1 (Production)
# - Model Artifacts: feature_scaler, model files
```

## Summary

### **Key Takeaways**

1. **Production-ready workflow**
   - 4-stage Airflow DAG
   - Scheduled daily at 23:00 UTC
   - 2 retries with 5-minute delay

2. **Efficient batch processing**
   - 29 stocks in ~120 seconds
   - Near-linear scalability
   - Low resource footprint (8 GB)

3. **MLflow integration**
   - Versioned model loading (v1)
   - Artifact management (feature scaler)
   - Model registry integration

4. **ACID-compliant storage**
   - Delta Lake predictions table
   - Time-travel capability
   - Schema evolution support

5. **FinPulse integration**
   - CSV exports for WebApp
   - REST API access to predictions
   - Real-time dashboard updates

---

---

## References

- **Inference Script**: `scripts/inference_batch.py`
- **Airflow DAG**: `scripts/stock_pipeline_dag.py`
- **Training Pipeline**: See `TRAINING_PIPELINE.md`
- **MLflow Experiments**: `mlflow_tracking/` directory
- **Delta Tables**: `delta_tables/stock_predictions/`

---
