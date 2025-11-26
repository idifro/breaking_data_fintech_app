# Apache Airflow Setup & DAG Documentation

> **Daily Stock Prediction Pipeline Orchestration**  
> **4-Stage Workflow: Extract → Enrich → Load → Inference**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [DAG Workflow](#dag-workflow)
- [Task Details](#task-details)
- [Scheduling](#scheduling)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Commands Reference](#commands-reference)

---

## Overview

Apache Airflow orchestrates the **daily stock prediction pipeline**, executing a 4-stage workflow every day at **23:00 UTC** (11 PM). The pipeline processes **29 stocks**, enriching price data with sentiment analysis and generating next-day predictions.

### **Pipeline Purpose**

✅ **Automated Data Collection**: Fetch daily stock prices from YFinance  
✅ **Sentiment Enrichment**: Integrate news sentiment from AstraDB  
✅ **Delta Lake Storage**: Append to ACID-compliant tables  
✅ **Batch Inference**: Generate next-day predictions using production model  

### **Key Metrics**

| Metric | Value |
|--------|-------|
| **Stocks Processed** | 29 |
| **Daily Runtime** | ~5-10 minutes |
| **Schedule** | Daily at 23:00 UTC |
| **Retries** | 2 attempts per task |
| **Retry Delay** | 5 minutes |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AIRFLOW DAG ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  Airflow Scheduler│
                         │  (Background)     │
                         │                   │
                         │  Triggers DAG at  │
                         │  23:00 UTC daily  │
                         └────────┬──────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  DAG: daily_stock_pipeline   │
                    │  (4 Tasks in Sequence)       │
                    └──────────────┬───────────────┘
                                   │
                ┌──────────────────┼──────────────────┬──────────────────┐
                │                  │                  │                  │
                ▼                  ▼                  ▼                  ▼
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
        │  TASK 1       │  │  TASK 2       │  │  TASK 3       │  │  TASK 4      │
        │  Extract      │→ │  Enrich       │→ │  Load         │→ │  Inference   │
        │               │  │               │  │               │  │              │
        │  extract_     │  │  enrich_with_ │  │  load_into_   │  │  inference_  │
        │  yfinance     │  │  sentiment    │  │  delta        │  │  batch       │
        └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬──────┘
                │                  │                  │                  │
                ▼                  ▼                  ▼                  ▼

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TASK 1: Extract YFinance Data                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Script: src/extract_yfinance.py                                               │
│  Function: run_extract_yfinance(output_path, stocks)                           │
│                                                                                 │
│  Input:                                                                         │
│    • stocks: List of 29 stock symbols (AAPL, GOOG, TSLA, etc.)                │
│                                                                                 │
│  Process:                                                                       │
│    1. Query YFinance API for yesterday's data (T-1)                            │
│    2. Fetch OHLCV data: Open, High, Low, Close, Volume, Adj_close             │
│    3. Combine all 29 stocks into single DataFrame                              │
│    4. Save to Parquet: tmp/yfinance_raw_{{ ds }}.parquet                       │
│                                                                                 │
│  Output:                                                                        │
│    • File: tmp/yfinance_raw_2025-11-26.parquet                                 │
│    • Schema: stock_symbol, Date, Open, High, Low, Close, Volume, Adj_close    │
│    • Rows: 29 (1 per stock)                                                    │
│                                                                                 │
│  Duration: ~30 seconds                                                          │
│                                                                                 │
│  Example Log:                                                                   │
│    [2025-11-26 23:00:15] INFO - [EXTRACT] ✅ Fetched AAPL: Close=180.25        │
│    [2025-11-26 23:00:18] INFO - [EXTRACT] ✅ Fetched GOOG: Close=142.50        │
│    ...                                                                          │
│    [2025-11-26 23:00:45] INFO - [EXTRACT] ✅ Saved 29 rows to                  │
│                                  tmp/yfinance_raw_2025-11-26.parquet           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                       TASK 2: Enrich with Sentiment                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Script: src/compute_sentiment_columns.py                                      │
│  Function: run_sentiment_enrichment(yfinance_path, output_path)                │
│                                                                                 │
│  Input:                                                                         │
│    • yfinance_path: tmp/yfinance_raw_2025-11-26.parquet                        │
│    • 29 rows with price data                                                   │
│                                                                                 │
│  Process:                                                                       │
│    1. Load YFinance data from Parquet                                          │
│    2. Query AstraDB for sentiment (news_sentiment_1 table)                     │
│    3. Extract event_date from published_at column (YYYY-MM-DD)                 │
│    4. Filter AstraDB rows to match (stock, date) pairs in YFinance             │
│    5. Group by (stock_symbol, date) and calculate mean sentiment_score         │
│    6. Left join YFinance + Sentiment (preserves rows with no news)             │
│    7. Add 3 columns:                                                            │
│       • Sentiment_gpt: Raw score (1-5) or 0 if no news                         │
│       • News_flag: Binary (1 if news exists, 0 otherwise)                      │
│       • Scaled_sentiment: (Sentiment_gpt - 0.9999) / 4 if score != 0, else 0  │
│    8. Save to Parquet: tmp/yfinance_enriched_{{ ds }}.parquet                  │
│                                                                                 │
│  Output:                                                                        │
│    • File: tmp/yfinance_enriched_2025-11-26.parquet                            │
│    • Schema: stock_symbol, Date, Open, High, Low, Close, Volume, Adj_close,   │
│              Sentiment_gpt, News_flag, Scaled_sentiment                        │
│    • Rows: 29 (1 per stock)                                                    │
│                                                                                 │
│  Duration: ~1-2 minutes                                                         │
│                                                                                 │
│  Example Log:                                                                   │
│    [2025-11-26 23:01:00] INFO - [ENRICH] Loaded 29 rows from YFinance          │
│    [2025-11-26 23:01:05] INFO - [ENRICH] Querying AstraDB for 29 stocks        │
│    [2025-11-26 23:01:15] INFO - [ENRICH] Found 450 sentiment records           │
│    [2025-11-26 23:01:20] INFO - [ENRICH] Computing daily aggregates            │
│    [2025-11-26 23:01:25] INFO - [ENRICH] ✅ Added sentiment to 22/29 stocks    │
│    [2025-11-26 23:01:30] INFO - [ENRICH] ✅ Saved to                            │
│                                  tmp/yfinance_enriched_2025-11-26.parquet      │
│                                                                                 │
│  Credential Loading (Fallback Chain):                                          │
│    1. Try: config/config.yaml (astradb section)                                │
│    2. If missing, try: Airflow Variables (ASTRA_DB_ENDPOINT, ASTRA_DB_TOKEN)  │
│    3. If missing, try: Environment Variables                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TASK 3: Load into Delta Lake                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Script: src/load_delta.py                                                     │
│  Function: load_to_delta(input_path, delta_base_path)                          │
│                                                                                 │
│  Input:                                                                         │
│    • input_path: tmp/yfinance_enriched_2025-11-26.parquet                      │
│    • delta_base_path: delta_tables/                                            │
│    • 29 rows (1 per stock) with 11 columns                                     │
│                                                                                 │
│  Process:                                                                       │
│    1. Initialize Spark session (local[8], 12g driver memory)                   │
│    2. Load enriched Parquet file into Spark DataFrame                          │
│    3. For each stock (29 iterations):                                          │
│       a. Filter row for stock (1 row)                                          │
│       b. Cast columns to Delta schema types (double, bigint, string, date)     │
│       c. Append to delta_tables/stock_<SYMBOL>/ using Delta merge              │
│       d. Auto-create table if first run                                        │
│    4. Log success/failure for each stock                                       │
│                                                                                 │
│  Output:                                                                        │
│    • 29 Delta tables: delta_tables/stock_AAPL/, delta_tables/stock_GOOG/, ...  │
│    • Schema (11 columns):                                                       │
│        stock_symbol (STRING), Date (DATE), Open (DOUBLE), High (DOUBLE),       │
│        Low (DOUBLE), Close (DOUBLE), Volume (BIGINT), Adj_close (DOUBLE),      │
│        Sentiment_gpt (DOUBLE), News_flag (INT), Scaled_sentiment (DOUBLE)      │
│    • Each table appended with 1 new row                                        │
│                                                                                 │
│  Duration: ~2-3 minutes                                                         │
│                                                                                 │
│  Example Log:                                                                   │
│    [2025-11-26 23:02:00] INFO - [LOAD] Starting Spark session                  │
│    [2025-11-26 23:02:10] INFO - [LOAD] Loaded 29 rows from Parquet             │
│    [2025-11-26 23:02:15] INFO - [DELTA] ✅ Appended 1 row to stock_AAPL        │
│    [2025-11-26 23:02:20] INFO - [DELTA] ✅ Appended 1 row to stock_GOOG        │
│    ...                                                                          │
│    [2025-11-26 23:04:50] INFO - [DELTA] ✅ All 29 tables updated successfully   │
│                                                                                 │
│  ACID Guarantees:                                                               │
│    • Atomic appends (all-or-nothing per stock)                                 │
│    • Time travel support (query historical versions)                           │
│    • Schema evolution (auto-add columns if needed)                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TASK 4: Batch Inference                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Script: scripts/inference_batch.py                                            │
│  Function: run_inference_batch()                                               │
│                                                                                 │
│  Input:                                                                         │
│    • Delta tables: delta_tables/stock_AAPL/, delta_tables/stock_GOOG/, ...    │
│    • MLflow model: stock_predictor_unified_29stocks (latest version)          │
│                                                                                 │
│  Process:                                                                       │
│    1. Load production model from MLflow (Spark RandomForest)                   │
│    2. For each stock (29 iterations):                                          │
│       a. Read last 50 rows from Delta table (sequence_length=50)              │
│       b. Validate data quality (continuity, missing values)                    │
│       c. Engineer features (MA_5, MA_10, RSI, Volatility, Price_change%, etc.)│
│       d. Create lag features (price_lag_1, sentiment_lag_1, volume_lag_1)     │
│       e. Scale features using StandardScaler                                   │
│       f. Make prediction for next trading day (T+1)                            │
│       g. Calculate predicted_close_price from predicted_gain%                  │
│    3. Combine all 29 predictions into single DataFrame                         │
│    4. Save to delta_tables/predictions/daily_predictions_{{ ds }}/             │
│    5. Log metrics to MLflow (inference_time, throughput, data_quality)         │
│                                                                                 │
│  Output:                                                                        │
│    • File: delta_tables/predictions/daily_predictions_2025-11-26/              │
│    • Schema (9 columns):                                                        │
│        stock_symbol (STRING), prediction_date (DATE), input_close (DOUBLE),    │
│        predicted_gain_percent (DOUBLE), predicted_close_price (DOUBLE),        │
│        lower_bound (DOUBLE), upper_bound (DOUBLE),                             │
│        confidence_score (DOUBLE), last_updated (TIMESTAMP)                     │
│    • Rows: 29 (1 per stock)                                                    │
│                                                                                 │
│  Duration: ~2-3 minutes                                                         │
│                                                                                 │
│  Example Log:                                                                   │
│    [2025-11-26 23:05:00] INFO - [INFERENCE] Loading model from MLflow          │
│    [2025-11-26 23:05:10] INFO - [INFERENCE] Model: stock_predictor_unified_v3  │
│    [2025-11-26 23:05:15] INFO - [INFERENCE] ✅ AAPL: Predicted +2.3%           │
│                                             (182.25 → 186.44)                  │
│    [2025-11-26 23:05:20] INFO - [INFERENCE] ✅ GOOG: Predicted +1.1%           │
│                                             (142.50 → 144.07)                  │
│    ...                                                                          │
│    [2025-11-26 23:07:45] INFO - [INFERENCE] ✅ 29 predictions saved             │
│    [2025-11-26 23:07:50] INFO - [INFERENCE] Throughput: 14.2 stocks/min        │
│    [2025-11-26 23:07:55] INFO - [INFERENCE] Logged to MLflow run abc123        │
│                                                                                 │
│  Quality Checks:                                                                │
│    • Data continuity: Ensure no gaps in last 50 days                           │
│    • Missing values: Max 10% allowed per feature                               │
│    • Skip weekends/holidays: Only predict for trading days                     │
└─────────────────────────────────────────────────────────────────────────────────┘

                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   Pipeline Complete      │
                    │   (Total: 5-10 minutes)  │
                    │                          │
                    │   ✅ 29 stocks processed  │
                    │   ✅ Delta tables updated │
                    │   ✅ Predictions saved    │
                    └──────────────────────────┘
```

---

## Installation

### **1. Install Apache Airflow**

```bash
# Activate conda environment
conda activate breaking_data

# Install Airflow with constraints (Python 3.9+)
pip install "apache-airflow==2.7.3" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-3.9.txt"

# Install additional providers
pip install apache-airflow-providers-amazon
pip install apache-airflow-providers-apache-spark
```

### **2. Initialize Airflow Database**

```bash
# Set AIRFLOW_HOME (default: ~/airflow)
export AIRFLOW_HOME=~/airflow

# Initialize database (SQLite by default)
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin123
```

### **3. Configure DAG Directory**

```bash
# Copy DAG file to Airflow DAGs folder
mkdir -p ~/airflow/dags
cp scripts/stock_pipeline_dag.py ~/airflow/dags/

# Verify DAG is detected
airflow dags list | grep daily_stock_pipeline
```

### **4. Set Airflow Variables** (For AstraDB Credentials)

```bash
# Option 1: CLI
airflow variables set ASTRA_DB_ENDPOINT "https://YOUR-DB-ID.apps.astra.datastax.com"
airflow variables set ASTRA_DB_TOKEN "AstraCS:..."

# Option 2: Web UI (Admin → Variables)
# Add key-value pairs in the UI

# Option 3: JSON import
cat > /tmp/airflow_vars.json <<EOF
{
  "ASTRA_DB_ENDPOINT": "https://YOUR-DB-ID.apps.astra.datastax.com",
  "ASTRA_DB_TOKEN": "AstraCS:..."
}
EOF

airflow variables import /tmp/airflow_vars.json
```

---

## Configuration

### **DAG Configuration** (scripts/stock_pipeline_dag.py)

```python
# Stock list (29 stocks)
STOCK_LIST = [
    "AAL", "AAPL", "ABBV", "AMD", "AMGN", "BABA", "BIIB",
    "CMCSA", "CMG", "COP", "COST", "CRM", "CVX", "EBAY",
    "GE", "GOOG", "GSK", "MRK", "NKE", "NVDA", "ORCL",
    "PEP", "PYPL", "QCOM", "QQQ", "TSLA", "TSM", "USO", "WFC"
]

# Paths (with Airflow template variables)
YF_RAW_PATH = "tmp/yfinance_raw_{{ ds }}.parquet"        # {{ ds }} = YYYY-MM-DD
YF_ENRICHED_PATH = "tmp/yfinance_enriched_{{ ds }}.parquet"
DELTA_BASE_PATH = "delta_tables"

# Default arguments
default_args = {
    "owner": "data_eng",
    "depends_on_past": False,  # Don't wait for previous runs
    "retries": 2,              # Retry failed tasks 2 times
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    dag_id="daily_stock_pipeline",
    default_args=default_args,
    schedule_interval="0 23 * * *",  # Daily at 23:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't backfill past dates
)
```

### **Airflow Configuration** (~/airflow/airflow.cfg)

```ini
[core]
# DAGs folder
dags_folder = /home/mha2cob/airflow/dags

# Parallel task execution
parallelism = 16
dag_concurrency = 16
max_active_runs_per_dag = 1

# Logging
base_log_folder = /home/mha2cob/airflow/logs
logging_level = INFO

[scheduler]
# Interval to scan DAGs folder
dag_dir_list_interval = 300

# Scheduler heartbeat
scheduler_heartbeat_sec = 5

# Number of processes
max_threads = 2

[webserver]
# Web UI settings
web_server_port = 8080
base_url = http://localhost:8080

# Authentication
authenticate = True
auth_backend = airflow.api.auth.backend.basic_auth

[email]
# Email alerts (optional)
email_backend = airflow.utils.email.send_email_smtp
smtp_host = smtp.gmail.com
smtp_port = 587
smtp_user = your_email@gmail.com
smtp_password = your_app_password
```

---

## DAG Workflow

### **Task Dependencies**

```python
extract_task >> enrich_task >> load_task >> inference_task
```

**Sequential Execution**:
1. **Extract** must complete before **Enrich**
2. **Enrich** must complete before **Load**
3. **Load** must complete before **Inference**

**Rationale**: Each task depends on output of previous task.

### **Task Definitions**

```python
# TASK 1: Extract YFinance
extract_task = PythonOperator(
    task_id="extract_yfinance",
    python_callable=run_extract_yfinance,
    op_kwargs={
        "output_path": YF_RAW_PATH,
        "stocks": STOCK_LIST,
    },
    dag=dag,
)

# TASK 2: Enrich with Sentiment
enrich_task = PythonOperator(
    task_id="enrich_with_sentiment",
    python_callable=run_sentiment_enrichment,
    op_kwargs={
        "yfinance_path": YF_RAW_PATH,
        "output_path": YF_ENRICHED_PATH,
    },
    dag=dag,
)

# TASK 3: Load into Delta
load_task = PythonOperator(
    task_id="load_into_delta",
    python_callable=load_to_delta,
    op_kwargs={
        "input_path": YF_ENRICHED_PATH,
        "delta_base_path": DELTA_BASE_PATH,
    },
    dag=dag,
)

# TASK 4: Batch Inference
inference_task = PythonOperator(
    task_id="inference_batch",
    python_callable=run_inference_batch,
    dag=dag,
)
```

---

## Task Details

### **Task 1: extract_yfinance**

**Script**: `src/extract_yfinance.py`

**Key Function**:
```python
def run_extract_yfinance(output_path: str, stocks: List[str]):
    """
    Extract yesterday's stock data from YFinance API.
    
    Args:
        output_path: Where to save Parquet file
        stocks: List of stock symbols
    
    Returns:
        Path to saved file
    """
    import yfinance as yf
    
    all_data = []
    for symbol in stocks:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            row = {
                "stock_symbol": symbol,
                "Date": hist.index[0],
                "Open": hist["Open"].iloc[0],
                "High": hist["High"].iloc[0],
                "Low": hist["Low"].iloc[0],
                "Close": hist["Close"].iloc[0],
                "Volume": hist["Volume"].iloc[0],
                "Adj_close": hist["Close"].iloc[0],  # YFinance doesn't split Adj Close
            }
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    df.to_parquet(output_path)
    return output_path
```

**Output Example**:
```
stock_symbol | Date       | Open   | High   | Low    | Close  | Volume    | Adj_close
-------------|------------|--------|--------|--------|--------|-----------|----------
AAPL         | 2025-11-26 | 179.50 | 181.25 | 178.90 | 180.25 | 52000000  | 180.25
GOOG         | 2025-11-26 | 141.30 | 143.20 | 140.80 | 142.50 | 28000000  | 142.50
```

### **Task 2: enrich_with_sentiment**

**Script**: `src/compute_sentiment_columns.py`

**Key Function**:
```python
def run_sentiment_enrichment(yfinance_path: str, output_path: str):
    """
    Enrich YFinance data with sentiment from AstraDB.
    
    Process:
    1. Load YFinance Parquet
    2. Query AstraDB for sentiment
    3. Join on (stock_symbol, date)
    4. Add Sentiment_gpt, News_flag, Scaled_sentiment
    5. Save enriched Parquet
    """
    # Load YFinance data
    yf_df = pd.read_parquet(yfinance_path)
    
    # Load sentiment from AstraDB
    astra_df = load_astra_rows(stocks=yf_df["stock_symbol"].unique().tolist())
    
    # Compute daily sentiments
    enriched_df = compute_daily_sentiments(yf_df, astra_df)
    
    # Save
    enriched_df.to_parquet(output_path)
    return output_path
```

**Output Example**:
```
stock_symbol | Date       | Close  | Sentiment_gpt | News_flag | Scaled_sentiment
-------------|------------|--------|---------------|-----------|------------------
AAPL         | 2025-11-26 | 180.25 | 4.5           | 1         | 0.8750625
GOOG         | 2025-11-26 | 142.50 | 3.0           | 1         | 0.5000625
TSLA         | 2025-11-26 | 265.80 | 0.0           | 0         | 0.0
```

### **Task 3: load_into_delta**

**Script**: `src/load_delta.py`

**Key Function**:
```python
def load_to_delta(input_path: str, delta_base_path: str):
    """
    Append enriched data to Delta tables.
    
    Process:
    1. Load enriched Parquet into Spark
    2. For each stock, append 1 row to Delta table
    3. Auto-create table if doesn't exist
    """
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(input_path)
    
    for stock in df.select("stock_symbol").distinct().collect():
        symbol = stock.stock_symbol
        stock_df = df.filter(col("stock_symbol") == symbol)
        
        delta_path = f"{delta_base_path}/stock_{symbol}"
        
        # Cast to correct types
        stock_df = stock_df.select(
            col("stock_symbol").cast("string"),
            col("Date").cast("date"),
            col("Open").cast("double"),
            # ... (all 11 columns)
        )
        
        # Append to Delta
        stock_df.write.format("delta").mode("append").save(delta_path)
```

**Output**: 29 Delta tables updated with 1 new row each.

### **Task 4: inference_batch**

**Script**: `scripts/inference_batch.py`

**Key Function**:
```python
def run_inference_batch():
    """
    Generate predictions for all 29 stocks.
    
    Process:
    1. Load MLflow model
    2. For each stock:
       - Read last 50 rows from Delta
       - Engineer features
       - Make prediction
    3. Save predictions to Delta
    """
    # Load model
    model_uri = "models:/stock_predictor_unified_29stocks/latest"
    model = mlflow.spark.load_model(model_uri)
    
    predictions = []
    for stock in STOCKS:
        # Read Delta table
        delta_df = spark.read.format("delta").load(f"delta_tables/stock_{stock}")
        last_50 = delta_df.orderBy(col("Date").desc()).limit(50)
        
        # Feature engineering
        features_df = engineer_features(last_50)
        
        # Predict
        pred_df = model.transform(features_df)
        predictions.append(pred_df.select("stock_symbol", "predicted_gain_percent"))
    
    # Save predictions
    all_preds = spark.createDataFrame(predictions)
    all_preds.write.format("delta").save("delta_tables/predictions/")
```

---

## Scheduling

### **Cron Expression**

```
0 23 * * *
```

**Breakdown**:
- `0`: Minute 0
- `23`: Hour 23 (11 PM)
- `*`: Every day of month
- `*`: Every month
- `*`: Every day of week

**Execution Time**: **23:00 UTC daily** (11 PM UTC)

### **Why 23:00 UTC?**

- **Markets Close**: US markets close at 16:00 ET (21:00 UTC)
- **Data Availability**: YFinance updates ~1 hour after close
- **Off-Peak Processing**: Minimal system load at night

### **Change Schedule**

```python
# Run every 6 hours
schedule_interval="0 */6 * * *"

# Run weekdays only (Mon-Fri)
schedule_interval="0 23 * * 1-5"

# Run manually only
schedule_interval=None
```

---

## Monitoring

### **1. Airflow Web UI**

```bash
# Start web server
airflow webserver --port 8080

# Access UI
http://localhost:8080

# Login with admin credentials
Username: admin
Password: admin123
```

**Key Views**:
- **DAGs**: View all DAGs, toggle on/off
- **Grid**: Task execution history (success/fail)
- **Graph**: Dependency visualization
- **Task Duration**: Performance trends
- **Logs**: Task-level logs

### **2. DAG Status Commands**

```bash
# List all DAGs
airflow dags list

# Show DAG structure
airflow dags show daily_stock_pipeline

# Trigger DAG manually
airflow dags trigger daily_stock_pipeline

# Pause/unpause DAG
airflow dags pause daily_stock_pipeline
airflow dags unpause daily_stock_pipeline

# List DAG runs
airflow dags list-runs -d daily_stock_pipeline

# Clear failed tasks (retry)
airflow tasks clear daily_stock_pipeline -s 2025-11-26 -e 2025-11-26
```

### **3. Task Monitoring**

```bash
# Test single task
airflow tasks test daily_stock_pipeline extract_yfinance 2025-11-26

# List task instances
airflow tasks list daily_stock_pipeline

# View task logs
airflow tasks logs daily_stock_pipeline extract_yfinance 2025-11-26 1
```

### **4. Scheduler Status**

```bash
# Start scheduler
airflow scheduler

# Check scheduler health
airflow jobs check --job-type SchedulerJob --hostname $(hostname)

# View scheduler logs
tail -f ~/airflow/logs/scheduler/latest/scheduler.log
```

### **5. MLflow Integration**

Task 4 (Inference) logs metrics to MLflow:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:///path/to/mlflow_tracking

# Access UI
http://localhost:5000

# View inference experiment
Experiment: pipeline_scalability_monitoring_inference
```

**Metrics Logged**:
- `inference_time_seconds`: Total runtime
- `throughput_stocks_per_second`: Processing rate
- `data_quality_score`: % stocks passing quality checks
- `predictions_saved`: Number of predictions

---

## Troubleshooting

### **Common Issues**

#### **Issue 1: DAG Not Showing in UI**

```bash
# Check DAG is in correct folder
ls ~/airflow/dags/stock_pipeline_dag.py

# Check for syntax errors
python ~/airflow/dags/stock_pipeline_dag.py

# Restart scheduler
pkill -f "airflow scheduler"
airflow scheduler &
```

#### **Issue 2: Task Fails with Import Error**

```bash
# Add project root to PYTHONPATH in DAG file
import sys
from pathlib import Path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))
```

**Or** set globally:
```bash
export PYTHONPATH="/home/mha2cob/Downloads/breaking_data/spark_ml_pipeline:$PYTHONPATH"
```

#### **Issue 3: AstraDB Connection Fails**

```bash
# Verify credentials
airflow variables get ASTRA_DB_ENDPOINT
airflow variables get ASTRA_DB_TOKEN

# Test connection manually
python -c "
from astrapy import DataAPIClient
client = DataAPIClient()
db = client.get_database('YOUR_ENDPOINT', token='YOUR_TOKEN')
print('✅ Connected')
"
```

#### **Issue 4: Task Stuck in "Running"**

```bash
# Check task logs
airflow tasks logs daily_stock_pipeline enrich_with_sentiment 2025-11-26 1

# Kill zombie tasks
airflow tasks clear daily_stock_pipeline -s 2025-11-26 -e 2025-11-26 -t enrich_with_sentiment

# Restart scheduler
pkill -f "airflow scheduler"
airflow scheduler &
```

#### **Issue 5: Retries Exhausted**

**Check Logs**:
```bash
airflow tasks logs daily_stock_pipeline load_into_delta 2025-11-26 1
```

**Common Causes**:
- YFinance API rate limit (wait 1 hour)
- AstraDB timeout (check network)
- Delta table corruption (rebuild table)

**Fix**:
```bash
# Clear and retry
airflow tasks clear daily_stock_pipeline -s 2025-11-26 -e 2025-11-26

# Or trigger new run
airflow dags trigger daily_stock_pipeline
```

---

## Commands Reference

### **Setup Commands**

```bash
# Initialize Airflow
export AIRFLOW_HOME=~/airflow
airflow db init

# Create user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Set variables
airflow variables set KEY VALUE
airflow variables import /tmp/vars.json
```

### **Service Commands**

```bash
# Start web server
airflow webserver --port 8080 &

# Start scheduler
airflow scheduler &

# Stop all Airflow processes
pkill -f "airflow"
```

### **DAG Management**

```bash
# List DAGs
airflow dags list

# Trigger DAG
airflow dags trigger daily_stock_pipeline

# Pause/Unpause
airflow dags pause daily_stock_pipeline
airflow dags unpause daily_stock_pipeline

# Delete DAG runs
airflow dags delete daily_stock_pipeline
```

### **Task Management**

```bash
# Test task (doesn't save state)
airflow tasks test daily_stock_pipeline extract_yfinance 2025-11-26

# Run task (saves state)
airflow tasks run daily_stock_pipeline extract_yfinance 2025-11-26

# Clear task (mark as "not run")
airflow tasks clear daily_stock_pipeline -s 2025-11-26 -e 2025-11-26

# View logs
airflow tasks logs daily_stock_pipeline extract_yfinance 2025-11-26 1
```

### **Debugging**

```bash
# Check DAG for errors
python ~/airflow/dags/stock_pipeline_dag.py

# List task instances
airflow tasks list daily_stock_pipeline -t

# Show task state
airflow tasks state daily_stock_pipeline extract_yfinance 2025-11-26

# Re-run failed tasks
airflow dags backfill daily_stock_pipeline -s 2025-11-25 -e 2025-11-26 --reset-dagruns
```

---

## Production Deployment

### **Systemd Services** (Auto-start on Boot)

**Create**: `/etc/systemd/system/airflow-webserver.service`
```ini
[Unit]
Description=Airflow webserver
After=network.target

[Service]
Type=simple
User=mha2cob
Environment="AIRFLOW_HOME=/home/mha2cob/airflow"
Environment="PATH=/home/mha2cob/miniconda3/envs/breaking_data/bin:/usr/bin"
ExecStart=/home/mha2cob/miniconda3/envs/breaking_data/bin/airflow webserver --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

**Create**: `/etc/systemd/system/airflow-scheduler.service`
```ini
[Unit]
Description=Airflow scheduler
After=network.target

[Service]
Type=simple
User=mha2cob
Environment="AIRFLOW_HOME=/home/mha2cob/airflow"
Environment="PATH=/home/mha2cob/miniconda3/envs/breaking_data/bin:/usr/bin"
ExecStart=/home/mha2cob/miniconda3/envs/breaking_data/bin/airflow scheduler
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable Services**:
```bash
sudo systemctl enable airflow-webserver
sudo systemctl enable airflow-scheduler
sudo systemctl start airflow-webserver
sudo systemctl start airflow-scheduler

# Check status
sudo systemctl status airflow-webserver
sudo systemctl status airflow-scheduler
```

---

## Summary

### **Key Points**

✅ **4-Stage Pipeline**: Extract → Enrich → Load → Inference  
✅ **Daily Schedule**: 23:00 UTC (11 PM)  
✅ **29 Stocks**: Processed in 5-10 minutes  
✅ **Auto-Retry**: 2 retries with 5-minute delay  
✅ **Delta Lake**: ACID-compliant storage  
✅ **MLflow**: Integrated monitoring  

### **Critical Paths**

- **DAG File**: `~/airflow/dags/stock_pipeline_dag.py`
- **Web UI**: `http://localhost:8080`
- **Logs**: `~/airflow/logs/`
- **Database**: `~/airflow/airflow.db` (SQLite)

### **Quick Start**

```bash
# 1. Start services
airflow webserver --port 8080 &
airflow scheduler &

# 2. Unpause DAG
airflow dags unpause daily_stock_pipeline

# 3. Trigger manually (optional)
airflow dags trigger daily_stock_pipeline

# 4. Monitor
http://localhost:8080
```

---

**Last Updated**: November 26, 2025  
**Version**: 1.0.0  
**Airflow Version**: 2.7.3  
**Status**: Production
