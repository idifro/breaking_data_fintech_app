# Airflow Setup Guide for Stock Pipeline

## Prerequisites
- Python environment with all dependencies installed
- Stock delta tables already populated with initial data
- AstraDB credentials configured in `config/config.yaml`

## Step 1: Install Airflow

```bash
# Install Airflow (version 2.7+)
pip install apache-airflow==2.7.3

# Or use the constraints file for your Python version
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==2.7.3" --constraint "${CONSTRAINT_URL}"
```

## Step 2: Initialize Airflow

```bash
# Set Airflow home directory (optional - defaults to ~/airflow)
export AIRFLOW_HOME=/home/mha2cob/Downloads/breaking_data/spark_ml_pipeline/airflow

# Initialize the database
airflow db init

# Create an admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

## Step 3: Configure Airflow DAGs Folder

```bash
# Create DAGs directory in Airflow home
mkdir -p $AIRFLOW_HOME/dags

# Copy or symlink your DAG file
cp scripts/stock_pipeline_dag.py $AIRFLOW_HOME/dags/

# Or create a symlink
ln -s /home/mha2cob/Downloads/breaking_data/spark_ml_pipeline/scripts/stock_pipeline_dag.py $AIRFLOW_HOME/dags/
```

## Step 4: Update airflow.cfg (Optional)

Edit `$AIRFLOW_HOME/airflow.cfg` to configure:

```ini
[core]
# Point to your project directory
dags_folder = /home/mha2cob/Downloads/breaking_data/spark_ml_pipeline/airflow/dags

# Parallelism settings
parallelism = 32
dag_concurrency = 16
max_active_runs_per_dag = 1

[scheduler]
# Check for new DAGs every 30 seconds
dag_dir_list_interval = 30
```

## Step 5: Set Environment Variables (if not using config.yaml)

```bash
# AstraDB credentials (if not in config.yaml)
export ASTRA_DB_ENDPOINT="https://d1091311-1ba8-4970-bd74-937d6cda55b0-us-east-2.apps.astra.datastax.com"
export ASTRA_DB_TOKEN="AstraCS:oIGnfXQXCRMJqDCqkYQbyCXM:..."

# Add project root to PYTHONPATH so imports work
export PYTHONPATH=/home/mha2cob/Downloads/breaking_data/spark_ml_pipeline:$PYTHONPATH
```

## Step 6: Start Airflow Services

### Option A: Run Standalone (Development)
```bash
# Run Airflow standalone (combines webserver + scheduler)
airflow standalone
```

### Option B: Run Webserver and Scheduler Separately (Production)
```bash
# Terminal 1: Start the web server (default port 8080)
airflow webserver --port 8080

# Terminal 2: Start the scheduler
airflow scheduler
```

## Step 7: Access Airflow UI

1. Open browser: http://localhost:8080
2. Login with credentials (username: `admin`, password: `admin`)
3. Find your DAG: `stock_pipeline_daily`
4. Toggle the DAG to "ON" (unpause it)

## Step 8: Trigger the DAG

### Via UI:
- Click on the DAG name
- Click the "Play" button (▶) in top right
- Click "Trigger DAG"

### Via CLI:
```bash
# Trigger manually
airflow dags trigger stock_pipeline_daily

# Test a specific task
airflow tasks test stock_pipeline_daily extract_task 2025-11-26

# Run backfill for a date range
airflow dags backfill stock_pipeline_daily \
    --start-date 2025-11-20 \
    --end-date 2025-11-26
```

## Step 9: Monitor the Pipeline

### Check DAG Status:
```bash
# List all DAGs
airflow dags list

# Get DAG state
airflow dags state stock_pipeline_daily

# List task instances
airflow tasks list stock_pipeline_daily
```

### View Logs:
- UI: Click on task → View Log
- CLI: `airflow tasks logs stock_pipeline_daily extract_task 2025-11-26`

## Pipeline Workflow

The DAG runs daily at 23:00 UTC with 4 tasks:

```
extract_task → enrich_task → load_task → inference_task
```

1. **extract_task**: Reads delta tables, calculates next business day, fetches YFinance data
2. **enrich_task**: Adds sentiment columns from AstraDB
3. **load_task**: Appends 1 row per stock to delta tables
4. **inference_task**: Runs ML inference using MLflow model

## Troubleshooting

### DAG not appearing in UI:
```bash
# Check for errors in DAG file
python scripts/stock_pipeline_dag.py

# List DAGs and see errors
airflow dags list-import-errors
```

### Import errors:
```bash
# Make sure PYTHONPATH includes project root
export PYTHONPATH=/home/mha2cob/Downloads/breaking_data/spark_ml_pipeline:$PYTHONPATH

# Test imports
python -c "from src.extract_yfinance import run_extract_yfinance"
```

### Task failures:
- Check task logs in UI or via CLI
- Verify delta tables exist in `delta_tables/stock_*/`
- Verify config.yaml has correct AstraDB credentials
- Check that MLflow model exists: `stock_predictor_spark_rf_unified_29stocks` version 1

### Schedule not running:
- Ensure DAG is unpaused (toggle ON in UI)
- Check scheduler is running: `ps aux | grep "airflow scheduler"`
- Verify schedule_interval in DAG: `"0 23 * * *"` (23:00 UTC daily)

## Production Deployment

For production, consider:

1. **Use a production database** (PostgreSQL/MySQL instead of SQLite)
2. **Configure executor** (CeleryExecutor or KubernetesExecutor for scaling)
3. **Set up monitoring** (integrate with Prometheus/Grafana)
4. **Configure alerts** (email/Slack notifications on failures)
5. **Use Airflow Variables/Connections** instead of environment variables
6. **Enable authentication** (LDAP, OAuth, etc.)

### Example: Store credentials in Airflow Variables
```bash
# Set AstraDB credentials as Airflow Variables
airflow variables set ASTRA_DB_ENDPOINT "https://d1091311-1ba8-4970-bd74-937d6cda55b0-us-east-2.apps.astra.datastax.com"
airflow variables set ASTRA_DB_TOKEN "AstraCS:..."
```

Then update `compute_sentiment_columns.py` to use Airflow Variables as fallback (already implemented).

## Quick Start Commands

```bash
# One-time setup
export AIRFLOW_HOME=/home/mha2cob/Downloads/breaking_data/spark_ml_pipeline/airflow
export PYTHONPATH=/home/mha2cob/Downloads/breaking_data/spark_ml_pipeline:$PYTHONPATH
airflow db init
mkdir -p $AIRFLOW_HOME/dags
cp scripts/stock_pipeline_dag.py $AIRFLOW_HOME/dags/

# Start Airflow
airflow standalone

# In another terminal - trigger DAG
airflow dags trigger stock_pipeline_daily
```
