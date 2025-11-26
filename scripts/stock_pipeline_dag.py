# dags/stock_pipeline_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path for imports
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from src.extract_yfinance import run_extract_yfinance
from src.compute_sentiment_columns import run_sentiment_enrichment
from src.load_delta import load_to_delta
from scripts.inference_batch import run_inference_batch


STOCK_LIST = [
    "AAL", "AAPL", "ABBV", "AMD", "AMGN", "BABA", "BIIB",
    "CMCSA", "CMG", "COP", "COST", "CRM", "CVX", "EBAY",
    "GE", "GOOG", "GSK", "MRK", "NKE", "NVDA", "ORCL",
    "PEP", "PYPL", "QCOM", "QQQ", "TSLA", "TSM", "USO", "WFC"
]

# Dynamic, safe temp paths per-day
YF_RAW_PATH = "tmp/yfinance_raw_{{ ds }}.parquet"
YF_ENRICHED_PATH = "tmp/yfinance_enriched_{{ ds }}.parquet"
DELTA_BASE_PATH = "delta_tables"


default_args = {
    "owner": "data_eng",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="daily_stock_pipeline",
    default_args=default_args,
    schedule_interval="0 23 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# -------------------------------------------------------------------

extract_task = PythonOperator(
    task_id="extract_yfinance",
    python_callable=run_extract_yfinance,
    op_kwargs={
        "output_path": YF_RAW_PATH,
        "stocks": STOCK_LIST,
    },
    dag=dag,
)

enrich_task = PythonOperator(
    task_id="enrich_with_sentiment",
    python_callable=run_sentiment_enrichment,
    op_kwargs={
        "yfinance_path": YF_RAW_PATH,
        "output_path": YF_ENRICHED_PATH,
    },
    dag=dag,
)

load_task = PythonOperator(
    task_id="load_into_delta",
    python_callable=load_to_delta,
    op_kwargs={
        "input_path": YF_ENRICHED_PATH,
        "delta_base_path": DELTA_BASE_PATH,
    },
    dag=dag,
)

inference_task = PythonOperator(
    task_id="inference_batch",
    python_callable=run_inference_batch,
    dag=dag,
)

extract_task >> enrich_task >> load_task >> inference_task
