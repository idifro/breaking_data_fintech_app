import os
import pandas as pd
import yaml
from astrapy import DataAPIClient
try:
    from airflow.models import Variable
except Exception:
    Variable = None
from typing import Optional
from pathlib import Path


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_astra_rows(db_endpoint: Optional[str] = None, db_token: Optional[str] = None, stocks: Optional[list] = None):
    """
    Loads the AstraDB sentiment rows.
    Credentials pulled from config.yaml first, then Airflow Variables, then Environment if not provided.
    """
    # Try to load from config.yaml first
    if db_endpoint is None or db_token is None:
        try:
            config = load_config()
            astra_config = config.get('astradb', {})
            db_endpoint = db_endpoint or astra_config.get('endpoint')
            db_token = db_token or astra_config.get('token')
            print(f"[ASTRA] Loaded credentials from config.yaml")
        except Exception as e:
            print(f"[ASTRA] Could not load config.yaml: {e}")
    
    # Fallback to Airflow Variables or Environment variables
    if Variable is not None:
        db_endpoint = db_endpoint or Variable.get("ASTRA_DB_ENDPOINT", default_var=os.getenv("ASTRA_DB_ENDPOINT"))
        db_token = db_token or Variable.get("ASTRA_DB_TOKEN", default_var=os.getenv("ASTRA_DB_TOKEN"))
    else:
        db_endpoint = db_endpoint or os.getenv("ASTRA_DB_ENDPOINT")
        db_token = db_token or os.getenv("ASTRA_DB_TOKEN")

    if not db_endpoint or not db_token:
        raise ValueError("Missing AstraDB credentials. Set Airflow Variables or ENV vars.")

    client = DataAPIClient()
    try:
        db = client.get_database(db_endpoint, token=db_token)
        # Use news_sentiment_1 table (confirmed via CQL)
        table = db.get_table("news_sentiment_1")

        rows = []
        if stocks:
            # query per-stock to avoid full-table scan; try a few likely key formats
            for s in stocks:
                s_up = str(s).upper()
                s_prefixed = f"stock_{s_up}"
                try:
                    rows.extend(list(table.find({"stock_symbol": s_up})))
                except Exception:
                    pass
                try:
                    rows.extend(list(table.find({"stock_symbol": s_prefixed})))
                except Exception:
                    pass
        else:
            rows = list(table.find({}))
    except Exception:
        print("[ASTRA] Error querying AstraDB, returning empty dataframe.")
        rows = []

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ASTRA] Warning: zero sentiment rows found.")
    return df


def compute_daily_sentiments(yf_df: pd.DataFrame, astra_df: pd.DataFrame):
    """
    Add Sentiment_gpt, News_flag, Scaled_sentiment.
    Filters Astra rows to only stocks/dates present in yf_df, uses YYYY-MM-DD from event_time.
    """
    if yf_df.empty:
        return yf_df.assign(Sentiment_gpt=0.0, News_flag=0, Scaled_sentiment=0.0)

    # Prepare yf keys
    yf = yf_df.copy()
    # support multiple possible date column names/index
    if "Date" in yf.columns:
        yf["date"] = pd.to_datetime(yf["Date"]).dt.strftime("%Y-%m-%d")
    elif "date" in yf.columns:
        yf["date"] = pd.to_datetime(yf["date"]).dt.strftime("%Y-%m-%d")
    else:
        # try index
        try:
            yf.index = pd.to_datetime(yf.index)
            yf["date"] = yf.index.strftime("%Y-%m-%d")
        except Exception:
            yf["date"] = pd.NaT

    # normalize symbol to uppercase without prefix
    yf["stock_symbol"] = yf["symbol"].astype(str).str.upper()

    # Early return if no astra rows
    if astra_df.empty:
        yf["Sentiment_gpt"] = 0.0
        yf["News_flag"] = 0
        yf["Scaled_sentiment"] = 0.0
        return yf

    # Normalize Astra and extract YYYY-MM-DD
    astra = astra_df.copy()
    # Extract date from published_at column (was event_time in requirements, actual column is published_at)
    if 'published_at' in astra.columns:
        astra["event_date"] = astra["published_at"].astype(str).str[:10]
    elif 'event_time' in astra.columns:
        astra["event_date"] = astra["event_time"].astype(str).str[:10]
    else:
        print("[ASTRA] Warning: No date column found in AstraDB data")
        yf["Sentiment_gpt"] = 0.0
        yf["News_flag"] = 0
        yf["Scaled_sentiment"] = 0.0
        return yf
    
    # normalize Astra stock_symbol to uppercase and remove optional 'stock_' prefix
    astra["stock_symbol"] = (
        astra["stock_symbol"].astype(str)
        .str.upper()
        .str.replace(r"^STOCK_", "", regex=True)
    )
    astra["sentiment_score"] = pd.to_numeric(astra["sentiment_score"], errors="coerce").fillna(0.0)

    # Filter to only relevant (stock_symbol, date) present in YF
    relevant_keys = set(zip(yf["stock_symbol"], yf["date"]))
    astra["key"] = list(zip(astra["stock_symbol"], astra["event_date"]))
    astra = astra[astra["key"].isin(relevant_keys)]

    if astra.empty:
        yf["Sentiment_gpt"] = 0.0
        yf["News_flag"] = 0
        yf["Scaled_sentiment"] = 0.0
        return yf

    grouped = (
        astra.groupby(["stock_symbol", "event_date"])["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "Sentiment_gpt", "event_date": "date"})
    )

    merged = yf.merge(grouped, how="left", left_on=["stock_symbol", "date"], right_on=["stock_symbol", "date"])
    merged["Sentiment_gpt"] = merged["Sentiment_gpt"].fillna(0.0)
    merged["News_flag"] = (merged["Sentiment_gpt"] != 0).astype(int)
    
    # Apply scaling formula: (Sentiment_gpt - 0.9999) / 4
    # This scales sentiment from 1-5 range to approximately 0-1 range
    # If Sentiment_gpt is 0 (no news), Scaled_sentiment remains 0
    merged["Scaled_sentiment"] = merged["Sentiment_gpt"].apply(
        lambda x: (x - 0.9999) / 4 if x != 0 else 0.0
    )
    
    # Clean up: Keep only expected columns and rename Adj Close to Adj close
    # Keep symbol column for CSV output
    final_columns = ['symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj_close', 'Volume', 
                     'Sentiment_gpt', 'News_flag', 'Scaled_sentiment']
    merged = merged[[col for col in final_columns if col in merged.columns]]
    merged = merged.rename(columns={'symbol': 'stock_symbol'})
    
    # Format Date column as YYYY-MM-DD HH:MM:SS+00:00
    merged['Date'] = pd.to_datetime(merged['Date']).dt.tz_localize('UTC').dt.strftime('%Y-%m-%d %H:%M:%S%z')
    # Add colon in timezone offset: +0000 -> +00:00
    merged['Date'] = merged['Date'].str[:-2] + ':' + merged['Date'].str[-2:]
    
    return merged


def run_sentiment_enrichment(yfinance_path: str, output_path: str, db_endpoint: Optional[str] = None, db_token: Optional[str] = None):
    """
    Wrapper used by Airflow DAG / CLI to read the YFinance parquet, enrich with AstraDB
    sentiment averages and write the enriched parquet file.
    Credentials may be provided or will be taken from Airflow Variables / ENV via load_astra_rows.
    """
    yf_df = pd.read_parquet(yfinance_path)
    # pass the unique stock list to Astra query to limit rows
    try:
        stocks = sorted(set(yf_df["symbol"].astype(str).str.upper().tolist()))
    except Exception:
        stocks = None
    astra_df = load_astra_rows(db_endpoint=db_endpoint, db_token=db_token, stocks=stocks)
    enriched = compute_daily_sentiments(yf_df, astra_df)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(str(out_path), index=False)
    print(f"[ENRICH] Wrote enriched data → {output_path}")

#run_sentiment_enrichment("/home/mha2cob/Downloads/breaking_data/spark_ml_pipeline/tmp/test_yfinance_raw.parquet", "/home/mha2cob/Downloads/breaking_data/spark_ml_pipeline/tmp/test_yfinance_enriched.parquet")