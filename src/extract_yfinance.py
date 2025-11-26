import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max
from datetime import datetime, timedelta


def find_next_business_date(date_val):
    """
    Get next business day (skip weekends)
    
    Args:
        date_val: datetime.date or datetime object
        
    Returns:
        datetime.date: Next business day
    """
    if date_val is None:
        return None
    
    # Convert to date if datetime
    if isinstance(date_val, datetime):
        date_val = date_val.date()
    
    next_day = date_val + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    
    return next_day

def extract_yfinance(stocks: list, start_date: str, end_date: str):
    # ...existing code...
    df_list = []
    for symbol in stocks:
        try:
            # Download with auto_adjust=False to get both Close and Adj Close
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        except Exception:
            print(f"[YFINANCE] Error downloading {symbol}, skipping.")
            continue

        if data is None or data.empty:
            print(f"[YFINANCE] No data for {symbol} between {start_date} and {end_date}")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            # Get the first level (price type) as column names
            data.columns = ['_'.join(filter(None, map(str, col))).strip() for col in data.columns.values]
        
        data = data.reset_index()
        
        # Rename columns to standard format
        col_map = {}
        for col in data.columns:
            lower_col = col.lower()
            if 'open' in lower_col:
                col_map[col] = 'Open'
            elif 'high' in lower_col:
                col_map[col] = 'High'
            elif 'low' in lower_col:
                col_map[col] = 'Low'
            elif 'adj' in lower_col and 'close' in lower_col:
                col_map[col] = 'Adj_close'
            elif 'close' in lower_col and 'adj' not in lower_col:
                col_map[col] = 'Close'
            elif 'volume' in lower_col:
                col_map[col] = 'Volume'
            elif 'date' in lower_col or col == 'Date':
                col_map[col] = 'Date'
        
        data = data.rename(columns=col_map)
        data["symbol"] = symbol
        
        # Keep only necessary columns (now including Adj Close)
        keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_close', 'Volume', 'symbol']
        data = data[[col for col in keep_cols if col in data.columns]]
        
        df_list.append(data)

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def run_extract_yfinance(output_path: str, stocks: list, run_date: str = None, delta_base_path: str = "delta_tables"):
    """
    Runs daily fetch for next business day after last date in delta tables.
    Reads delta tables to get last date, calculates next business day, and extracts data for that date.
    
    Args:
        output_path: Path to save parquet output
        stocks: List of stock symbols
        run_date: Optional override date (for testing)
        delta_base_path: Base path for delta tables (default: delta_tables)
    """
    print(f"[YFINANCE] Starting extraction for {len(stocks)} stocks")
    
    # Start Spark session
    print("[YFINANCE] Initializing Spark session...")
    builder = (
        SparkSession.builder
        .appName("stock_yfinance_extract")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Dictionary to store next business dates for each stock
        stock_dates = {}
        
        # Read each stock's delta table and get last date
        for stock in stocks:
            delta_path = f"{delta_base_path}/stock_{stock}"
            try:
                # Read delta table
                df = spark.read.format("delta").load(delta_path)
                
                # Get last date
                last_date_row = df.agg(spark_max(col("Date")).alias("max_date")).collect()[0]
                last_date = last_date_row["max_date"]
                
                if last_date:
                    # Convert to date if timestamp
                    if isinstance(last_date, datetime):
                        last_date = last_date.date()
                    
                    # Find next business day
                    next_biz_date = find_next_business_date(last_date)
                    stock_dates[stock] = {
                        'last_date': str(last_date),
                        'next_date': str(next_biz_date)
                    }
                    print(f"[YFINANCE] {stock}: Last date = {last_date}, Next business day = {next_biz_date}")
                else:
                    print(f"[YFINANCE] Warning: No date found for {stock}, skipping")
                    
            except Exception as e:
                print(f"[YFINANCE] Error reading delta table for {stock}: {e}")
                continue
        
        if not stock_dates:
            print("[YFINANCE] ERROR: No valid dates found for any stock!")
            spark.stop()
            return
        
        # Use the most common next date (they should all be the same)
        next_dates = [info['next_date'] for info in stock_dates.values()]
        extraction_date = max(set(next_dates), key=next_dates.count)
        
        print(f"\n[YFINANCE] ✅ Extraction Date: {extraction_date}")
        print(f"[YFINANCE] Fetching data for {len(stock_dates)} stocks...\n")
        
        # Extract data for the next business day
        start = extraction_date
        end = str(datetime.strptime(extraction_date, "%Y-%m-%d").date() + timedelta(days=1))
        
        df = extract_yfinance(list(stock_dates.keys()), start, end)
        
    finally:
        # Close Spark session
        print("[YFINANCE] Closing Spark session...")
        spark.stop()
    
    # Save output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[YFINANCE] ⚠️  No rows fetched for {extraction_date}. YFinance may not have data yet.")
        print(f"[YFINANCE] Writing empty parquet with NaN values...")
        # Create empty dataframe with expected schema
        empty_data = {
            'Date': [pd.NaT] * len(stocks),
            'Open': [float('nan')] * len(stocks),
            'High': [float('nan')] * len(stocks),
            'Low': [float('nan')] * len(stocks),
            'Close': [float('nan')] * len(stocks),
            'Adj_close': [float('nan')] * len(stocks),
            'Volume': [float('nan')] * len(stocks),
            'symbol': list(stock_dates.keys())
        }
        df = pd.DataFrame(empty_data)
        df.to_parquet(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)
        print(f"[YFINANCE] ✅ Saved {len(df)} rows → {output_path}")

    print(f"[YFINANCE] Extraction complete!")
