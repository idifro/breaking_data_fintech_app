# AstraDB Configuration & Integration

> **Cassandra NoSQL Database for News Sentiment Storage**  
> **Cloud-Native, Scalable, Multi-Region Support**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup Guide](#setup-guide)
- [Database Schema](#database-schema)
- [Configuration](#configuration)
- [Data Access](#data-access)
- [Integration](#integration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

**AstraDB** (powered by Apache Cassandra) stores **financial news sentiment data** for 29 stocks. It serves as the **sentiment data layer** for the ML pipeline, providing:

- ✅ **Scalable NoSQL Storage**: Handle millions of sentiment records
- ✅ **Low-Latency Queries**: < 100ms response time for stock queries
- ✅ **Multi-Region Support**: Global availability
- ✅ **Cloud-Native**: Fully managed by DataStax
- ✅ **Python Integration**: astrapy SDK for seamless access

### **Use Cases**

1. **Sentiment Storage**: Store GPT-scored news sentiment (1-5 scale)
2. **Daily Enrichment**: Query sentiment for 29 stocks in Airflow pipeline
3. **Historical Analysis**: Access full sentiment history for backtesting
4. **Real-Time Streaming**: (Future) Ingest from Kafka streaming pipeline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ASTRADB ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   NewsAPI → GPT-3.5 Sentiment → AstraDB                             │
│                                                                       │
│   1. Fetch financial news (NewsAPI)                                  │
│   2. Score sentiment with GPT-3.5 (1-5 scale)                        │
│   3. Write to AstraDB (news_sentiment_1 table)                       │
│                                                                       │
│   Script: FNSPID_Financial_News_Dataset/data_processor/              │
│           score_by_gpt.py                                            │
│                                                                       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ASTRADB (Cassandra)                             │
│                    Cloud-Hosted NoSQL Database                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Endpoint: https://YOUR-DB-ID-REGION.apps.astra.datastax.com         │
│   Region: us-east-1 (or configured region)                             │
│   Keyspace: default_keyspace                                           │
│                                                                         │
│   ┌───────────────────────────────────────────────────────┐           │
│   │  Table: news_sentiment_1                              │           │
│   │                                                        │           │
│   │  Partition Key: stock_symbol (TEXT)                   │           │
│   │  Clustering Key: published_at (TIMESTAMP) DESC        │           │
│   │                                                        │           │
│   │  Columns:                                             │           │
│   │    • stock_symbol (TEXT)                              │           │
│   │    • published_at (TIMESTAMP)                         │           │
│   │    • title (TEXT)                                     │           │
│   │    • description (TEXT)                               │           │
│   │    • url (TEXT)                                       │           │
│   │    • source (TEXT)                                    │           │
│   │    • sentiment_score (DOUBLE)                         │           │
│   │    • scaled_sentiment (DOUBLE)                        │           │
│   │    • created_at (TIMESTAMP)                           │           │
│   │                                                        │           │
│   │  Indexes:                                             │           │
│   │    • Primary: (stock_symbol, published_at)            │           │
│   │                                                        │           │
│   │  Typical Rows: ~500,000 (growing)                     │           │
│   │  Storage: ~50-100 MB                                  │           │
│   └───────────────────────────────────────────────────────┘           │
│                                                                         │
│   Data Distribution:                                                   │
│   • Partitioned by stock_symbol (29 partitions)                        │
│   • Clustered by published_at (time-series ordering)                   │
│   • Each stock partition: ~17,000 rows (500k / 29)                     │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA CONSUMPTION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Daily Airflow Pipeline (23:00 UTC)                                   │
│                                                                         │
│   1. Query AstraDB for 29 stocks (per-stock partition queries)         │
│   2. Filter to relevant dates (event_date = YYYY-MM-DD)                │
│   3. Group by (stock_symbol, date) and calculate mean sentiment        │
│   4. Join with YFinance price data                                     │
│   5. Add features: Sentiment_gpt, News_flag, Scaled_sentiment          │
│   6. Save to Delta Lake                                                 │
│                                                                         │
│   Script: src/compute_sentiment_columns.py                             │
│                                                                         │
│   Query Pattern (Efficient):                                           │
│   SELECT * FROM news_sentiment_1                                       │
│   WHERE stock_symbol = 'AAPL'                                          │
│   AND published_at >= '2025-11-01'                                     │
│                                                                         │
│   Performance:                                                          │
│   • Per-stock query: ~50-100ms                                          │
│   • 29 stocks sequential: ~3-5 seconds                                 │
│   • Result: ~500-1000 rows (filtered)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW DIAGRAM                                │
└─────────────────────────────────────────────────────────────────────────┘

NewsAPI
   │
   ├─ Fetch news for AAPL, GOOG, TSLA, etc.
   │
   ▼
GPT-3.5 Turbo
   │
   ├─ Score sentiment: 1 (negative) to 5 (positive)
   │
   ▼
AstraDB Write
   │
   ├─ INSERT INTO news_sentiment_1 (stock_symbol, published_at, ...)
   │
   ▼
AstraDB Storage (Cassandra)
   │
   ├─ Partition by stock_symbol
   ├─ Cluster by published_at DESC
   │
   ▼
Airflow DAG (Daily)
   │
   ├─ Query per stock: WHERE stock_symbol = ?
   ├─ Filter by date: AND published_at >= ?
   │
   ▼
Pandas DataFrame
   │
   ├─ Extract event_date (YYYY-MM-DD)
   ├─ Group by (stock, date)
   ├─ Calculate mean(sentiment_score)
   │
   ▼
YFinance Join
   │
   ├─ Left join on (stock_symbol, date)
   ├─ Add Sentiment_gpt (1-5 or 0)
   ├─ Add News_flag (1 if news exists, 0 otherwise)
   ├─ Add Scaled_sentiment: (Sentiment_gpt - 0.9999) / 4
   │
   ▼
Delta Lake
   │
   ├─ Append to delta_tables/stock_AAPL/
   ├─ 11 columns including sentiment features
   │
   ▼
ML Training/Inference
   │
   └─ Use Scaled_sentiment as feature for stock price prediction
```

---

## Setup Guide

### **1. Create AstraDB Account**

1. Visit: https://astra.datastax.com
2. Sign up (free tier: 25GB storage, 5M read/1M write per month)
3. Create database:
   - **Database Name**: `stock_sentiment_db`
   - **Keyspace**: `default_keyspace`
   - **Cloud Provider**: AWS / GCP / Azure
   - **Region**: `us-east-1` (or nearest)

### **2. Get Credentials**

**Database Endpoint**:
```
https://YOUR-DB-ID-us-east-1.apps.astra.datastax.com
```

**Generate Application Token**:
1. Go to: Database → Settings → Application Tokens
2. Select Role: **Database Administrator**
3. Generate Token → Save token (starts with `AstraCS:...`)

**Example**:
```
Token: AstraCS:AbCdEfGhIjKlMnOpQrStUvWxYz1234567890...
```

⚠️ **Important**: Store token securely (never commit to git)

### **3. Install Python SDK**

```bash
# Activate conda environment
conda activate breaking_data

# Install astrapy
pip install astrapy
```

### **4. Verify Connection**

```python
from astrapy import DataAPIClient

# Configure
endpoint = "https://YOUR-DB-ID-us-east-1.apps.astra.datastax.com"
token = "AstraCS:..."

# Connect
client = DataAPIClient()
db = client.get_database(endpoint, token=token)

# Test
collections = list(db.list_collection_names())
print(f"✅ Connected to AstraDB")
print(f"Available collections: {collections}")
```

---

## Database Schema

### **Table: news_sentiment_1**

```sql
CREATE TABLE news_sentiment_1 (
    stock_symbol TEXT,
    published_at TIMESTAMP,
    title TEXT,
    description TEXT,
    url TEXT,
    source TEXT,
    sentiment_score DOUBLE,
    scaled_sentiment DOUBLE,
    created_at TIMESTAMP,
    PRIMARY KEY (stock_symbol, published_at)
) WITH CLUSTERING ORDER BY (published_at DESC);
```

### **Column Descriptions**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| **stock_symbol** | TEXT | Stock ticker (partition key) | `AAPL` |
| **published_at** | TIMESTAMP | News publish timestamp (clustering key) | `2025-11-26 10:30:00` |
| **title** | TEXT | News headline | `"Apple Reports Q4 Earnings Beat"` |
| **description** | TEXT | News summary/excerpt | `"Apple Inc. exceeded analyst expectations..."` |
| **url** | TEXT | Article URL | `https://www.bloomberg.com/news/...` |
| **source** | TEXT | News source | `"Bloomberg"` |
| **sentiment_score** | DOUBLE | GPT sentiment (1-5) | `4.5` |
| **scaled_sentiment** | DOUBLE | Normalized (0-1 range) | `0.8750625` |
| **created_at** | TIMESTAMP | Record insertion time | `2025-11-26 11:00:00` |

### **Primary Key Design**

```
PRIMARY KEY (stock_symbol, published_at)
```

**Rationale**:
- **Partition Key** (`stock_symbol`): Distributes data across nodes, enables efficient per-stock queries
- **Clustering Key** (`published_at`): Orders data within partition by time (DESC = newest first)

**Query Efficiency**:
✅ **Fast**: `WHERE stock_symbol = 'AAPL'` (partition key)  
✅ **Fast**: `WHERE stock_symbol = 'AAPL' AND published_at >= '2025-11-01'` (partition + range)  
❌ **Slow**: `WHERE published_at >= '2025-11-01'` (full table scan)  

### **Sample Data**

```
stock_symbol | published_at        | title                        | sentiment_score | scaled_sentiment
-------------|---------------------|------------------------------|-----------------|------------------
AAPL         | 2025-11-26 10:30:00 | Apple Reports Q4 Earnings... | 5.0             | 1.000025
AAPL         | 2025-11-26 08:15:00 | iPhone 16 Sales Exceed...   | 4.0             | 0.750025
AAPL         | 2025-11-25 14:20:00 | Apple Announces New...       | 3.5             | 0.625025
GOOG         | 2025-11-26 09:00:00 | Google AI Breakthrough...    | 4.5             | 0.8750625
TSLA         | 2025-11-26 11:45:00 | Tesla Stock Drops 3%...      | 2.0             | 0.250025
```

### **Data Volume Estimate**

```
Stocks: 29
Articles per stock per day: ~10-20
Daily inserts: 29 × 15 = ~435 rows/day
Monthly: 435 × 30 = ~13,000 rows
Yearly: 435 × 365 = ~159,000 rows

After 3 years: ~500,000 rows
Storage: ~50-100 MB (with text content)
```

---

## Configuration

### **Method 1: config.yaml** (Recommended)

**File**: `config/config.yaml`

```yaml
astradb:
  endpoint: "https://YOUR-DB-ID-us-east-1.apps.astra.datastax.com"
  token: "${ASTRA_DB_TOKEN}"  # Use environment variable for security
  table_name: "news_sentiment_1"
  keyspace: "default_keyspace"
  region: "us-east-1"
```

**Load in Python**:
```python
import yaml
import os

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Substitute environment variable
astra_config = config['astradb']
astra_config['token'] = os.getenv('ASTRA_DB_TOKEN', astra_config['token'])
```

### **Method 2: Airflow Variables**

```bash
# Set via CLI
airflow variables set ASTRA_DB_ENDPOINT "https://YOUR-DB-ID.apps.astra.datastax.com"
airflow variables set ASTRA_DB_TOKEN "AstraCS:..."

# Or via Web UI
Admin → Variables → Add Variable
```

**Load in DAG**:
```python
from airflow.models import Variable

endpoint = Variable.get("ASTRA_DB_ENDPOINT")
token = Variable.get("ASTRA_DB_TOKEN")
```

### **Method 3: Environment Variables**

```bash
# Add to ~/.bashrc or ~/.zshrc
export ASTRA_DB_ENDPOINT="https://YOUR-DB-ID.apps.astra.datastax.com"
export ASTRA_DB_TOKEN="AstraCS:..."

# Reload
source ~/.bashrc
```

**Load in Python**:
```python
import os

endpoint = os.getenv("ASTRA_DB_ENDPOINT")
token = os.getenv("ASTRA_DB_TOKEN")
```

### **Credential Fallback Chain** (Production Pattern)

**Implementation** (from `src/compute_sentiment_columns.py`):

```python
def load_config():
    """
    Load AstraDB credentials with fallback chain:
    1. config.yaml
    2. Airflow Variables
    3. Environment Variables
    """
    # Try config file first
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            if 'astradb' in config:
                return config['astradb']
    except FileNotFoundError:
        pass
    
    # Try Airflow Variables
    try:
        from airflow.models import Variable
        return {
            'endpoint': Variable.get("ASTRA_DB_ENDPOINT"),
            'token': Variable.get("ASTRA_DB_TOKEN"),
        }
    except:
        pass
    
    # Fallback to environment
    return {
        'endpoint': os.getenv("ASTRA_DB_ENDPOINT"),
        'token': os.getenv("ASTRA_DB_TOKEN"),
    }
```

---

## Data Access

### **Connection Example**

```python
from astrapy import DataAPIClient
import pandas as pd

# Load config
config = load_config()

# Connect
client = DataAPIClient()
db = client.get_database(config['endpoint'], token=config['token'])
table = db.get_table("news_sentiment_1")

# Query data
rows = list(table.find({"stock_symbol": "AAPL"}))
df = pd.DataFrame(rows)

print(f"✅ Loaded {len(df)} rows for AAPL")
```

### **Query Patterns**

#### **1. Get All Sentiment for a Stock**

```python
# Query by partition key (efficient)
aapl_data = list(table.find({"stock_symbol": "AAPL"}))

df = pd.DataFrame(aapl_data)
print(df.head())
```

**CQL Equivalent**:
```sql
SELECT * FROM news_sentiment_1 WHERE stock_symbol = 'AAPL';
```

#### **2. Get Sentiment for Date Range**

```python
from datetime import datetime

# Query with date range
start_date = datetime(2025, 11, 1)
end_date = datetime(2025, 11, 30)

aapl_nov = list(table.find({
    "stock_symbol": "AAPL",
    "published_at": {
        "$gte": start_date,
        "$lt": end_date
    }
}))

df = pd.DataFrame(aapl_nov)
print(f"AAPL sentiment for November: {len(df)} rows")
```

**CQL Equivalent**:
```sql
SELECT * FROM news_sentiment_1
WHERE stock_symbol = 'AAPL'
AND published_at >= '2025-11-01'
AND published_at < '2025-12-01';
```

#### **3. Get Sentiment for Multiple Stocks** (Pipeline Pattern)

```python
def load_astra_rows(stocks: List[str]):
    """
    Load sentiment for multiple stocks efficiently.
    
    Args:
        stocks: List of stock symbols (e.g., ['AAPL', 'GOOG', 'TSLA'])
    
    Returns:
        DataFrame with all sentiment data
    """
    client = DataAPIClient()
    db = client.get_database(endpoint, token=token)
    table = db.get_table("news_sentiment_1")
    
    all_rows = []
    for stock in stocks:
        # Query per partition (efficient)
        rows = list(table.find({"stock_symbol": stock.upper()}))
        all_rows.extend(rows)
    
    return pd.DataFrame(all_rows)

# Usage
stocks = ['AAPL', 'GOOG', 'TSLA', 'NVDA']
sentiment_df = load_astra_rows(stocks)

print(f"Loaded {len(sentiment_df)} rows for {len(stocks)} stocks")
```

#### **4. Calculate Daily Average Sentiment**

```python
# Load data
sentiment_df = load_astra_rows(['AAPL', 'GOOG'])

# Extract date from timestamp
sentiment_df['event_date'] = pd.to_datetime(sentiment_df['published_at']).dt.date

# Group by stock + date, calculate mean
daily_avg = sentiment_df.groupby(['stock_symbol', 'event_date'])['sentiment_score'].mean()

print(daily_avg)
```

**Output**:
```
stock_symbol  event_date
AAPL          2025-11-25    4.2
              2025-11-26    3.8
GOOG          2025-11-25    4.5
              2025-11-26    4.0
Name: sentiment_score, dtype: float64
```

---

## Integration

### **Pipeline Integration** (`src/compute_sentiment_columns.py`)

**Full Workflow**:

```python
from astrapy import DataAPIClient
import pandas as pd
import yaml

def load_config():
    """Load AstraDB credentials (fallback chain)"""
    # ... (see Configuration section)

def load_astra_rows(stocks=None):
    """
    Load sentiment data from AstraDB.
    
    Args:
        stocks: List of stock symbols (optional, filters data)
    
    Returns:
        DataFrame with columns: stock_symbol, published_at, title, 
                                sentiment_score, scaled_sentiment, etc.
    """
    config = load_config()
    
    # Connect
    client = DataAPIClient()
    db = client.get_database(config['endpoint'], token=config['token'])
    table = db.get_table("news_sentiment_1")
    
    # Query
    rows = []
    if stocks:
        # Per-stock queries (efficient)
        for stock in stocks:
            rows.extend(list(table.find({"stock_symbol": stock.upper()})))
    else:
        # Full table scan (avoid in production)
        rows = list(table.find({}))
    
    return pd.DataFrame(rows)

def compute_daily_sentiments(yf_df, astra_df):
    """
    Enrich YFinance data with daily average sentiment.
    
    Args:
        yf_df: DataFrame with YFinance price data (29 rows)
        astra_df: DataFrame with AstraDB sentiment data (500+ rows)
    
    Returns:
        DataFrame with 11 columns (price + sentiment features)
    """
    # Extract event date (YYYY-MM-DD)
    astra_df['event_date'] = pd.to_datetime(astra_df['published_at']).dt.strftime('%Y-%m-%d')
    yf_df['date'] = pd.to_datetime(yf_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Normalize stock symbols
    astra_df['stock_symbol'] = astra_df['stock_symbol'].str.upper()
    astra_df['stock_symbol'] = astra_df['stock_symbol'].str.replace('stock_', '', regex=False)
    
    # Filter to relevant (stock, date) pairs
    relevant_keys = set(zip(yf_df['stock_symbol'], yf_df['date']))
    astra_df = astra_df[
        astra_df[['stock_symbol', 'event_date']].apply(tuple, axis=1).isin(relevant_keys)
    ]
    
    # Group by (stock, date) and calculate mean sentiment
    grouped = astra_df.groupby(['stock_symbol', 'event_date'])['sentiment_score'] \
                      .mean() \
                      .reset_index() \
                      .rename(columns={'sentiment_score': 'Sentiment_gpt'})
    
    # Left join with YFinance (preserves rows with no news)
    merged = yf_df.merge(
        grouped,
        left_on=['stock_symbol', 'date'],
        right_on=['stock_symbol', 'event_date'],
        how='left'
    )
    
    # Add features
    merged['Sentiment_gpt'] = merged['Sentiment_gpt'].fillna(0.0)
    merged['News_flag'] = (merged['Sentiment_gpt'] != 0).astype(int)
    merged['Scaled_sentiment'] = merged['Sentiment_gpt'].apply(
        lambda x: (x - 0.9999) / 4 if x != 0 else 0.0
    )
    
    return merged

def run_sentiment_enrichment(yfinance_path: str, output_path: str):
    """
    Airflow task: Enrich YFinance data with sentiment.
    
    Args:
        yfinance_path: Input Parquet file (29 rows)
        output_path: Output Parquet file (29 rows with sentiment)
    """
    # Load YFinance data
    yf_df = pd.read_parquet(yfinance_path)
    
    # Load sentiment from AstraDB
    stocks = yf_df['stock_symbol'].unique().tolist()
    astra_df = load_astra_rows(stocks=stocks)
    
    # Compute daily sentiments
    enriched_df = compute_daily_sentiments(yf_df, astra_df)
    
    # Save
    enriched_df.to_parquet(output_path)
    print(f"✅ [ENRICH] Saved {len(enriched_df)} rows to {output_path}")
    
    return output_path
```

### **Airflow Integration**

**DAG Task** (`scripts/stock_pipeline_dag.py`):

```python
enrich_task = PythonOperator(
    task_id="enrich_with_sentiment",
    python_callable=run_sentiment_enrichment,
    op_kwargs={
        "yfinance_path": "tmp/yfinance_raw_{{ ds }}.parquet",
        "output_path": "tmp/yfinance_enriched_{{ ds }}.parquet",
    },
    dag=dag,
)
```

**Execution Flow**:
1. Load yesterday's YFinance data (29 rows)
2. Query AstraDB for sentiment (filter to 29 stocks)
3. Extract event dates, group by (stock, date)
4. Calculate mean sentiment per stock/date
5. Join with YFinance data
6. Add 3 features: Sentiment_gpt, News_flag, Scaled_sentiment
7. Save enriched data (11 columns)

---

## Performance

### **Query Performance**

| Query Type | Rows Returned | Latency | Notes |
|------------|---------------|---------|-------|
| Single stock (partition key) | ~17,000 | 50-100ms | ✅ Efficient |
| Single stock + date range | ~500 | 30-50ms | ✅ Very efficient |
| 29 stocks (sequential) | ~500,000 | 3-5 sec | ✅ Good (parallelizable) |
| Full table scan | ~500,000 | 10-20 sec | ❌ Avoid |

### **Optimization Tips**

**1. Always Filter by Partition Key**

```python
# ✅ Good: Uses partition key
table.find({"stock_symbol": "AAPL"})

# ❌ Bad: Full table scan
table.find({"published_at": {"$gte": "2025-11-01"}})
```

**2. Batch Per-Stock Queries**

```python
# ✅ Good: Parallel per-stock queries
from concurrent.futures import ThreadPoolExecutor

def query_stock(stock):
    return list(table.find({"stock_symbol": stock}))

with ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(query_stock, stocks)
    all_rows = [row for result in results for row in result]
```

**3. Use Date Range Filters**

```python
# ✅ Good: Limits clustering key range
table.find({
    "stock_symbol": "AAPL",
    "published_at": {"$gte": datetime(2025, 11, 1)}
})
```

### **Throughput**

| Operation | Throughput | Notes |
|-----------|------------|-------|
| **Reads** | 5M/month (free tier) | ~6 reads/second sustained |
| **Writes** | 1M/month (free tier) | ~0.4 writes/second sustained |
| **Concurrent Queries** | 10-20 | Use connection pooling |

**Free Tier Limits**:
- 25 GB storage
- 5M read operations/month
- 1M write operations/month

**Paid Tier** (if needed):
- $0.10 per 1M reads
- $0.50 per 1M writes
- $0.25/GB storage/month

---

## Troubleshooting

### **Issue 1: Connection Timeout**

**Error**:
```
astrapy.exceptions.DataAPIException: Connection timeout
```

**Solutions**:
```python
# 1. Check endpoint URL (include https://)
endpoint = "https://YOUR-DB-ID.apps.astra.datastax.com"  # ✅ Correct
endpoint = "YOUR-DB-ID.apps.astra.datastax.com"          # ❌ Wrong

# 2. Verify token (starts with AstraCS:)
token = "AstraCS:..."  # ✅ Correct
token = "eyJhbGc..."    # ❌ Wrong (JWT token, not application token)

# 3. Check network/firewall
curl https://YOUR-DB-ID.apps.astra.datastax.com
```

### **Issue 2: Authentication Failed**

**Error**:
```
astrapy.exceptions.DataAPIException: 401 Unauthorized
```

**Solutions**:
```python
# 1. Regenerate token
# Go to: AstraDB Console → Database → Settings → Application Tokens → Generate

# 2. Verify token has correct permissions
# Required Role: Database Administrator

# 3. Check token expiration
# Application tokens don't expire, but JWT tokens do
```

### **Issue 3: Table Not Found**

**Error**:
```
astrapy.exceptions.DataAPIException: Table 'news_sentiment_1' not found
```

**Solutions**:
```python
# 1. List existing tables
collections = list(db.list_collection_names())
print(f"Available tables: {collections}")

# 2. Create table if missing
from astrapy import DataAPIClient

client = DataAPIClient()
db = client.get_database(endpoint, token=token)

# Create table (CQL equivalent)
# Use AstraDB Console → CQL Console or Python SDK
```

### **Issue 4: Slow Queries**

**Symptom**: Queries take > 5 seconds

**Diagnosis**:
```python
import time

start = time.time()
rows = list(table.find({"stock_symbol": "AAPL"}))
end = time.time()

print(f"Query time: {end - start:.2f} seconds")
print(f"Rows returned: {len(rows)}")
```

**Solutions**:
```python
# 1. Add date range filter
table.find({
    "stock_symbol": "AAPL",
    "published_at": {"$gte": datetime(2025, 11, 1)}
})

# 2. Use projection (select specific columns)
table.find(
    {"stock_symbol": "AAPL"},
    projection={"stock_symbol": 1, "published_at": 1, "sentiment_score": 1}
)

# 3. Parallelize queries
# See Performance → Optimization Tips
```

### **Issue 5: Empty Results**

**Symptom**: Query returns 0 rows despite data existing

**Common Causes**:
```python
# 1. Case sensitivity
table.find({"stock_symbol": "aapl"})  # ❌ Wrong case
table.find({"stock_symbol": "AAPL"})  # ✅ Correct

# 2. Prefix mismatch
astra_df['stock_symbol'] = astra_df['stock_symbol'].str.replace('stock_', '')

# 3. Date format
# Ensure published_at is TIMESTAMP, not STRING
```

---

## Best Practices

### **1. Security**

```bash
# ✅ Use environment variables for credentials
export ASTRA_DB_TOKEN="AstraCS:..."

# ✅ Never commit credentials to git
echo "config/config.yaml" >> .gitignore

# ✅ Use Airflow Variables for DAGs
airflow variables set ASTRA_DB_TOKEN "AstraCS:..."

# ❌ Don't hardcode credentials
token = "AstraCS:..."  # Bad
```

### **2. Query Optimization**

```python
# ✅ Always filter by partition key
table.find({"stock_symbol": "AAPL"})

# ✅ Add date range for time-series data
table.find({
    "stock_symbol": "AAPL",
    "published_at": {"$gte": datetime(2025, 11, 1)}
})

# ✅ Use projection to reduce data transfer
table.find(
    {"stock_symbol": "AAPL"},
    projection={"sentiment_score": 1, "published_at": 1}
)

# ❌ Avoid full table scans
table.find({})  # Bad for large tables
```

### **3. Error Handling**

```python
from astrapy.exceptions import DataAPIException

def load_astra_rows_safe(stocks):
    """Load sentiment with error handling"""
    try:
        client = DataAPIClient()
        db = client.get_database(endpoint, token=token)
        table = db.get_table("news_sentiment_1")
        
        rows = []
        for stock in stocks:
            try:
                rows.extend(list(table.find({"stock_symbol": stock})))
            except DataAPIException as e:
                print(f"⚠️ Failed to query {stock}: {e}")
                continue
        
        return pd.DataFrame(rows)
    
    except DataAPIException as e:
        print(f"❌ AstraDB connection failed: {e}")
        return pd.DataFrame()  # Return empty DataFrame
```

### **4. Data Validation**

```python
def validate_sentiment_data(df):
    """Validate sentiment data quality"""
    issues = []
    
    # Check required columns
    required_cols = ['stock_symbol', 'published_at', 'sentiment_score']
    missing = set(required_cols) - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check sentiment range
    if (df['sentiment_score'] < 1).any() or (df['sentiment_score'] > 5).any():
        issues.append("Sentiment scores outside 1-5 range")
    
    # Check for nulls
    if df[required_cols].isnull().any().any():
        issues.append("Null values in required columns")
    
    if issues:
        print(f"⚠️ Data quality issues: {issues}")
    
    return len(issues) == 0
```

### **5. Monitoring**

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_astra_rows(stocks):
    """Load sentiment with monitoring"""
    start = time.time()
    
    logger.info(f"[ASTRA] Querying {len(stocks)} stocks")
    rows = []
    
    for i, stock in enumerate(stocks):
        stock_start = time.time()
        stock_rows = list(table.find({"stock_symbol": stock}))
        stock_time = time.time() - stock_start
        
        rows.extend(stock_rows)
        logger.info(f"[ASTRA] {i+1}/{len(stocks)} {stock}: "
                   f"{len(stock_rows)} rows in {stock_time:.2f}s")
    
    total_time = time.time() - start
    logger.info(f"[ASTRA] ✅ Loaded {len(rows)} rows in {total_time:.2f}s")
    
    return pd.DataFrame(rows)
```

---

## Summary

### **Key Points**

✅ **AstraDB**: Cloud-native Cassandra NoSQL database  
✅ **Table**: `news_sentiment_1` (stock_symbol, published_at, sentiment_score)  
✅ **Partition Key**: `stock_symbol` (enables efficient per-stock queries)  
✅ **Integration**: Airflow DAG Task 2 (Sentiment Enrichment)  
✅ **Performance**: < 100ms per-stock queries, 3-5 sec for 29 stocks  
✅ **Free Tier**: 25GB storage, 5M reads/month, 1M writes/month  

### **Quick Reference**

**Connection**:
```python
from astrapy import DataAPIClient

client = DataAPIClient()
db = client.get_database(endpoint, token=token)
table = db.get_table("news_sentiment_1")
```

**Query**:
```python
# Single stock
rows = list(table.find({"stock_symbol": "AAPL"}))

# Date range
rows = list(table.find({
    "stock_symbol": "AAPL",
    "published_at": {"$gte": datetime(2025, 11, 1)}
}))
```

**Pipeline**:
```python
# Load sentiment for 29 stocks
sentiment_df = load_astra_rows(stocks=['AAPL', 'GOOG', ...])

# Enrich YFinance data
enriched_df = compute_daily_sentiments(yf_df, sentiment_df)
```

---

**Last Updated**: November 26, 2025  
**Version**: 1.0.0  
**AstraDB SDK**: astrapy 1.x  
**Status**: Production
