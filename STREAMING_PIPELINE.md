# Streaming Pipeline Documentation

> **Real-Time Financial News Processing with Kafka & Spark Structured Streaming**  
> **LLM-Powered Sentiment Analysis and AstraDB Integration**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Components](#components)
- [NewsAPI Integration](#newsapi-integration)
- [GPT Sentiment Analysis](#gpt-sentiment-analysis)
- [Kafka Streaming](#kafka-streaming)
- [Spark Structured Streaming](#spark-structured-streaming)
- [AstraDB Storage](#astradb-storage)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Monitoring](#monitoring)

---

## Overview

The **Streaming Pipeline** processes real-time financial news to generate sentiment scores for **29 stocks**, enriching daily price predictions with market sentiment. It integrates:
- **NewsAPI** for financial news ingestion
- **OpenAI GPT-3.5** for sentiment analysis (1-5 scale)
- **Apache Kafka** for message streaming
- **Spark Structured Streaming** for distributed processing
- **AstraDB (Cassandra)** for scalable NoSQL storage

### **Key Features**

✅ **Real-time Processing**: Stream financial news as it's published  
✅ **LLM-Powered Sentiment**: GPT-3.5 Turbo analyzes news sentiment  
✅ **Scalable Storage**: AstraDB with multi-region support  
✅ **Batch Integration**: Sentiment data enriches training/inference pipelines  
✅ **Multi-Stock Support**: 29 stocks monitored simultaneously  

### **Business Value**

- **Market Sentiment Tracking**: Real-time news sentiment for trading decisions
- **Prediction Enhancement**: Sentiment features improve ML model accuracy
- **Historical Analysis**: Full sentiment history for backtesting
- **Alert Generation**: React to significant sentiment changes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STREAMING PIPELINE ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   NewsAPI        │  Financial news source
│   (REST API)     │  • Stock-specific queries
│                  │  • Real-time updates
│   29 stocks:     │  • Article metadata
│   AAPL, GOOG,    │  • Content extraction
│   TSLA, etc.     │
└────────┬─────────┘
         │ HTTP GET
         │ Polling (every N minutes)
         ▼
┌────────────────────────────────────┐
│   News Producer                    │
│   (Python Script)                  │
│                                    │
│   • Fetch news per stock           │
│   • Extract metadata               │
│   • Format JSON messages           │
│   • Publish to Kafka topic         │
└────────┬───────────────────────────┘
         │ Kafka Publish
         │ Topic: financial_news
         ▼
┌──────────────────────────────────────────────────┐
│   Apache Kafka 3.3+                              │
│   (Message Broker)                               │
│                                                   │
│   Topic: financial_news                          │
│   Partitions: 3                                  │
│   Replication: 1 (local dev)                     │
│                                                   │
│   Message Schema:                                │
│   {                                              │
│     "stock_symbol": "AAPL",                      │
│     "title": "...",                              │
│     "description": "...",                        │
│     "content": "...",                            │
│     "url": "...",                                │
│     "published_at": "2025-11-26T10:30:00Z",     │
│     "source": "Bloomberg"                        │
│   }                                              │
└────────┬─────────────────────────────────────────┘
         │ Kafka Consume
         │ Subscribe: financial_news
         ▼
┌─────────────────────────────────────────────────────┐
│   Spark Structured Streaming                        │
│   (Consumer + Processor)                            │
│                                                      │
│   1️⃣  Read Kafka Stream                              │
│   2️⃣  Parse JSON Messages                            │
│   3️⃣  Batch News (4 articles/batch)                  │
│   4️⃣  Call GPT Sentiment API                         │
│   5️⃣  Process Sentiment Scores                       │
│   6️⃣  Write to AstraDB                                │
│                                                      │
│   Processing Mode: Micro-batch (10 sec interval)   │
│   Parallelism: 8 cores (local[8])                  │
└────────┬────────────────────────────────────────────┘
         │
         │ ┌────────────────────────┐
         │ │  GPT-3.5 Turbo API     │
         ├─┤  (OpenAI)              │
         │ │                        │
         │ │  Input: 4 news texts   │
         │ │  Output: 5,3,4,5       │
         │ │  (1-5 scale)           │
         │ │                        │
         │ │  Formula:              │
         │ │  • 1 = Negative        │
         │ │  • 2 = Somewhat -      │
         │ │  • 3 = Neutral         │
         │ │  • 4 = Somewhat +      │
         │ │  • 5 = Positive        │
         │ └────────────────────────┘
         │
         ▼ Write to AstraDB
┌───────────────────────────────────────────────────┐
│   AstraDB (Cassandra)                             │
│   (NoSQL Storage)                                 │
│                                                    │
│   Table: news_sentiment_1                         │
│                                                    │
│   Schema:                                         │
│   • stock_symbol (TEXT, Partition Key)           │
│   • published_at (TIMESTAMP, Clustering Key)     │
│   • title (TEXT)                                 │
│   • description (TEXT)                           │
│   • url (TEXT)                                   │
│   • source (TEXT)                                │
│   • sentiment_score (DOUBLE)                     │
│   • scaled_sentiment (DOUBLE)                    │
│   • created_at (TIMESTAMP)                       │
│                                                    │
│   Query Pattern:                                  │
│   SELECT * FROM news_sentiment_1                 │
│   WHERE stock_symbol = 'AAPL'                    │
│   AND published_at >= '2025-11-01'               │
└────────┬──────────────────────────────────────────┘
         │
         │ Daily Read (Airflow DAG)
         ▼
┌───────────────────────────────────────────────────┐
│   Sentiment Enrichment                            │
│   (Daily Batch Process)                           │
│                                                    │
│   • Read YFinance daily data                      │
│   • Query AstraDB for sentiment (per stock/date) │
│   • Join price + sentiment                        │
│   • Calculate Scaled_sentiment: (score-0.9999)/4 │
│   • Add News_flag (binary indicator)             │
│   • Save to tmp/yfinance_enriched.parquet        │
│   • Load into Delta tables                        │
│                                                    │
│   File: src/compute_sentiment_columns.py         │
└───────────────────────────────────────────────────┘
```

---

## Data Flow

### **End-to-End Flow**

```
1. NewsAPI Query
   ↓
2. News Producer → Kafka Topic
   ↓
3. Spark Structured Streaming Consumer
   ↓
4. GPT Sentiment Analysis (Batch of 4)
   ↓
5. AstraDB Storage (news_sentiment_1 table)
   ↓
6. Daily Enrichment (Airflow DAG)
   ↓
7. Delta Lake Storage (stock_<SYMBOL> tables)
   ↓
8. Training/Inference Pipeline (ML features)
```

### **Data Transformation**

#### **Stage 1: News Ingestion** (NewsAPI → Kafka)

```json
// Input: NewsAPI Response
{
  "articles": [
    {
      "source": {"name": "Bloomberg"},
      "title": "Apple Reports Q4 Earnings Beat",
      "description": "Apple Inc. exceeded analyst expectations...",
      "url": "https://...",
      "publishedAt": "2025-11-26T10:30:00Z",
      "content": "Full article text..."
    }
  ]
}

// Transform to Kafka Message
{
  "stock_symbol": "AAPL",
  "title": "Apple Reports Q4 Earnings Beat",
  "description": "Apple Inc. exceeded analyst expectations...",
  "content": "Full article text...",
  "url": "https://...",
  "published_at": "2025-11-26T10:30:00Z",
  "source": "Bloomberg"
}
```

#### **Stage 2: Sentiment Analysis** (GPT Processing)

```python
# Input: Batch of 4 news articles
texts = [
    "Apple (AAPL) increase 22%",
    "Apple (AAPL) price decreased 30%",
    "Apple announces iPhone 15",
    "Apple will release VisionPro on Feb 2, 2024"
]

# GPT Prompt
conversation = [
    {"role": "system", "content": "You are a financial expert..."},
    {"role": "user", "content": "News to Stock Symbol -- AAPL: ..."},
    {"role": "assistant", "content": "5, 1, 4, 4"}
]

# Output: Sentiment scores (1-5)
sentiments = [5, 1, 4, 4]
```

#### **Stage 3: Storage** (AstraDB)

```sql
-- Insert into news_sentiment_1
INSERT INTO news_sentiment_1 (
    stock_symbol,
    published_at,
    title,
    sentiment_score,
    scaled_sentiment,
    created_at
) VALUES (
    'AAPL',
    '2025-11-26 10:30:00',
    'Apple Reports Q4 Earnings Beat',
    5.0,
    1.000025,  -- (5 - 0.9999) / 4
    '2025-11-26 11:00:00'
);
```

#### **Stage 4: Daily Enrichment** (Batch Processing)

```python
# Query AstraDB for daily sentiment
astra_df = load_astra_rows(stocks=['AAPL', 'GOOG', ...])

# Filter to relevant dates
astra_df['event_date'] = astra_df['published_at'].str[:10]  # YYYY-MM-DD

# Group by stock + date, calculate mean sentiment
daily_sentiment = astra_df.groupby(['stock_symbol', 'event_date'])['sentiment_score'].mean()

# Join with YFinance data
enriched = yf_df.merge(daily_sentiment, on=['stock_symbol', 'date'])

# Add features
enriched['Sentiment_gpt'] = enriched['sentiment_score'].fillna(0.0)
enriched['News_flag'] = (enriched['Sentiment_gpt'] != 0).astype(int)
enriched['Scaled_sentiment'] = (enriched['Sentiment_gpt'] - 0.9999) / 4

# Output columns (11 total)
['stock_symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_close',
 'Sentiment_gpt', 'News_flag', 'Scaled_sentiment']
```

---

## Components

### **1. News Producer** (Python Script)

**Responsibility**: Fetch news from NewsAPI and publish to Kafka

```python
import requests
from kafka import KafkaProducer
import json
from datetime import datetime

class NewsProducer:
    def __init__(self, api_key, kafka_bootstrap_servers):
        self.api_key = api_key
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = 'financial_news'
    
    def fetch_news(self, stock_symbol):
        """Fetch news for a specific stock from NewsAPI"""
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': f'{stock_symbol} OR {self.get_company_name(stock_symbol)}',
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10
        }
        response = requests.get(url, params=params)
        return response.json()
    
    def publish_news(self, stock_symbol):
        """Publish news articles to Kafka"""
        news_data = self.fetch_news(stock_symbol)
        
        for article in news_data.get('articles', []):
            message = {
                'stock_symbol': stock_symbol,
                'title': article.get('title'),
                'description': article.get('description'),
                'content': article.get('content'),
                'url': article.get('url'),
                'published_at': article.get('publishedAt'),
                'source': article.get('source', {}).get('name')
            }
            
            self.producer.send(self.topic, value=message)
            print(f"[PRODUCER] Published news for {stock_symbol}")
        
        self.producer.flush()
```

**Deployment**: Run as cron job or continuous service

```bash
# Cron job (every 15 minutes)
*/15 * * * * /path/to/conda/envs/breaking_data/bin/python /path/to/news_producer.py

# Or as continuous service
python news_producer.py --mode continuous --interval 900  # 15 min
```

### **2. GPT Sentiment Analyzer**

**File**: `FNSPID_Financial_News_Dataset/data_processor/score_by_gpt.py`

**Key Function**:

```python
def get_sentiment(symbol, *texts):
    """
    Use GPT-3.5 Turbo to score sentiment (1-5) for financial news.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        *texts: Variable number of news texts (up to 4)
    
    Returns:
        List of sentiment scores [1-5] or [np.nan] on error
    """
    client = OpenAI(api_key='sk-...')
    
    # Build conversation context
    conversation = [
        {
            "role": "system",
            "content": "You are a financial expert with stock recommendation experience. "
                      "Score news sentiment from 1 to 5: "
                      "1=negative, 2=somewhat negative, 3=neutral, "
                      "4=somewhat positive, 5=positive."
        },
        {
            "role": "user",
            "content": "News to Stock Symbol -- AAPL: Apple increase 22% ### "
                      "News to Stock Symbol -- AAPL: Apple price decreased 30%"
        },
        {"role": "assistant", "content": "5, 1"},
        {"role": "user", "content": " ".join([f"News to Stock Symbol -- {symbol}: {text}" 
                                              for text in texts])}
    ]
    
    # Call GPT API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0,
        max_tokens=50
    )
    
    # Parse sentiment scores
    content = response.choices[0].message.content
    sentiments = [int(s.strip()) for s in content.split(',')]
    return sentiments
```

**GPT Prompt Engineering**:
- **System Role**: Financial expert with scoring guidelines
- **Few-shot Examples**: 2 example conversations with correct outputs
- **Temperature**: 0 (deterministic responses)
- **Max Tokens**: 50 (short numeric output)

**Sentiment Scale**:
| Score | Meaning | Example |
|-------|---------|---------|
| 1 | Negative | "Stock price plummeted 30%" |
| 2 | Somewhat Negative | "Quarterly earnings missed expectations" |
| 3 | Neutral | "Company announces routine board meeting" |
| 4 | Somewhat Positive | "New product announced" |
| 5 | Positive | "Stock surged 25% on earnings beat" |

### **3. Kafka Configuration**

**Topic Setup**:

```bash
# Create topic (if not exists)
kafka-topics.sh --create \
    --bootstrap-server localhost:9092 \
    --replication-factor 1 \
    --partitions 3 \
    --topic financial_news

# Verify topic
kafka-topics.sh --describe \
    --bootstrap-server localhost:9092 \
    --topic financial_news
```

**Configuration**:
```yaml
Topic: financial_news
Partitions: 3 (parallel processing)
Replication Factor: 1 (local dev)
Retention: 7 days (604800000 ms)
Compression: snappy
```

### **4. Spark Structured Streaming Consumer**

**Pseudo-code** (Production Implementation):

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import *

# Create Spark session
spark = SparkSession.builder \
    .appName("FinancialNewsStreaming") \
    .config("spark.streaming.kafka.maxRatePerPartition", 100) \
    .getOrCreate()

# Define schema for Kafka messages
schema = StructType([
    StructField("stock_symbol", StringType()),
    StructField("title", StringType()),
    StructField("description", StringType()),
    StructField("content", StringType()),
    StructField("url", StringType()),
    StructField("published_at", TimestampType()),
    StructField("source", StringType())
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "financial_news") \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON messages
parsed = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Batch news for GPT processing (window of 4 articles)
windowed = parsed.groupBy(
    col("stock_symbol"),
    window(col("published_at"), "10 minutes")
).agg(
    collect_list("title").alias("titles"),
    collect_list("content").alias("contents")
)

# UDF for GPT sentiment analysis
@udf(returnType=ArrayType(DoubleType()))
def get_sentiments(symbol, contents):
    # Call GPT API (batch of 4)
    return get_sentiment(symbol, *contents[:4])

# Apply sentiment analysis
with_sentiment = windowed.withColumn(
    "sentiments",
    get_sentiments(col("stock_symbol"), col("contents"))
)

# Write to AstraDB
query = with_sentiment.writeStream \
    .foreachBatch(lambda batch_df, batch_id: write_to_astradb(batch_df)) \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .start()

query.awaitTermination()
```

**Streaming Configuration**:
```yaml
Processing Time: 10 seconds (micro-batch)
Max Rate per Partition: 100 msgs/sec
Checkpoint Location: /tmp/streaming_checkpoint
Watermark: 5 minutes (late data tolerance)
```

### **5. AstraDB Integration**

**File**: `src/compute_sentiment_columns.py`

**Key Functions**:

```python
from astrapy import DataAPIClient

def load_astra_rows(db_endpoint, db_token, stocks=None):
    """
    Load sentiment data from AstraDB.
    
    Args:
        db_endpoint: AstraDB endpoint URL
        db_token: Authentication token
        stocks: List of stock symbols to filter (optional)
    
    Returns:
        DataFrame with sentiment data
    """
    client = DataAPIClient()
    db = client.get_database(db_endpoint, token=db_token)
    table = db.get_table("news_sentiment_1")
    
    rows = []
    if stocks:
        # Query per stock (efficient with partition key)
        for stock in stocks:
            rows.extend(list(table.find({"stock_symbol": stock.upper()})))
    else:
        # Full table scan (use sparingly)
        rows = list(table.find({}))
    
    return pd.DataFrame(rows)

def compute_daily_sentiments(yf_df, astra_df):
    """
    Enrich YFinance data with daily average sentiment from AstraDB.
    
    Process:
    1. Extract date from published_at (YYYY-MM-DD)
    2. Filter Astra rows to relevant (stock, date) pairs
    3. Group by (stock, date) and calculate mean sentiment
    4. Left join with YFinance data
    5. Add Sentiment_gpt, News_flag, Scaled_sentiment columns
    """
    # Extract dates
    astra_df["event_date"] = astra_df["published_at"].str[:10]
    yf_df["date"] = pd.to_datetime(yf_df["Date"]).dt.strftime("%Y-%m-%d")
    
    # Filter to relevant keys
    relevant_keys = set(zip(yf_df["stock_symbol"], yf_df["date"]))
    astra_df = astra_df[astra_df[["stock_symbol", "event_date"]].apply(
        lambda x: tuple(x) in relevant_keys, axis=1
    )]
    
    # Group and calculate mean
    grouped = astra_df.groupby(["stock_symbol", "event_date"])["sentiment_score"] \
                      .mean() \
                      .reset_index() \
                      .rename(columns={"sentiment_score": "Sentiment_gpt"})
    
    # Merge
    merged = yf_df.merge(grouped, left_on=["stock_symbol", "date"],
                         right_on=["stock_symbol", "event_date"], how="left")
    
    # Add features
    merged["Sentiment_gpt"] = merged["Sentiment_gpt"].fillna(0.0)
    merged["News_flag"] = (merged["Sentiment_gpt"] != 0).astype(int)
    merged["Scaled_sentiment"] = merged["Sentiment_gpt"].apply(
        lambda x: (x - 0.9999) / 4 if x != 0 else 0.0
    )
    
    return merged
```

---

## NewsAPI Integration

### **API Configuration**

```python
# NewsAPI Endpoint
url = "https://newsapi.org/v2/everything"

# Query Parameters
params = {
    'q': 'AAPL OR Apple',           # Search query (stock symbol or company name)
    'apiKey': 'YOUR_API_KEY',        # Authentication
    'language': 'en',                # English only
    'sortBy': 'publishedAt',         # Most recent first
    'pageSize': 10,                  # Articles per request
    'from': '2025-11-26',            # Start date (optional)
    'to': '2025-11-27'               # End date (optional)
}
```

### **Stock-to-Company Mapping**

```python
STOCK_COMPANY_MAP = {
    'AAPL': 'Apple',
    'GOOG': 'Google OR Alphabet',
    'TSLA': 'Tesla',
    'NVDA': 'NVIDIA',
    'BABA': 'Alibaba',
    'AMZN': 'Amazon',
    # ... (29 stocks total)
}
```

### **Rate Limits**

**Free Tier**:
- 100 requests/day
- 1 request/second

**Developer Tier** (Recommended for Production):
- 250,000 requests/month
- ~350 requests/hour sustained
- $449/month

**Optimization**:
```python
# Batch multiple stocks in single query
query = 'AAPL OR GOOG OR TSLA'  # Up to 5 stocks per request

# Cache results to minimize API calls
import redis
cache = redis.Redis()
cache_key = f"news:{stock}:{date}"
```

---

## GPT Sentiment Analysis

### **API Configuration**

```python
from openai import OpenAI

client = OpenAI(api_key='sk-...')

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=conversation,
    temperature=0,        # Deterministic
    max_tokens=50,        # Short numeric output
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0
)
```

### **Prompt Design**

**System Prompt**:
```
You are a financial expert with stock recommendation experience. 
Based on a specific stock, score for range from 1 to 5, where:
- 1 is negative
- 2 is somewhat negative
- 3 is neutral
- 4 is somewhat positive
- 5 is positive

{num_text} summarized news will be passed in each time, you will 
give scores in format as shown below in the response from assistant.
```

**Few-Shot Examples**:
```
User: "News to Stock Symbol -- AAPL: Apple increase 22% ### 
       News to Stock Symbol -- AAPL: Apple price decreased 30%"
Assistant: "5, 1"

User: "News to Stock Symbol -- AAPL: Apple announced iPhone 15 ### 
       News to Stock Symbol -- AAPL: Apple will release VisionPro"
Assistant: "4, 4"
```

### **Batch Processing Strategy**

```python
# Process 4 articles at a time (optimal for GPT-3.5)
batch_size = 4

for i in range(0, len(articles), batch_size):
    batch = articles[i:i+batch_size]
    sentiments = get_sentiment(stock_symbol, *batch)
    
    for article, sentiment in zip(batch, sentiments):
        save_to_astradb(article, sentiment)
```

**Why Batch Size = 4?**
- Balances API efficiency (fewer calls) with accuracy
- GPT-3.5 context window: 4,096 tokens (4 articles ~1,000 tokens)
- Error recovery: smaller batches = less data loss on failure

### **Cost Analysis**

**GPT-3.5 Turbo Pricing** (as of 2025):
- Input: $0.50 / 1M tokens
- Output: $1.50 / 1M tokens

**Daily Cost Estimate** (29 stocks, 10 articles/stock/day):
```
Articles/day: 29 × 10 = 290
Batches: 290 / 4 = 73
Tokens/batch: ~250 (input) + ~10 (output)
Daily tokens: 73 × 260 = 18,980
Monthly cost: 18,980 × 30 × $0.50 / 1M ≈ $0.28/month
```

**Very cost-effective** for production use.

### **Scaling Formula**

The **Scaled_sentiment** feature normalizes GPT scores (1-5) to ~0-1 range:

```python
Scaled_sentiment = (Sentiment_gpt - 0.9999) / 4

# Examples:
# Sentiment_gpt = 1 (negative)  → Scaled = (1 - 0.9999) / 4 = 0.000025
# Sentiment_gpt = 3 (neutral)   → Scaled = (3 - 0.9999) / 4 = 0.500025
# Sentiment_gpt = 5 (positive)  → Scaled = (5 - 0.9999) / 4 = 1.000025

# Special case:
# Sentiment_gpt = 0 (no news)   → Scaled = 0.0 (unchanged)
```

**Rationale**: ML models perform better with normalized features (0-1 range).

---

## Kafka Streaming

### **Kafka Setup**

```bash
# 1. Start Zookeeper
zookeeper-server-start.sh config/zookeeper.properties &

# 2. Start Kafka Broker
kafka-server-start.sh config/server.properties &

# 3. Create Topic
kafka-topics.sh --create \
    --bootstrap-server localhost:9092 \
    --replication-factor 1 \
    --partitions 3 \
    --topic financial_news \
    --config retention.ms=604800000 \
    --config compression.type=snappy

# 4. Verify
kafka-topics.sh --describe \
    --bootstrap-server localhost:9092 \
    --topic financial_news
```

### **Producer Configuration**

```python
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    compression_type='snappy',
    acks='all',                    # Wait for all replicas
    retries=3,                     # Retry on failure
    batch_size=16384,              # Batch messages for efficiency
    linger_ms=10,                  # Wait 10ms for batching
    buffer_memory=33554432         # 32 MB buffer
)
```

### **Consumer Configuration** (Spark)

```python
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "financial_news") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .option("kafka.max.partition.fetch.bytes", "1048576") \
    .load()
```

### **Message Schema**

```json
{
  "stock_symbol": "AAPL",
  "title": "Apple Reports Record Quarterly Revenue",
  "description": "Apple Inc. reported record quarterly revenue...",
  "content": "Full article text (up to 5,000 chars)...",
  "url": "https://www.bloomberg.com/news/...",
  "published_at": "2025-11-26T10:30:00Z",
  "source": "Bloomberg",
  "timestamp": 1732618200000
}
```

### **Monitoring Commands**

```bash
# Monitor consumer lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
    --describe --group spark-streaming-consumer

# View topic messages
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
    --topic financial_news --from-beginning

# Check topic throughput
kafka-run-class.sh kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 \
    --topic financial_news --time -1
```

---

## Spark Structured Streaming

### **Session Configuration**

```python
spark = SparkSession.builder \
    .appName("FinancialNewsStreaming") \
    .master("local[8]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.streaming.checkpointLocation", "/tmp/streaming_checkpoint") \
    .config("spark.streaming.kafka.maxRatePerPartition", 100) \
    .config("spark.sql.streaming.stateStore.providerClass", 
            "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider") \
    .getOrCreate()
```

### **Processing Pipeline**

```python
# 1. Read Kafka stream
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "financial_news") \
    .load()

# 2. Parse JSON
from pyspark.sql.functions import from_json, col

schema = StructType([...])  # Define message schema
parsed_df = kafka_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# 3. Window aggregation (10-minute windows)
from pyspark.sql.functions import window, collect_list

windowed_df = parsed_df \
    .withWatermark("published_at", "5 minutes") \
    .groupBy(
        col("stock_symbol"),
        window(col("published_at"), "10 minutes")
    ) \
    .agg(
        collect_list("title").alias("titles"),
        collect_list("content").alias("contents")
    )

# 4. Apply GPT sentiment UDF
from pyspark.sql.functions import udf

@udf(returnType=ArrayType(DoubleType()))
def get_sentiments_udf(symbol, contents):
    return get_sentiment(symbol, *contents[:4])

with_sentiment = windowed_df.withColumn(
    "sentiments",
    get_sentiments_udf(col("stock_symbol"), col("contents"))
)

# 5. Write to AstraDB
def write_to_astradb(batch_df, batch_id):
    """Write each micro-batch to AstraDB"""
    pandas_df = batch_df.toPandas()
    # Insert into AstraDB (batch insert)
    astra_client.batch_insert("news_sentiment_1", pandas_df)

query = with_sentiment.writeStream \
    .foreachBatch(write_to_astradb) \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .option("checkpointLocation", "/tmp/streaming_checkpoint") \
    .start()

query.awaitTermination()
```

### **Performance Tuning**

```yaml
# Micro-batch interval
trigger(processingTime="10 seconds")  # Process every 10 sec

# Parallelism
spark.default.parallelism: 8
spark.sql.shuffle.partitions: 8

# Memory
spark.driver.memory: 8g
spark.executor.memory: 6g
spark.memory.fraction: 0.8

# Checkpointing
checkpointLocation: /tmp/streaming_checkpoint
cleanSource: delete  # Delete processed files

# Watermarking (handle late data)
withWatermark("published_at", "5 minutes")
```

---

## AstraDB Storage

### **Database Schema**

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

**Key Design**:
- **Partition Key**: `stock_symbol` (efficient per-stock queries)
- **Clustering Key**: `published_at` (time-series ordering)
- **Order**: DESC (most recent first)

### **Data Access Patterns**

```python
from astrapy import DataAPIClient

# 1. Query by stock symbol (efficient - uses partition key)
client = DataAPIClient()
db = client.get_database(endpoint, token=token)
table = db.get_table("news_sentiment_1")

# Get all sentiment for AAPL
aapl_sentiment = list(table.find({"stock_symbol": "AAPL"}))

# 2. Query by stock + date range
from datetime import datetime, timedelta

start_date = datetime(2025, 11, 1)
end_date = datetime(2025, 11, 30)

aapl_nov = list(table.find({
    "stock_symbol": "AAPL",
    "published_at": {"$gte": start_date, "$lt": end_date}
}))

# 3. Aggregate daily sentiment (done in Python/Pandas)
import pandas as pd

df = pd.DataFrame(aapl_sentiment)
df['date'] = pd.to_datetime(df['published_at']).dt.date
daily_avg = df.groupby('date')['sentiment_score'].mean()
```

### **Configuration** (config/config.yaml)

```yaml
astradb:
  endpoint: "https://YOUR-DB-ID-YOUR-REGION.apps.astra.datastax.com"
  token: "AstraCS:..."
  table_name: "news_sentiment_1"
  keyspace: "default_keyspace"
  region: "us-east-1"
```

### **Connection Example**

```python
from astrapy import DataAPIClient
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

astra_config = config['astradb']

# Connect
client = DataAPIClient()
db = client.get_database(
    astra_config['endpoint'],
    token=astra_config['token']
)

# Insert data
table = db.get_table("news_sentiment_1")
table.insert_one({
    "stock_symbol": "AAPL",
    "published_at": datetime.now(),
    "title": "Apple announces new product",
    "sentiment_score": 4.5,
    "scaled_sentiment": 0.8750625
})
```

---

## Configuration

### **Environment Variables**

```bash
# NewsAPI
export NEWS_API_KEY="your-newsapi-key"

# OpenAI GPT
export OPENAI_API_KEY="sk-..."

# Kafka
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"

# AstraDB
export ASTRA_DB_ENDPOINT="https://..."
export ASTRA_DB_TOKEN="AstraCS:..."

# Spark
export SPARK_HOME="/path/to/spark"
export PYSPARK_PYTHON="/path/to/conda/envs/breaking_data/bin/python"
```

### **config/config.yaml**

```yaml
streaming:
  newsapi:
    api_key: ${NEWS_API_KEY}
    base_url: "https://newsapi.org/v2"
    polling_interval: 900  # 15 minutes
  
  kafka:
    bootstrap_servers: "localhost:9092"
    topic: "financial_news"
    partitions: 3
    replication_factor: 1
  
  gpt:
    api_key: ${OPENAI_API_KEY}
    model: "gpt-3.5-turbo"
    batch_size: 4
    temperature: 0
    max_tokens: 50
  
  spark:
    app_name: "FinancialNewsStreaming"
    master: "local[8]"
    checkpoint_location: "/tmp/streaming_checkpoint"
    processing_time: "10 seconds"

astradb:
  endpoint: "https://YOUR-DB-ID.apps.astra.datastax.com"
  token: ${ASTRA_DB_TOKEN}
  table_name: "news_sentiment_1"
  keyspace: "default_keyspace"
```

---

## Usage Guide

### **1. Start Streaming Pipeline**

```bash
# 1. Start Kafka infrastructure
./start_kafka.sh

# 2. Start News Producer (continuous mode)
conda activate breaking_data
python scripts/news_producer.py --mode continuous --interval 900

# 3. Start Spark Structured Streaming Consumer
spark-submit \
    --master local[8] \
    --driver-memory 8g \
    --executor-memory 6g \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
    scripts/streaming_consumer.py

# 4. Monitor Kafka topic
kafka-console-consumer.sh \
    --bootstrap-server localhost:9092 \
    --topic financial_news
```

### **2. Daily Enrichment (Airflow DAG)**

Already integrated in **Stage 2** of the daily DAG:

```python
# File: scripts/stock_pipeline_dag.py

enrich_task = PythonOperator(
    task_id="enrich_with_sentiment",
    python_callable=run_sentiment_enrichment,
    op_kwargs={
        "yfinance_path": YF_RAW_PATH,
        "output_path": YF_ENRICHED_PATH,
    },
    dag=dag,
)

# Extract → Enrich → Load → Inference
extract_task >> enrich_task >> load_task >> inference_task
```

### **3. Manual Sentiment Enrichment**

```bash
# Enrich a specific date's data
python src/compute_sentiment_columns.py \
    --yfinance-path tmp/yfinance_raw_2025-11-26.parquet \
    --output-path tmp/yfinance_enriched_2025-11-26.parquet
```

### **4. Query AstraDB**

```python
from src.compute_sentiment_columns import load_astra_rows

# Load sentiment for specific stocks
sentiment_df = load_astra_rows(stocks=['AAPL', 'GOOG', 'TSLA'])

# View data
print(sentiment_df.head())
print(f"Total rows: {len(sentiment_df)}")
print(f"Date range: {sentiment_df['published_at'].min()} to {sentiment_df['published_at'].max()}")

# Calculate daily averages
sentiment_df['date'] = pd.to_datetime(sentiment_df['published_at']).dt.date
daily = sentiment_df.groupby(['stock_symbol', 'date'])['sentiment_score'].mean()
print(daily)
```

---

## Monitoring

### **1. Kafka Monitoring**

```bash
# Consumer lag (how far behind is Spark consumer?)
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
    --describe --group spark-streaming-consumer

# Topic metrics
kafka-run-class.sh kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 \
    --topic financial_news --time -1
```

### **2. Spark Streaming UI**

```bash
# Access Spark UI
http://localhost:4040

# Key metrics to monitor:
# - Processing Time (should be < 10 sec)
# - Scheduling Delay (should be ~0)
# - Input Rate (messages/sec)
# - Batch Duration
```

### **3. GPT API Monitoring**

```python
# Track API usage
from openai import OpenAI
client = OpenAI()

# Get usage stats (requires API key with billing access)
usage = client.usage.retrieve()
print(f"Total tokens used today: {usage.total_tokens}")
print(f"Cost estimate: ${usage.total_tokens * 0.0015 / 1000:.4f}")
```

### **4. AstraDB Monitoring**

```python
# Query table statistics
from astrapy import DataAPIClient

client = DataAPIClient()
db = client.get_database(endpoint, token=token)
table = db.get_table("news_sentiment_1")

# Count total rows
total_rows = table.estimated_document_count()
print(f"Total sentiment records: {total_rows}")

# Count by stock
from collections import Counter
stocks = [row['stock_symbol'] for row in table.find({}, projection={'stock_symbol': 1})]
print(Counter(stocks))
```

### **5. Alerts**

```python
# Example: Alert on low sentiment processing rate
import smtplib

def check_sentiment_pipeline():
    """Alert if sentiment processing falls below threshold"""
    sentiment_df = load_astra_rows(stocks=['AAPL'])
    today = pd.Timestamp.now().date()
    today_count = len(sentiment_df[sentiment_df['published_at'].str[:10] == str(today)])
    
    if today_count < 5:  # Expect at least 5 news/day for AAPL
        send_alert(f"Low sentiment count for AAPL: {today_count}")

# Run as cron job
# 0 */4 * * * python check_sentiment_pipeline.py
```

---

## Summary

### **Key Components**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **News Source** | NewsAPI | Financial news ingestion |
| **Message Broker** | Kafka 3.3+ | Stream buffering & distribution |
| **Stream Processing** | Spark Structured Streaming | Real-time data transformation |
| **Sentiment Analysis** | OpenAI GPT-3.5 Turbo | AI-powered sentiment scoring |
| **Storage** | AstraDB (Cassandra) | Scalable NoSQL storage |
| **Batch Integration** | Python/Pandas | Daily enrichment for ML pipeline |

### **Data Flow Summary**

1. **News Ingestion**: NewsAPI → Kafka (every 15 min)
2. **Streaming**: Kafka → Spark → GPT → AstraDB (real-time)
3. **Enrichment**: AstraDB → Pandas → Delta Lake (daily batch)
4. **ML Pipeline**: Delta Lake → Training/Inference (Airflow)

### **Performance**

- **Latency**: 10-second micro-batches
- **Throughput**: 100 messages/sec/partition (300 total)
- **Cost**: ~$0.28/month (GPT API)
- **Scalability**: Horizontal scaling via Kafka partitions

### **Production Readiness**

✅ **Working**: Sentiment enrichment integrated in daily DAG  
⚠️ **Planned**: Full Kafka+Spark streaming deployment  
📊 **Operational**: AstraDB storage and querying functional  

---

## References

- **Enrichment Script**: `src/compute_sentiment_columns.py`
- **GPT Sentiment**: `FNSPID_Financial_News_Dataset/data_processor/score_by_gpt.py`
- **Configuration**: `config/config.yaml` (astradb section)
- **Airflow DAG**: `scripts/stock_pipeline_dag.py` (Stage 2: enrich_task)

---

**Last Updated**: November 26, 2025  
**Version**: 1.0.0  
**Status**: Production (Batch Enrichment), Planned (Full Streaming)
