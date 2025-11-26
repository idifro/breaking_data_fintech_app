#!/usr/bin/env python3
"""
Script to read and display top 10 rows from a stock delta table
Usage: python read_stock_delta.py [STOCK_SYMBOL]
Example: python read_stock_delta.py AAPL
"""

import sys
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

def read_stock_delta(stock_symbol="AAPL", num_rows=10):
    """
    Read and display top rows from a stock delta table
    
    Args:
        stock_symbol: Stock symbol (default: AAPL)
        num_rows: Number of rows to display (default: 10)
    """
    print(f"\n{'='*80}")
    print(f"Reading Delta Table: stock_{stock_symbol}")
    print(f"{'='*80}\n")
    
    # Initialize Spark session with Delta support
    builder = (
        SparkSession.builder
        .appName(f"read_stock_{stock_symbol}")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "2g")
    )
    
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # Path to delta table
        delta_path = f"delta_tables/stock_{stock_symbol}"
        
        # Read delta table
        df = spark.read.format("delta").load(delta_path)
        
        # Get basic info
        total_rows = df.count()
        columns = df.columns
        
        print(f"📊 Table Info:")
        print(f"   - Total Rows: {total_rows}")
        print(f"   - Columns: {len(columns)}")
        print(f"   - Column Names: {', '.join(columns)}")
        print()
        
        # Show schema
        print(f"📋 Schema:")
        df.printSchema()
        print()
        
        # Show top N rows
        print(f"📝 Top {num_rows} Rows:")
        print(f"{'-'*80}")
        df.tail(num_rows)
        
        # Show date range
        print(f"\n📅 Date Range:")
        date_stats = df.selectExpr("min(Date) as min_date", "max(Date) as max_date").collect()[0]
        print(f"   - First Date: {date_stats['min_date']}")
        print(f"   - Last Date:  {date_stats['max_date']}")
        print()
        
        # Show summary statistics for numeric columns
        print(f"📈 Summary Statistics:")
        print(f"{'-'*80}")
        numeric_cols = [f.name for f in df.schema.fields 
                       if str(f.dataType) in ['DoubleType', 'IntegerType', 'LongType', 'FloatType']]
        if numeric_cols:
            df.select(numeric_cols).describe().show()
        
    except Exception as e:
        print(f"❌ Error reading delta table: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop Spark session
        spark.stop()
        print(f"\n{'='*80}")
        print("✅ Done!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Get stock symbol from command line argument
    if len(sys.argv) > 1:
        stock = sys.argv[1].upper()
    else:
        stock = "AAPL"
    
    # Get number of rows from command line argument
    if len(sys.argv) > 2:
        try:
            num_rows = int(sys.argv[2])
        except ValueError:
            print("Warning: Invalid number of rows, using default (10)")
            num_rows = 10
    else:
        num_rows = 10
    
    read_stock_delta(stock, num_rows)
