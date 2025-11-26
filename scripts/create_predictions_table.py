#!/usr/bin/env python3
"""
Create Stock Predictions Delta Table
Sets up the Delta table structure for batch inference predictions
"""

import sys
from pathlib import Path

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from delta import configure_spark_with_delta_pip


def create_stock_predictions_table():
    """Create the stock_predictions Delta table with proper schema"""
    
    # Create Spark session
    builder = SparkSession.builder \
        .appName("CreateStockPredictionsTable") \
        .master("local[4]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    
    # Define schema for stock_predictions table
    schema = StructType([
        StructField("prediction_date", DateType(), nullable=False),
        StructField("prediction_timestamp", TimestampType(), nullable=False), 
        StructField("predicted_close", DoubleType(), nullable=False),
        StructField("model_confidence", DoubleType(), nullable=False),
        StructField("model_version", StringType(), nullable=False),
        StructField("features_used", StringType(), nullable=False),
        StructField("created_at", TimestampType(), nullable=False),
        StructField("days_ahead", IntegerType(), nullable=False),
        StructField("stock_symbol", StringType(), nullable=False)
    ])
    
    # Create empty DataFrame with schema
    empty_df = spark.createDataFrame([], schema)
    
    # Create Delta table path
    delta_table_path = "delta_tables/stock_predictions"
    Path(delta_table_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write empty table to establish schema
    empty_df.write \
           .format("delta") \
           .mode("overwrite") \
           .save(delta_table_path)
    
    print(f"✅ Stock predictions Delta table created: {delta_table_path}")
    print("📋 Schema:")
    empty_df.printSchema()
    
    spark.stop()


if __name__ == "__main__":
    create_stock_predictions_table()