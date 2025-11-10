#!/usr/bin/env python3
"""
Parquet Table Restructuring Script
Converts single stock_data table to separate parquet tables per stock
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from src.utils import Config, Logger, load_environment

class ParquetTableRestructurer:
    """Restructure tables into separate stock parquet tables"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        self.base_path = Path("parquet_tables")
        self.base_path.mkdir(exist_ok=True)
    
    def create_separate_stock_tables(self):
        """Create separate parquet tables for each stock"""
        
        self.logger.info("🔄 Starting parquet table restructuring...")
        
        # Read existing combined data
        try:
            combined_df = self.spark.read.parquet("delta_tables/stock_data")
            self.logger.info("✅ Loaded existing combined stock data from delta_tables")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load from delta_tables: {e}")
            # Try to load from existing parquet tables
            try:
                combined_df = self.spark.read.parquet("parquet_tables/stock_data")
                self.logger.info("✅ Loaded existing combined stock data from parquet_tables")
            except Exception as e2:
                self.logger.warning(f"⚠️ Failed to load from parquet_tables: {e2}")
                # If no existing data, load from CSV
                return self._create_from_csv()
        
        # Get unique stocks in the data
        stocks = [row.stock_symbol for row in combined_df.select("stock_symbol").distinct().collect()]
        self.logger.info(f"📊 Found stocks in data: {stocks}")
        
        # Create separate tables for each stock
        for stock in stocks:
            self._create_stock_table(combined_df, stock)
            self._create_inference_copy_table(combined_df, stock)
        
        self.logger.info("✅ Parquet table restructuring completed!")
    
    def _create_stock_table(self, combined_df, stock_symbol):
        """Create training table for a specific stock"""
        
        self.logger.info(f"📈 Creating stock table for {stock_symbol}...")
        
        # Filter data for this stock
        stock_df = combined_df.filter(col("stock_symbol") == stock_symbol)
        
        # Define table path
        stock_table_path = self.base_path / f"stock_{stock_symbol}"
        stock_table_path.mkdir(exist_ok=True, parents=True)
        
        # Write Parquet table
        stock_df.write \
            .mode("overwrite") \
            .parquet(str(stock_table_path))
        
        row_count = stock_df.count()
        self.logger.info(f"✅ Created stock_{stock_symbol} table with {row_count:,} rows")
    
    def _create_inference_copy_table(self, combined_df, stock_symbol):
        """Create inference copy table for a specific stock"""
        
        self.logger.info(f"🔮 Creating inference table for {stock_symbol}...")
        
        # Filter data for this stock
        stock_df = combined_df.filter(col("stock_symbol") == stock_symbol)
        
        # Define inference table path
        inference_table_path = self.base_path / f"inference_{stock_symbol}"
        inference_table_path.mkdir(exist_ok=True, parents=True)
        
        # Write Parquet table (copy of stock data for inference)
        stock_df.write \
            .mode("overwrite") \
            .parquet(str(inference_table_path))
        
        row_count = stock_df.count()
        self.logger.info(f"✅ Created inference_{stock_symbol} table with {row_count:,} rows")
    
    def _create_from_csv(self):
        """Create tables directly from CSV files if no existing tables"""
        
        self.logger.info("📁 Creating tables from CSV files...")
        
        csv_path = Path("data_csv")
        if not csv_path.exists():
            self.logger.error("❌ No CSV data found")
            return
        
        # Process each CSV file
        csv_files = list(csv_path.glob("*.csv"))
        
        for csv_file in csv_files:
            stock_symbol = csv_file.stem.upper()
            self.logger.info(f"📊 Processing {stock_symbol} from {csv_file.name}...")
            
            # Read CSV
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(str(csv_file))
            
            # Add stock symbol column if it doesn't exist
            if "stock_symbol" not in df.columns:
                df = df.withColumn("stock_symbol", lit(stock_symbol))
            
            # Create stock table
            stock_table_path = self.base_path / f"stock_{stock_symbol}"
            stock_table_path.mkdir(exist_ok=True, parents=True)
            
            df.write.mode("overwrite").parquet(str(stock_table_path))
            
            # Create inference copy
            inference_table_path = self.base_path / f"inference_{stock_symbol}"
            inference_table_path.mkdir(exist_ok=True, parents=True)
            
            df.write.mode("overwrite").parquet(str(inference_table_path))
            
            row_count = df.count()
            self.logger.info(f"✅ Created tables for {stock_symbol} with {row_count:,} rows")
    
    def verify_tables(self):
        """Verify all tables were created correctly"""
        
        self.logger.info("🔍 Verifying created tables...")
        
        available_stocks = self.config.data.available_stocks
        
        for stock in available_stocks:
            # Check stock table
            stock_path = self.base_path / f"stock_{stock}"
            if stock_path.exists():
                try:
                    df = self.spark.read.parquet(str(stock_path))
                    count = df.count()
                    self.logger.info(f"✅ stock_{stock}: {count:,} rows")
                except Exception as e:
                    self.logger.error(f"❌ Error reading stock_{stock}: {e}")
            else:
                self.logger.warning(f"⚠️ stock_{stock} table not found")
            
            # Check inference table
            inference_path = self.base_path / f"inference_{stock}"
            if inference_path.exists():
                try:
                    df = self.spark.read.parquet(str(inference_path))
                    count = df.count()
                    self.logger.info(f"✅ inference_{stock}: {count:,} rows")
                except Exception as e:
                    self.logger.error(f"❌ Error reading inference_{stock}: {e}")
            else:
                self.logger.warning(f"⚠️ inference_{stock} table not found")


def main():
    """Main restructuring function"""
    
    print("🚀 Starting Parquet Table Restructuring")
    print("=" * 50)
    
    try:
        # Load environment and config
        load_environment()
        config = Config()
        
        # Create Spark session (simplified without Delta)
        spark = SparkSession.builder \
            .appName("ParquetTableRestructuring") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        # Initialize restructurer
        restructurer = ParquetTableRestructurer(config, spark)
        
        # Perform restructuring
        restructurer.create_separate_stock_tables()
        
        # Verify results
        restructurer.verify_tables()
        
        print("\n" + "=" * 50)
        print("🎉 Parquet table restructuring completed successfully!")
        print("=" * 50)
        
        print("\n📊 Created Tables:")
        for stock in config.data.available_stocks:
            print(f"   📈 parquet_tables/stock_{stock}/")
            print(f"   🔮 parquet_tables/inference_{stock}/")
        
        print(f"\n🎯 Training will use: {config.data.training_stocks}")
        print(f"🎯 Inference will use: {config.data.inference_stocks}")
        
    except Exception as e:
        print(f"\n❌ Restructuring failed: {e}")
        raise
    
    finally:
        if 'spark' in locals():
            spark.stop()


if __name__ == "__main__":
    main()