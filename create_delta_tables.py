#!/usr/bin/env python3
"""
Delta Lake Table Restructuring Script
Creates separate Delta tables for stock data and simplified inference tables
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType, BooleanType
from src.utils import Config, Logger, load_environment

class DeltaTableRestructurer:
    """Restructure Delta tables into separate stock tables"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        self.base_path = Path("delta_tables")
        self.base_path.mkdir(exist_ok=True)
    
class DeltaTableRestructurer:
    """Restructure Delta tables into separate stock tables and simplified inference tables"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        self.base_path = Path("delta_tables")
        self.base_path.mkdir(exist_ok=True)
    
    def create_separate_stock_tables(self):
        """Create separate Delta tables for each stock from CSV files"""
        
        self.logger.info("🔄 Creating separate stock Delta tables from CSV...")
        
        csv_path = Path("data_csv")
        if not csv_path.exists():
            self.logger.error("❌ No CSV data found in data_csv/ directory")
            return
        
        # Process each CSV file
        csv_files = list(csv_path.glob("*.csv"))
        
        if not csv_files:
            self.logger.error("❌ No CSV files found in data_csv/ directory")
            return
        
        for csv_file in csv_files:
            stock_symbol = csv_file.stem.upper()
            
            # Skip if not in available stocks
            if stock_symbol not in self.config.data.available_stocks:
                self.logger.warning(f"⚠️ Skipping {stock_symbol} - not in available_stocks config")
                continue
                
            self.logger.info(f"� Processing {stock_symbol} from {csv_file.name}...")
            
            try:
                # Read CSV
                df = (self.spark.read
                     .option("header", "true")
                     .option("inferSchema", "true")
                     .csv(str(csv_file))
                     )
                
                # Fix column names for Delta compatibility (remove spaces and special characters)
                for old_col in df.columns:
                    new_col = old_col.replace(" ", "_").replace(".", "_")
                    if old_col != new_col:
                        df = df.withColumnRenamed(old_col, new_col)
                        self.logger.info(f"   Renamed column: '{old_col}' -> '{new_col}'")
                
                # Add stock symbol column if it doesn't exist
                if "stock_symbol" not in df.columns:
                    df = df.withColumn("stock_symbol", lit(stock_symbol))
                
                # Create stock table (full data for feature engineering)
                self._create_stock_table(df, stock_symbol)
                
                self.logger.info(f"✅ Processed {stock_symbol} successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {stock_symbol}: {e}")
    
    def _create_stock_table(self, df, stock_symbol):
        """Create full stock table for feature engineering and training"""
        
        stock_table_path = self.base_path / f"stock_{stock_symbol}"
        
        # Write Delta table with all columns (for feature engineering)
        (df.write
         .format("delta")
         .mode("overwrite")
         .option("overwriteSchema", "true")  # Handle schema conflicts
         .save(str(stock_table_path))
         )
        
        row_count = df.count()
        column_count = len(df.columns)
        self.logger.info(f"✅ Created stock_{stock_symbol} Delta table with {row_count:,} rows, {column_count} columns")
    
    def verify_tables(self):
        """Verify all tables were created correctly"""
        
        self.logger.info("🔍 Verifying created Delta tables...")
        
        for stock in self.config.data.available_stocks:
            # Check stock table (full data)
            stock_path = self.base_path / f"stock_{stock}"
            if stock_path.exists():
                try:
                    df = self.spark.read.format("delta").load(str(stock_path))
                    count = df.count()
                    columns = len(df.columns)
                    self.logger.info(f"✅ stock_{stock}: {count:,} rows, {columns} columns")
                except Exception as e:
                    self.logger.error(f"❌ Error reading stock_{stock}: {e}")
            else:
                self.logger.warning(f"⚠️ stock_{stock} Delta table not found")


def main():
    """Main restructuring function"""
    
    print("🚀 Starting Delta Lake Table Restructuring")
    print("=" * 60)
    
    try:
        # Load environment and config
        load_environment()
        config = Config()
        
        # Create Spark session with Delta Lake support
        from delta import configure_spark_with_delta_pip
        
        builder = SparkSession.builder \
            .appName("DeltaTableRestructuring") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        # Initialize restructurer
        restructurer = DeltaTableRestructurer(config, spark)
        
        # Create stock tables from CSV
        restructurer.create_separate_stock_tables()
        
        # Verify results
        restructurer.verify_tables()
        
        print("\n" + "=" * 60)
        print("🎉 Delta Lake table restructuring completed successfully!")
        print("=" * 60)
        
        print("\n📊 Created Delta Tables:")
        for stock in config.data.available_stocks:
            print(f"   📈 delta_tables/stock_{stock}/ (full data for feature engineering and training)")
        
        print(f"\n🎯 Training will use: {config.data.training_stocks}")
        print(f"🎯 Inference will use: {config.data.inference_stocks}")
        
        print(f"\n💡 Table Structure:")
        print(f"   📈 Stock tables: All columns for feature engineering and training")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Stock tables are ready for unified model training")
        print(f"   2. Run scripts/train_model.py to train the unified model")
        print(f"   3. Run scripts/inference.py to make predictions")
        
    except Exception as e:
        print(f"\n❌ Delta table restructuring failed: {e}")
        raise
    
    finally:
        if 'spark' in locals():
            spark.stop()


if __name__ == "__main__":
    main()