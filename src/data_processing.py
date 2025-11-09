"""
Data Processing Module for Stock Price Prediction
Handles data loading, preprocessing, and train/test splitting
"""

from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip

from src.utils import Config, PathManager
from src.feature_engineering import FeatureEngineer


class DataProcessor:
    """Data processing class for stock price prediction pipeline"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.path_manager = PathManager()
        self.feature_engineer = FeatureEngineer(config, spark)
    
    def load_csv_to_delta(self) -> None:
        """Load CSV files into Delta tables"""
        print("📁 Loading CSV files to Delta tables...")
        
        csv_files = self.path_manager.get_csv_files()
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data_csv directory")
        
        # Process each CSV file
        all_data = []
        
        for csv_path in csv_files:
            stock_symbol = csv_path.stem.upper()
            print(f"Processing {stock_symbol}...")
            
            # Read CSV with proper schema
            df = self.spark.read.option("header", "true") \
                              .option("inferSchema", "true") \
                              .csv(str(csv_path))
            
            # Add stock symbol
            df = df.withColumn("stock_symbol", lit(stock_symbol))
            
            # Ensure proper data types
            df = self._ensure_proper_types(df)
            
            # Validate data
            df = self._validate_data(df, stock_symbol)
            
            all_data.append(df)
        
        # Union all stock data
        if len(all_data) == 1:
            combined_df = all_data[0]
        else:
            combined_df = all_data[0]
            for df in all_data[1:]:
                combined_df = combined_df.union(df)
        
        # Sort by stock and date
        combined_df = combined_df.orderBy("stock_symbol", "Date")
        
        # Write to Delta table
        delta_path = str(self.path_manager.delta_tables_dir / "stock_data")
        combined_df.write.format("delta") \
                   .mode("overwrite") \
                   .option("overwriteSchema", "true") \
                   .save(delta_path)
        
        print(f"✅ Data loaded to Delta table: {delta_path}")
        print(f"Total rows: {combined_df.count()}")
        
        # Print data summary
        self._print_data_summary(combined_df)
    
    def _ensure_proper_types(self, df: DataFrame) -> DataFrame:
        """Ensure proper data types for all columns"""
        
        # Convert Date column to timestamp
        df = df.withColumn("Date", to_timestamp(col("Date")))
        
        # Rename columns with invalid characters for Delta
        if "Adj close" in df.columns:
            df = df.withColumnRenamed("Adj close", "Adj_Close")

        # Ensure numeric columns are double type
        numeric_columns = ["Open", "High", "Low", "Close", "Adj_Close", "Volume", 
                          "Sentiment_gpt", "Scaled_sentiment"]
        
        for col_name in numeric_columns:
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast("double"))
        
        # Ensure News_flag is integer
        if "News_flag" in df.columns:
            df = df.withColumn("News_flag", col("News_flag").cast("integer"))
        
        return df
    
    def _validate_data(self, df: DataFrame, stock_symbol: str) -> DataFrame:
        """Validate and clean data"""
        
        original_count = df.count()
        
        # Remove rows with null values in critical columns
        critical_columns = ["Date", "Close", "Volume", "Scaled_sentiment"]
        for col_name in critical_columns:
            if col_name in df.columns:
                df = df.filter(col(col_name).isNotNull())
        
        # Remove rows with invalid prices (negative or zero)
        df = df.filter(col("Close") > 0)
        if "Volume" in df.columns:
            df = df.filter(col("Volume") >= 0)
        
        # Remove extreme outliers (more than 50% daily change)
        # Fix: Add partitionBy for proper window partitioning
        window_spec = Window.partitionBy(lit(stock_symbol)).orderBy("Date")
        df = df.withColumn("prev_close", lag("Close", 1).over(window_spec))
        df = df.withColumn("daily_change", 
                          abs((col("Close") - col("prev_close")) / col("prev_close")))
        
        df = df.filter((col("daily_change").isNull()) | (col("daily_change") <= 0.5))
        df = df.drop("prev_close", "daily_change")
        
        final_count = df.count()
        removed_rows = original_count - final_count
        
        if removed_rows > 0:
            print(f"⚠️  {stock_symbol}: Removed {removed_rows} invalid rows")
        
        return df
    
    def _print_data_summary(self, df: DataFrame) -> None:
        """Print data summary statistics"""
        
        print("\n📊 Data Summary:")
        print("=" * 50)
        
        # Count by stock
        stock_counts = df.groupBy("stock_symbol").count().collect()
        for row in stock_counts:
            print(f"{row.stock_symbol}: {row.count} rows")
        
        # Date range
        date_range = df.agg(
            min("Date").alias("min_date"),
            max("Date").alias("max_date")
        ).collect()[0]
        
        print(f"\nDate range: {date_range.min_date} to {date_range.max_date}")
        
        # Basic statistics
        stats = df.select("Close", "Volume", "Scaled_sentiment").describe().collect()
        print("\nBasic Statistics:")
        for stat in stats:
            print(f"{stat.summary}: Close={float(stat.Close):.2f}, "
                  f"Volume={float(stat.Volume):.0f}, "
                  f"Sentiment={float(stat.Scaled_sentiment):.4f}")
    
    def load_data_from_delta(self) -> DataFrame:
        """Load data from Delta table"""
        
        delta_path = str(self.path_manager.delta_tables_dir / "stock_data")
        
        try:
            df = self.spark.read.format("delta").load(delta_path)
            print(f"✅ Loaded data from Delta table: {delta_path}")
            return df.orderBy("stock_symbol", "Date")
        
        except Exception as e:
            raise FileNotFoundError(f"Delta table not found: {delta_path}. Error: {e}")
    
    def prepare_training_data(self, df: DataFrame = None) -> Tuple[DataFrame, DataFrame, List[str]]:
        """
        Prepare training and test data with features
        
        Returns:
            Tuple of (train_df, test_df, feature_names)
        """
        
        if df is None:
            df = self.load_data_from_delta()
        
        print("🔧 Preparing training data...")
        
        # Create features
        df_with_features, feature_names = self.feature_engineer.create_all_features(df)
        
        # Split data by stock and time (following CNN approach)
        df_with_split = self._create_train_test_split(df_with_features)
        
        # Create feature vectors
        df_final, scaler_model = self.feature_engineer.create_feature_vector(
            df_with_split, feature_names
        )
        
        # Split into train and test DataFrames
        train_df = df_final.filter(col("split") == "train")
        test_df = df_final.filter(col("split") == "test")
        
        print(f"✅ Training data prepared:")
        print(f"   Training samples: {train_df.count()}")
        print(f"   Test samples: {test_df.count()}")
        print(f"   Features: {len(feature_names)}")
        
        # Save scaler model
        scaler_path = str(self.path_manager.models_dir / "feature_scaler")
        scaler_model.write().overwrite().save(scaler_path)
        print(f"✅ Scaler model saved to: {scaler_path}")
        
        return train_df, test_df, feature_names
    
    def _create_train_test_split(self, df: DataFrame) -> DataFrame:
        """Create train/test split by time for each stock (like CNN approach)"""
        
        print(f"📊 Creating train/test split ({self.config.data.train_split:.0%} train)...")
        
        # Process each stock separately
        stocks = df.select("stock_symbol").distinct().collect()
        split_dfs = []
        
        for stock_row in stocks:
            stock_symbol = stock_row.stock_symbol
            stock_df = df.filter(col("stock_symbol") == stock_symbol).orderBy("Date")
            
            # Calculate split point
            total_rows = stock_df.count()
            train_rows = int(total_rows * self.config.data.train_split)
            
            # Add row numbers - Fix: Partition by stock_symbol for proper window partitioning
            window_spec = Window.partitionBy("stock_symbol").orderBy("Date")
            stock_df = stock_df.withColumn("row_num", row_number().over(window_spec))
            
            # Add split column
            stock_df = stock_df.withColumn(
                "split",
                when(col("row_num") <= train_rows, "train").otherwise("test")
            )
            
            # Remove row_num column
            stock_df = stock_df.drop("row_num")
            
            split_dfs.append(stock_df)
            
            print(f"   {stock_symbol}: {train_rows} train, {total_rows - train_rows} test")
        
        # Union all stocks
        if len(split_dfs) == 1:
            result_df = split_dfs[0]
        else:
            result_df = split_dfs[0]
            for df_split in split_dfs[1:]:
                result_df = result_df.union(df_split)
        
        return result_df
    
    def load_scaler_model(self):
        """Load the saved scaler model"""
        from pyspark.ml.feature import StandardScalerModel
        
        scaler_path = str(self.path_manager.models_dir / "feature_scaler")
        
        try:
            scaler_model = StandardScalerModel.load(scaler_path)
            return scaler_model
        except Exception as e:
            raise FileNotFoundError(f"Scaler model not found: {scaler_path}. Error: {e}")
    
    def get_latest_data_for_inference(self, sequence_length: int = None) -> DataFrame:
        """
        Get the latest data for inference
        
        Args:
            sequence_length: Number of latest rows to get per stock
            
        Returns:
            DataFrame with latest data for each stock
        """
        
        if sequence_length is None:
            sequence_length = self.config.data.sequence_length
        
        df = self.load_data_from_delta()
        
        # Get latest rows for each stock - Fix: Add proper partitioning
        window_spec = Window.partitionBy("stock_symbol").orderBy(desc("Date"))
        
        latest_df = df.withColumn("row_num", row_number().over(window_spec)) \
                     .filter(col("row_num") <= sequence_length) \
                     .drop("row_num")
        
        return latest_df