"""
Data Processing Module for Stock Price Prediction
Handles data loading, preprocessing, and train/test splitting
"""

from typing import List, Tuple, Dict, Any
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
    
    def load_data_from_delta(self, stock_selection: List[str] = None, table_type: str = "stock") -> DataFrame:
        """
        Load data from stock-specific Delta tables
        
        Args:
            stock_selection: List of stocks to load (defaults to config training_stocks)
            table_type: "stock" for training data, "inference" for inference copies
            
        Returns:
            Combined DataFrame from selected stock tables
        """
        
        # Use config stock selection if not provided
        if stock_selection is None:
            stock_selection = self.config.data.training_stocks
        
        print(f"📊 Loading {table_type} data for stocks: {stock_selection}")
        
        # Load data from each stock-specific Delta table
        stock_dfs = []
        
        for stock in stock_selection:
            table_name = f"{table_type}_{stock}"
            delta_path = str(self.path_manager.delta_tables_dir / table_name)
            
            try:
                stock_df = self.spark.read.format("delta").load(delta_path)
                row_count = stock_df.count()
                print(f"   ✅ {table_name}: {row_count:,} rows")
                stock_dfs.append(stock_df)
                
            except Exception as e:
                print(f"   ⚠️ Failed to load {table_name}: {e}")
                continue
        
        if not stock_dfs:
            raise FileNotFoundError(f"No valid Delta tables found for stocks: {stock_selection}")
        
        # Combine all stock DataFrames
        if len(stock_dfs) == 1:
            combined_df = stock_dfs[0]
        else:
            combined_df = stock_dfs[0]
            for df in stock_dfs[1:]:
                combined_df = combined_df.union(df)
        
        total_rows = combined_df.count()
        print(f"✅ Combined data: {total_rows:,} total rows from {len(stock_dfs)} stocks")
        
        return combined_df.orderBy("stock_symbol", "Date")
    
    def prepare_training_data(self, stock_selection: List[str] = None) -> Tuple[DataFrame, DataFrame, List[str], Any]:
        """
        Prepare training and test data with features
        
        Args:
            stock_selection: List of stocks to train on (defaults to config training_stocks)
            
        Returns:
            Tuple of (train_df, test_df, feature_names, scaler_model)
        """
        
        # Load data from stock-specific Delta tables
        df = self.load_data_from_delta(stock_selection=stock_selection, table_type="stock")
        
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
        
        # Save scaler model , we are overwriting any existing model
        scaler_path = str(self.path_manager.models_dir / "feature_scaler")
        scaler_model.write().overwrite().save(scaler_path)
        print(f"✅ Scaler model saved to: {scaler_path}")
        
        return train_df, test_df, feature_names, scaler_model
    
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
    
    def get_latest_data_for_inference(self, stock_selection: List[str] = None, sequence_length: int = None) -> DataFrame:
        """
        Get the latest data for inference from inference tables
        
        Args:
            stock_selection: List of stocks for inference (defaults to config inference_stocks)
            sequence_length: Number of latest rows to get per stock
            
        Returns:
            DataFrame with latest data for each stock
        """
        
        if sequence_length is None:
            sequence_length = self.config.data.sequence_length
            
        # Use config inference stock selection if not provided
        if stock_selection is None:
            stock_selection = self.config.data.inference_stocks
        
        # Load data from inference Delta tables
        df = self.load_data_from_delta(stock_selection=stock_selection, table_type="inference")
        
        # Get latest rows for each stock
        window_spec = Window.partitionBy("stock_symbol").orderBy(desc("Date"))
        
        latest_df = df.withColumn("row_num", row_number().over(window_spec)) \
                     .filter(col("row_num") <= sequence_length) \
                     .drop("row_num")
        
        print(f"📊 Latest data for inference:")
        for stock in stock_selection:
            stock_count = latest_df.filter(col("stock_symbol") == stock).count()
            print(f"   {stock}: {stock_count} rows")
        
        return latest_df
    
    def load_legacy_combined_data(self) -> DataFrame:
        """
        Load data from legacy combined Delta table (for backwards compatibility)
        
        Returns:
            DataFrame from the original stock_data table
        """
        
        delta_path = str(self.path_manager.delta_tables_dir / "stock_data")
        
        try:
            df = self.spark.read.format("delta").load(delta_path)
            print(f"✅ Loaded legacy data from Delta table: {delta_path}")
            return df.orderBy("stock_symbol", "Date")
        
        except Exception as e:
            raise FileNotFoundError(f"Legacy Delta table not found: {delta_path}. Error: {e}")
    
    def get_available_stocks(self) -> List[str]:
        """
        Get list of available stocks from Delta tables
        
        Returns:
            List of stock symbols that have Delta tables
        """
        
        available_stocks = []
        
        for stock in self.config.data.available_stocks:
            stock_table_path = self.path_manager.delta_tables_dir / f"stock_{stock}"
            if stock_table_path.exists():
                available_stocks.append(stock)
        
        print(f"📊 Available stocks in Delta tables: {available_stocks}")
        return available_stocks
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names used for model training"""
        # Get feature names from FeatureEngineer to ensure consistency
        from src.feature_engineering import FeatureEngineer
        
        # Create feature engineer and get feature names
        feature_engineer = FeatureEngineer(self.config, self.spark)
        
        # Return a sample feature list - this will be updated when features are created
        # For now, return the main features we know exist
        feature_names = [
            # Core lag features
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
            
            # Price change features  
            'price_change_1d', 'price_change_3d', 'price_change_5d',
            
            # Moving averages
            'ma_5', 'ma_10', 'ma_20',
            'close_vs_ma5', 'close_vs_ma10', 'close_vs_ma20',
            
            # Technical indicators
            'high_low_spread', 'avg_gain_14', 'avg_loss_14', 'rsi_14',
            
            # Volume features
            'volume_lag_1', 'volume_ma_5', 'volume_ma_20', 'volume_change_1d',
            
            # Sentiment features
            'sentiment_lag_1', 'sentiment_lag_3', 'sentiment_ma_5', 'sentiment_ma_10', 'sentiment_ma_20',
            'volume_std_5', 'volume_std_10', 'volume_std_20',
            'volume_vs_sma_5', 'volume_vs_sma_10', 'volume_vs_sma_20',
            'price_volume_trend',
            
            # Sentiment features (if available)
            'Sentiment_gpt_lag_1', 'Sentiment_gpt_lag_2', 'Sentiment_gpt_lag_3',
            'Scaled_sentiment_lag_1', 'Scaled_sentiment_lag_2', 'Scaled_sentiment_lag_3',
            'sentiment_sma_3', 'sentiment_sma_5', 'sentiment_sma_10',
            'News_flag_lag_1', 'News_flag_lag_2', 'News_flag_lag_3',
            
            # Technical indicators
            'rsi_14', 'bb_upper', 'bb_lower', 'bb_position',
            'macd_line', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d',
            'atr_14', 'volatility_5', 'volatility_10', 'volatility_20'
        ]
        
        return feature_names
    
    def _create_features(self, df: DataFrame) -> DataFrame:
        """Create features for prediction using FeatureEngineer"""
        from src.feature_engineering import FeatureEngineer
        
        # Create feature engineer
        feature_engineer = FeatureEngineer(self.config, self.spark)
        
        # Create features
        df_with_features, feature_names = feature_engineer.create_all_features(df)
        
        return df_with_features