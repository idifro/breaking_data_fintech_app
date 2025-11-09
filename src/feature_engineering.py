"""
Feature Engineering Module for Stock Price Prediction
Implements optimal features for GBT model following CNN-style approach
"""

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import VectorUDT
from loguru import logger

from src.utils import Config, FeatureConfig


class FeatureEngineer:
    """Feature engineering class for stock price prediction"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.feature_config = config.features
        self.spark = spark
        self.feature_names = []
    
    def create_all_features(self, df: DataFrame) -> Tuple[DataFrame, List[str]]:
        """
        Create all features for stock price prediction
        
        Args:
            df: Input DataFrame with stock data
            
        Returns:
            Tuple of (DataFrame with features, list of feature names)
        """
        logger.info("Starting feature engineering...")
        
        # Define window for each stock
        window_spec = Window.partitionBy("stock_symbol").orderBy("Date")
        
        # Start with original dataframe
        df_features = df
        
        # Create features step by step
        df_features = self._create_price_features(df_features, window_spec)
        df_features = self._create_volume_features(df_features, window_spec)
        df_features = self._create_sentiment_features(df_features, window_spec)
        df_features = self._create_technical_features(df_features, window_spec)
        df_features = self._create_interaction_features(df_features, window_spec)
        df_features = self._create_volatility_features(df_features, window_spec)
        
        # Create target variable
        df_features = self._create_target(df_features, window_spec)
        
        # Clean features (remove highly correlated ones)
        df_features, final_feature_names = self._clean_features(df_features)
        
        logger.info(f"Feature engineering completed. Created {len(final_feature_names)} features")
        
        return df_features, final_feature_names
    
    def _create_price_features(self, df: DataFrame, window_spec: Window) -> DataFrame:
        """Create price-based features"""
        logger.info("Creating price features...")
        
        # Price lag features
        for lag_days in self.feature_config.price_lags:
            col_name = f"close_lag_{lag_days}"
            df = df.withColumn(col_name, lag("Close", lag_days).over(window_spec))
            self.feature_names.append(col_name)
        
        # Price changes and momentum
        df = df.withColumn("price_change_1d", 
                          (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec))
        df = df.withColumn("price_change_3d", 
                          (col("Close") - lag("Close", 3).over(window_spec)) / lag("Close", 3).over(window_spec))
        df = df.withColumn("price_change_5d", 
                          (col("Close") - lag("Close", 5).over(window_spec)) / lag("Close", 5).over(window_spec))
        
        self.feature_names.extend(["price_change_1d", "price_change_3d", "price_change_5d"])
        
        # Moving averages
        for window_size in self.feature_config.ma_windows:
            ma_col = f"ma_{window_size}"
            df = df.withColumn(ma_col, avg("Close").over(window_spec.rowsBetween(-window_size+1, 0)))
            
            # Price relative to moving average
            ratio_col = f"close_vs_ma{window_size}"
            df = df.withColumn(ratio_col, col("Close") / col(ma_col) - 1)
            
            self.feature_names.extend([ma_col, ratio_col])
        
        # High-Low spread
        df = df.withColumn("high_low_spread", (col("High") - col("Low")) / col("Close"))
        self.feature_names.append("high_low_spread")
        
        # Normalized price features (CNN-style)
        df = df.withColumn("close_normalized_vs_lag1", 
                          col("Close") / lag("Close", 1).over(window_spec) - 1)
        df = df.withColumn("close_normalized_vs_lag5", 
                          col("Close") / lag("Close", 5).over(window_spec) - 1)
        
        self.feature_names.extend(["close_normalized_vs_lag1", "close_normalized_vs_lag5"])
        
        return df
    
    def _create_volume_features(self, df: DataFrame, window_spec: Window) -> DataFrame:
        """Create volume-based features"""
        logger.info("Creating volume features...")
        
        # Volume lag features
        for lag_val in self.feature_config.volume_lags:
            col_name = f"volume_lag_{lag_val}"
            df = df.withColumn(col_name, lag("Volume", lag_val).over(window_spec))
            self.feature_names.append(col_name)
        
        # Volume moving averages
        df = df.withColumn("volume_ma_5", avg("Volume").over(window_spec.rowsBetween(-4, 0)))
        df = df.withColumn("volume_ma_20", avg("Volume").over(window_spec.rowsBetween(-19, 0)))
        
        # Volume changes and ratios
        df = df.withColumn("volume_change_1d", 
                          (col("Volume") - lag("Volume", 1).over(window_spec)) / lag("Volume", 1).over(window_spec))
        df = df.withColumn("volume_vs_ma5", col("Volume") / col("volume_ma_5") - 1)
        df = df.withColumn("volume_vs_ma20", col("Volume") / col("volume_ma_20") - 1)
        
        # Volume normalization (CNN-style)
        df = df.withColumn("volume_normalized_vs_lag1", 
                          col("Volume") / lag("Volume", 1).over(window_spec) - 1)
        
        self.feature_names.extend([
            "volume_ma_5", "volume_ma_20", "volume_change_1d", 
            "volume_vs_ma5", "volume_vs_ma20", "volume_normalized_vs_lag1"
        ])
        
        return df
    
    def _create_sentiment_features(self, df: DataFrame, window_spec: Window) -> DataFrame:
        """Create sentiment-based features"""
        logger.info("Creating sentiment features...")
        
        # Sentiment lag features
        for lag_val in self.feature_config.sentiment_lags:
            col_name = f"sentiment_lag_{lag_val}"
            df = df.withColumn(col_name, lag("Scaled_sentiment", lag_val).over(window_spec))
            self.feature_names.append(col_name)
        
        # Sentiment moving averages
        df = df.withColumn("sentiment_ma_5", avg("Scaled_sentiment").over(window_spec.rowsBetween(-4, 0)))
        df = df.withColumn("sentiment_ma_10", avg("Scaled_sentiment").over(window_spec.rowsBetween(-9, 0)))
        df = df.withColumn("sentiment_ma_20", avg("Scaled_sentiment").over(window_spec.rowsBetween(-19, 0)))
        
        # Sentiment changes and momentum
        df = df.withColumn("sentiment_change_1d", 
                          col("Scaled_sentiment") - lag("Scaled_sentiment", 1).over(window_spec))
        df = df.withColumn("sentiment_change_3d", 
                          col("Scaled_sentiment") - lag("Scaled_sentiment", 3).over(window_spec))
        
        # Sentiment volatility
        df = df.withColumn("sentiment_volatility_5d", 
                          stddev("Scaled_sentiment").over(window_spec.rowsBetween(-4, 0)))
        
        self.feature_names.extend([
            "sentiment_ma_5", "sentiment_ma_10", "sentiment_ma_20",
            "sentiment_change_1d", "sentiment_change_3d", "sentiment_volatility_5d"
        ])
        
        return df
    
    def _create_technical_features(self, df: DataFrame, window_spec: Window) -> DataFrame:
        """Create technical analysis features"""
        logger.info("Creating technical features...")
        
        if self.feature_config.enable_rsi:
            # Simplified RSI calculation
            df = df.withColumn("price_gain", 
                              when(col("price_change_1d") > 0, col("price_change_1d")).otherwise(0))
            df = df.withColumn("price_loss", 
                              when(col("price_change_1d") < 0, abs(col("price_change_1d"))).otherwise(0))
            
            # Average gains and losses
            df = df.withColumn("avg_gain_14", avg("price_gain").over(window_spec.rowsBetween(-13, 0)))
            df = df.withColumn("avg_loss_14", avg("price_loss").over(window_spec.rowsBetween(-13, 0)))
            
            # RSI-like momentum indicator
            df = df.withColumn("momentum_ratio", 
                              col("avg_gain_14") / (col("avg_loss_14") + 0.001))
            
            self.feature_names.extend(["momentum_ratio"])
        
        # Price position within recent range
        df = df.withColumn("high_5d", max("High").over(window_spec.rowsBetween(-4, 0)))
        df = df.withColumn("low_5d", min("Low").over(window_spec.rowsBetween(-4, 0)))
        df = df.withColumn("price_position_5d", 
                          (col("Close") - col("low_5d")) / (col("high_5d") - col("low_5d") + 0.001))
        
        self.feature_names.extend(["price_position_5d"])
        
        return df
    
    def _create_interaction_features(self, df: DataFrame, window_spec: Window) -> DataFrame:
        """Create interaction features between price, volume, and sentiment"""
        logger.info("Creating interaction features...")
        
        if self.feature_config.enable_interactions:
            # Price-Volume interaction (key financial indicator)
            df = df.withColumn("price_volume_interaction", 
                              col("price_change_1d") * col("volume_change_1d"))
            
            # Price-Sentiment interaction
            df = df.withColumn("price_sentiment_interaction", 
                              col("price_change_1d") * col("sentiment_change_1d"))
            
            # Volume-Sentiment interaction
            df = df.withColumn("volume_sentiment_interaction", 
                              col("volume_change_1d") * col("sentiment_change_1d"))
            
            # Three-way interaction
            df = df.withColumn("price_volume_sentiment_interaction", 
                              col("price_change_1d") * col("volume_change_1d") * col("sentiment_change_1d"))
            
            self.feature_names.extend([
                "price_volume_interaction", "price_sentiment_interaction",
                "volume_sentiment_interaction", "price_volume_sentiment_interaction"
            ])
        
        return df
    
    def _create_volatility_features(self, df: DataFrame, window_spec: Window) -> DataFrame:
        """Create volatility and risk features"""
        logger.info("Creating volatility features...")
        
        if self.feature_config.enable_volatility:
            # Price volatility
            df = df.withColumn("price_volatility_5d", 
                              stddev("price_change_1d").over(window_spec.rowsBetween(-4, 0)))
            df = df.withColumn("price_volatility_10d", 
                              stddev("price_change_1d").over(window_spec.rowsBetween(-9, 0)))
            
            # Volume volatility
            df = df.withColumn("volume_volatility_5d", 
                              stddev("volume_change_1d").over(window_spec.rowsBetween(-4, 0)))
            
            # Volatility ratios
            df = df.withColumn("volatility_ratio_5_10", 
                              col("price_volatility_5d") / (col("price_volatility_10d") + 0.001))
            
            # Risk-adjusted return
            df = df.withColumn("risk_adjusted_return", 
                              col("price_change_1d") / (col("price_volatility_5d") + 0.001))
            
            self.feature_names.extend([
                "price_volatility_5d", "price_volatility_10d", "volume_volatility_5d",
                "volatility_ratio_5_10", "risk_adjusted_return"
            ])
        
        return df
    
    def _create_target(self, df: DataFrame, window_spec: Window) -> DataFrame:
        """Create target variable (next day's return)"""
        logger.info("Creating target variable...")
        
        # Target: next day's return (like CNN approach)
        df = df.withColumn("target", 
                          (lead("Close", 1).over(window_spec) / col("Close") - 1))
        
        return df
    
    def _clean_features(self, df: DataFrame) -> Tuple[DataFrame, List[str]]:
        """Clean features by removing highly correlated ones and null values"""
        logger.info("Cleaning features...")
        
        # Remove rows with null target
        df_clean = df.filter(col("target").isNotNull())
        
        # Remove rows with null features
        for feature_name in self.feature_names:
            df_clean = df_clean.filter(col(feature_name).isNotNull())
        
        # TODO: Add correlation analysis to remove highly correlated features
        # For now, keep all features
        final_feature_names = self.feature_names.copy()
        
        # Limit number of features if specified
        if self.feature_config.max_features > 0:
            final_feature_names = final_feature_names[:self.feature_config.max_features]
        
        logger.info(f"Final feature count: {len(final_feature_names)}")
        
        return df_clean, final_feature_names
    
    def create_feature_vector(self, df: DataFrame, feature_names: List[str]) -> Tuple[DataFrame, object]:
        """Create feature vector and scaler for ML model"""
        logger.info("Creating feature vector...")
        
        # Create feature vector
        assembler = VectorAssembler(inputCols=feature_names, outputCol="features_raw")
        df_features = assembler.transform(df)
        
        # Split data to fit scaler only on training data
        train_df = df_features.filter(col("split") == "train")
        
        # Create and fit scaler
        scaler = StandardScaler(
            inputCol="features_raw", 
            outputCol="features", 
            withStd=True, 
            withMean=True
        )
        scaler_model = scaler.fit(train_df)
        
        # Transform all data
        df_scaled = scaler_model.transform(df_features)
        
        logger.info("Feature vector creation completed")
        
        return df_scaled, scaler_model
    
    def analyze_feature_correlation(self, df: DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Analyze feature correlation and return correlation matrix"""
        logger.info("Analyzing feature correlations...")
        
        # Convert to pandas for correlation analysis
        feature_data = df.select(feature_names).toPandas()
        
        # Calculate correlation matrix
        correlation_matrix = feature_data.corr()
        
        return correlation_matrix
    
    def get_feature_importance_names(self) -> List[str]:
        """Get feature names for importance analysis"""
        return self.feature_names