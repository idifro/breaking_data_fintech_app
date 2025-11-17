#!/usr/bin/env python3
"""
Pandas-based Hyperparameter Tuning Script for Stock Price Prediction
Uses Optuna for hyperparameter optimization with GradientBoostingRegressor

CONDA ENVIRONMENT: breaking_data
Usage: 
    conda activate breaking_data
    python scripts/hyperparameter_tuning_pandas.py
"""

import sys
import os
import argparse
import time
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Local imports
from src.utils import Config


class PandasFeatureEngineer:
    """Pandas-based feature engineering class matching Spark pipeline features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_config = config.features
        self.feature_names = []
    
    def create_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create all 25 features matching the Spark pipeline
        
        Args:
            df: Input DataFrame with stock data
            
        Returns:
            Tuple of (DataFrame with features, list of feature names)
        """
        print("🔧 Starting feature engineering...")
        
        # Clean input data first
        df_clean = self._filter_problematic_rows(df)
        
        # Start with cleaned dataframe
        df_features = df_clean.copy()
        
        # Create features step by step per stock
        stock_dataframes = []
        for stock_symbol in df_features['stock_symbol'].unique():
            stock_df = df_features[df_features['stock_symbol'] == stock_symbol].copy()
            stock_df = stock_df.sort_values('Date').reset_index(drop=True)
            
            # Create features for this stock
            stock_df = self._create_price_features(stock_df)
            stock_df = self._create_volume_features(stock_df)
            stock_df = self._create_sentiment_features(stock_df)
            stock_df = self._create_technical_features(stock_df)
            stock_df = self._create_interaction_features(stock_df)
            stock_df = self._create_volatility_features(stock_df)
            
            stock_dataframes.append(stock_df)
        
        # Combine all stocks
        df_features = pd.concat(stock_dataframes, ignore_index=True)
        
        # Create target variable
        df_features = self._create_target(df_features)
        
        # Clean features and get final feature names
        df_features, final_feature_names = self._clean_features(df_features)
        
        print(f"✅ Feature engineering completed. Created {len(final_feature_names)} features")
        
        return df_features, final_feature_names
    
    def _filter_problematic_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out problematic rows to prevent division by zero"""
        
        print("🔍 Filtering problematic rows...")
        initial_count = len(df)
        
        # Remove rows where essential price columns are null or zero
        df_filtered = df.dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume', 'Scaled_sentiment'])
        df_filtered = df_filtered[
            (df_filtered['Close'] > 0) & 
            (df_filtered['High'] > 0) & 
            (df_filtered['Low'] > 0) & 
            (df_filtered['Open'] > 0) & 
            (df_filtered['Volume'] >= 0)
        ]
        
        # Ensure market data consistency
        df_filtered = df_filtered[
            (df_filtered['High'] >= df_filtered['Low']) &
            (df_filtered['Close'] >= df_filtered['Low']) &
            (df_filtered['Close'] <= df_filtered['High'])
        ]
        
        final_count = len(df_filtered)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            removal_rate = (removed_count / initial_count) * 100
            print(f"   Initial rows: {initial_count}")
            print(f"   Final rows: {final_count}")
            print(f"   Removed rows: {removed_count} ({removal_rate:.2f}%)")
        
        return df_filtered.reset_index(drop=True)
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        
        # Price lag features
        for lag in self.feature_config.price_lags:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        
        # Price change features
        df['price_change_1d'] = df['Close'].pct_change(1)
        df['price_change_3d'] = df['Close'].pct_change(3)
        df['price_change_5d'] = df['Close'].pct_change(5)
        
        # Moving averages
        for window in self.feature_config.ma_windows:
            df[f'ma_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
            df[f'close_vs_ma{window}'] = df['Close'] / df[f'ma_{window}']
        
        # Relative position features
        df['close_normalized_vs_lag1'] = df['Close'] / df['close_lag_1']
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        
        # Volume lag features
        for lag in self.feature_config.volume_lags:
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Volume moving averages
        df['volume_ma_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['volume_vs_ma20'] = df['Volume'] / df['volume_ma_20']
        
        # Normalized volume
        df['volume_normalized'] = df['Volume'] / 1000000  # Scale to millions
        
        return df
    
    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-based features"""
        
        # Sentiment lag features
        for lag in self.feature_config.sentiment_lags:
            df[f'sentiment_lag_{lag}'] = df['Scaled_sentiment'].shift(lag)
        
        # Sentiment moving average
        df['sentiment_ma_5'] = df['Scaled_sentiment'].rolling(window=5, min_periods=1).mean()
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        
        # RSI-like indicator
        if self.feature_config.enable_rsi:
            price_change = df['Close'].diff()
            gains = price_change.where(price_change > 0, 0)
            losses = -price_change.where(price_change < 0, 0)
            
            avg_gains = gains.rolling(window=14, min_periods=1).mean()
            avg_losses = losses.rolling(window=14, min_periods=1).mean()
            
            rs = avg_gains / (avg_losses + 1e-8)  # Avoid division by zero
            df['rsi_like'] = 100 - (100 / (1 + rs))
        
        # Price range features
        df['price_range'] = df['High'] - df['Low']
        df['price_range_normalized'] = df['price_range'] / df['Close']
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        
        if self.feature_config.enable_interactions:
            # Price-volume interactions
            df['price_volume_momentum'] = df['price_change_1d'] * df['volume_normalized']
            
            # Price-sentiment interactions
            df['price_sentiment_signal'] = df['price_change_1d'] * df['sentiment_lag_1']
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features"""
        
        if self.feature_config.enable_volatility:
            # Historical volatility
            df['volatility_5d'] = df['price_change_1d'].rolling(window=5, min_periods=1).std()
            df['volatility_10d'] = df['price_change_1d'].rolling(window=10, min_periods=1).std()
        
        return df
    
    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable - next day's close price change"""
        
        # Sort by stock and date to ensure proper target creation
        df = df.sort_values(['stock_symbol', 'Date']).reset_index(drop=True)
        
        # Create target as next day's price change per stock
        df['target'] = df.groupby('stock_symbol')['Close'].pct_change(-1)  # Forward looking
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Clean features and select final feature set"""
        
        print("🧹 Cleaning features...")
        
        # Define all potential feature names
        all_feature_names = [
            # Price features
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
            'price_change_1d', 'price_change_3d', 'price_change_5d',
            'close_vs_ma5', 'close_vs_ma10', 'close_vs_ma20',
            'close_normalized_vs_lag1',
            
            # Volume features
            'volume_lag_1', 'volume_vs_ma20', 'volume_ma_20', 'volume_normalized',
            
            # Sentiment features
            'sentiment_lag_1', 'sentiment_lag_3', 'sentiment_ma_5',
            
            # Technical features
            'rsi_like', 'price_range', 'price_range_normalized',
            
            # Interaction features
            'price_volume_momentum', 'price_sentiment_signal',
            
            # Volatility features
            'volatility_5d', 'volatility_10d'
        ]
        
        # Select features that exist in the dataframe
        existing_features = [f for f in all_feature_names if f in df.columns]
        
        # Limit to max_features if specified
        if hasattr(self.feature_config, 'max_features') and self.feature_config.max_features:
            existing_features = existing_features[:self.feature_config.max_features]
        
        # Remove rows with any null values in features or target
        essential_columns = existing_features + ['target', 'stock_symbol', 'Date', 'Close']
        df_clean = df[essential_columns].dropna()
        
        print(f"✅ Final feature count: {len(existing_features)}")
        print(f"✅ Clean samples: {len(df_clean)}")
        
        return df_clean, existing_features


class PandasHyperparameterTuner:
    """Hyperparameter tuning using pandas and scikit-learn"""
    
    def __init__(self, config: Config, selected_stocks: List[str]):
        self.config = config
        self.selected_stocks = selected_stocks
        self.feature_engineer = PandasFeatureEngineer(config)
        self.scaler = StandardScaler()
        
        # Results storage
        self.best_params = None
        self.best_scores = {}
        self.tuning_history = []
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV files"""
        
        print(f"📁 Loading data for stocks: {self.selected_stocks}")
        
        all_data = []
        csv_dir = Path("data_csv")
        
        for stock in self.selected_stocks:
            csv_path = csv_dir / f"{stock}.csv"
            
            if not csv_path.exists():
                print(f"⚠️  Warning: {csv_path} not found, skipping {stock}")
                continue
            
            # Load CSV
            df = pd.read_csv(csv_path)
            df['stock_symbol'] = stock
            
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Ensure proper data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Scaled_sentiment']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            all_data.append(df)
            print(f"   ✅ {stock}: {len(df)} rows")
        
        if not all_data:
            raise FileNotFoundError("No valid stock data files found!")
        
        # Combine all stock data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['stock_symbol', 'Date']).reset_index(drop=True)
        
        print(f"✅ Combined data: {len(combined_df)} rows from {len(all_data)} stocks")
        
        return combined_df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Prepare data with feature engineering and train/test split"""
        
        print("🔧 Preparing data with feature engineering...")
        
        # Create features
        df_features, feature_names = self.feature_engineer.create_all_features(df)
        
        # Create time-based train/test split (same as Spark pipeline)
        train_dfs = []
        test_dfs = []
        
        print(f"📊 Creating train/test split ({self.config.data.train_split:.0%} train)...")
        
        for stock in df_features['stock_symbol'].unique():
            stock_df = df_features[df_features['stock_symbol'] == stock].copy()
            stock_df = stock_df.sort_values('Date').reset_index(drop=True)
            
            # Remove last row (target is NaN due to forward-looking target)
            stock_df = stock_df[:-1]
            
            # Time-based split
            split_idx = int(len(stock_df) * self.config.data.train_split)
            
            train_stock = stock_df[:split_idx].copy()
            test_stock = stock_df[split_idx:].copy()
            
            train_dfs.append(train_stock)
            test_dfs.append(test_stock)
            
            print(f"   {stock}: {len(train_stock)} train, {len(test_stock)} test")
        
        # Combine splits
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print(f"✅ Data prepared:")
        print(f"   Training samples: {len(train_df)}")
        print(f"   Test samples: {len(test_df)}")
        print(f"   Features: {len(feature_names)}")
        
        return train_df, test_df, feature_names
    
    def tune_hyperparameters(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           feature_names: List[str], n_trials: int = 50) -> Dict[str, Any]:
        """Perform hyperparameter tuning with Optuna"""
        
        print(f"🔍 Starting hyperparameter tuning with {n_trials} trials...")
        
        # Prepare feature matrices
        X_train = train_df[feature_names].values
        y_train = train_df['target'].values
        X_test = test_df[feature_names].values
        y_test = test_df['target'].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   Training samples: {X_train_scaled.shape}")
        print(f"   Test samples: {X_test_scaled.shape}")
        
        def objective(trial):
            try:
                # Define hyperparameter search space (matching Spark config)
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators', 
                                                           self.config.model.hyperparameter_ranges['maxIter']),
                    'max_depth': trial.suggest_categorical('max_depth', 
                                                        self.config.model.hyperparameter_ranges['maxDepth']),
                    'learning_rate': trial.suggest_categorical('learning_rate', 
                                                            self.config.model.hyperparameter_ranges['stepSize']),
                    'subsample': trial.suggest_categorical('subsample', 
                                                        self.config.model.hyperparameter_ranges['subsamplingRate']),
                    'max_features': trial.suggest_categorical('max_features', 
                                                           self.config.model.hyperparameter_ranges['featureSubsetStrategy']),
                    'random_state': 42
                }
                
                # Convert max_features to scikit-learn format
                if params['max_features'] == 'auto':
                    params['max_features'] = 1.0
                elif params['max_features'] == 'log2':
                    params['max_features'] = 'log2'
                elif params['max_features'] == 'sqrt':
                    params['max_features'] = 'sqrt'
                
                # Train model
                model = GradientBoostingRegressor(**params)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred_test)
                mse = mean_squared_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
                
                # Store trial results
                trial_result = {
                    'trial_number': trial.number,
                    'params': params.copy(),
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'timestamp': datetime.now().isoformat()
                }
                self.tuning_history.append(trial_result)
                
                # Primary optimization metric (MAE)
                return mae
                
            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(
            direction='minimize',  # Minimize MAE
            study_name='stock_prediction_gbt_tuning',
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials)
        total_time = time.time() - start_time
        
        # Get best results
        self.best_params = study.best_params.copy()
        
        # Calculate final metrics with best params
        best_model = GradientBoostingRegressor(**self.best_params)
        best_model.fit(X_train_scaled, y_train)
        
        y_pred_train = best_model.predict(X_train_scaled)
        y_pred_test = best_model.predict(X_test_scaled)
        
        self.best_scores = {
            'train_metrics': {
                'mae': mean_absolute_error(y_train, y_pred_train),
                'mse': mean_squared_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test_metrics': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            },
            'best_trial_number': study.best_trial.number,
            'total_trials': n_trials,
            'optimization_time_seconds': total_time
        }
        
        # Print results
        print(f"\n🎉 Hyperparameter tuning completed!")
        print(f"   ⏱️  Total time: {total_time:.2f} seconds")
        print(f"   🏆 Best trial: {study.best_trial.number}")
        print(f"   📊 Best MAE: {study.best_value:.6f}")
        
        print(f"\n🏆 Best parameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        print(f"\n📊 Best model performance:")
        print(f"   Train - MAE: {self.best_scores['train_metrics']['mae']:.6f}, "
              f"MSE: {self.best_scores['train_metrics']['mse']:.6f}, "
              f"R²: {self.best_scores['train_metrics']['r2']:.4f}")
        print(f"   Test  - MAE: {self.best_scores['test_metrics']['mae']:.6f}, "
              f"MSE: {self.best_scores['test_metrics']['mse']:.6f}, "
              f"R²: {self.best_scores['test_metrics']['r2']:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_scores': self.best_scores,
            'optimization_history': self.tuning_history,
            'study_summary': {
                'best_value': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials)
            }
        }
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON and update config"""
        
        print("💾 Saving hyperparameter tuning results...")
        
        # Create results directory
        results_dir = Path("results/hyperparameter_tuning")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to JSON
        json_path = results_dir / f"tuning_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   ✅ Detailed results: {json_path}")
        
        # Create config_pandas.yaml with best parameters
        config_pandas = {
            'data': {
                'selected_stocks': self.selected_stocks,
                'train_split': self.config.data.train_split,
                'sequence_length': self.config.data.sequence_length
            },
            'model': {
                'algorithm': 'GradientBoostingRegressor',
                'best_params': self.best_params,
                'performance': self.best_scores,
                'tuning_info': {
                    'tuning_date': datetime.now().isoformat(),
                    'n_trials': results['study_summary']['n_trials'],
                    'best_trial': results['study_summary']['best_trial'],
                    'optimization_time_seconds': self.best_scores['optimization_time_seconds']
                }
            },
            'features': {
                'max_features': self.feature_engineer.feature_config.max_features if hasattr(self.feature_engineer.feature_config, 'max_features') else 25,
                'price_lags': self.feature_engineer.feature_config.price_lags,
                'ma_windows': self.feature_engineer.feature_config.ma_windows,
                'volume_lags': self.feature_engineer.feature_config.volume_lags,
                'sentiment_lags': self.feature_engineer.feature_config.sentiment_lags,
                'enable_rsi': self.feature_engineer.feature_config.enable_rsi,
                'enable_volatility': self.feature_engineer.feature_config.enable_volatility,
                'enable_interactions': self.feature_engineer.feature_config.enable_interactions
            }
        }
        
        # Save config_pandas.yaml
        config_path = Path("config/config_pandas.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_pandas, f, default_flow_style=False, indent=2)
        
        print(f"   ✅ Best parameters config: {config_path}")
        
        # Save best parameters separately for easy access
        best_params_path = results_dir / f"best_params_{timestamp}.json"
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'performance': self.best_scores,
                'stocks': self.selected_stocks,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print(f"   ✅ Best parameters only: {best_params_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for stock prediction with pandas')
    
    parser.add_argument(
        '--stocks', 
        type=str, 
        default='AAPL,GOOG,TSLA,BABA,NVDA',
        help='Comma-separated list of stock symbols (default: AAPL,GOOG,TSLA,BABA,NVDA)'
    )
    parser.add_argument(
        '--trials', 
        type=int, 
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    return parser.parse_args()


def main():
    """Main hyperparameter tuning pipeline"""
    
    print("🚀 Starting Pandas-based Hyperparameter Tuning Pipeline")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        config = Config()
        
        # Parse stock selection
        selected_stocks = [s.strip().upper() for s in args.stocks.split(',')]
        print(f"🎯 Selected stocks: {selected_stocks}")
        print(f"🔬 Optimization trials: {args.trials}")
        
        # Initialize tuner
        tuner = PandasHyperparameterTuner(config, selected_stocks)
        
        # Load data
        print("\n📁 Step 1: Loading data...")
        df = tuner.load_data()
        
        # Prepare data with feature engineering
        print("\n🔧 Step 2: Feature engineering and data preparation...")
        train_df, test_df, feature_names = tuner.prepare_data(df)
        
        # Perform hyperparameter tuning
        print("\n🔍 Step 3: Hyperparameter optimization...")
        results = tuner.tune_hyperparameters(train_df, test_df, feature_names, args.trials)
        
        # Save results
        print("\n💾 Step 4: Saving results...")
        tuner.save_results(results)
        
        print("\n🎉 Hyperparameter tuning pipeline completed successfully!")
        print(f"   📊 Best MAE: {results['best_scores']['test_metrics']['mae']:.6f}")
        print(f"   📊 Best R²: {results['best_scores']['test_metrics']['r2']:.4f}")
        print(f"   🏆 Best trial: {results['study_summary']['best_trial']} / {results['study_summary']['n_trials']}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Hyperparameter tuning pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()