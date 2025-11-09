"""
Model Training Module for Stock Price Prediction
Implements GBT training with hyperparameter tuning and MLflow tracking
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import time
import mlflow
import mlflow.spark
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hyperparameter tuning imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.utils import Config, MLflowManager, PathManager
from src.visualization import Visualizer


class ModelTrainer:
    """Model training class with hyperparameter tuning and evaluation"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.mlflow_manager = MLflowManager(config)
        self.path_manager = PathManager()
        self.visualizer = Visualizer(config)
        
    def train_model(self, train_df: DataFrame, test_df: DataFrame, feature_names: List[str]) -> Tuple[object, Dict[str, Any]]:
        """
        Train GBT model with optimal parameters
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature_names: List of feature names
            
        Returns:
            Tuple of (trained_model, evaluation_results)
        """
        
        run_name = f"{self.config.mlflow_config['run_name_prefix']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            print("🚀 Starting model training...")
            
            # Log configuration
            self.mlflow_manager.log_config(self.config)
            self.mlflow_manager.log_feature_config(feature_names)
            
            # Get best hyperparameters
            if self.config.hyperparameter_tuning['enabled']:
                print("🔍 Hyperparameter tuning is ENABLED")
                best_params = self._tune_hyperparameters(train_df, test_df)
            else:
                print("⚙️ Hyperparameter tuning is DISABLED - using default parameters")
                best_params = self.config.model.default_params
                print(f"✅ Using default parameters: {best_params}")
            
            # Train final model with best parameters
            model = self._train_final_model(train_df, best_params)
            
            # Evaluate model
            evaluation_results = self._evaluate_model(model, train_df, test_df, feature_names)
            
            # Log metrics
            self._log_metrics(evaluation_results)
            
            # Register model
            self.mlflow_manager.register_model(model)
            
            # Save evaluation results
            self._save_evaluation_results(evaluation_results)
            
            # Create visualizations
            self._create_visualizations(model, train_df, test_df, feature_names, evaluation_results)
            
            print("✅ Model training completed successfully!")
            
            return model, evaluation_results
    
    def _tune_hyperparameters(self, train_df: DataFrame, test_df: DataFrame) -> Dict[str, Any]:
        """Tune hyperparameters using either Optuna or Spark CrossValidator based on config"""
        
        tuning_method = self.config.hyperparameter_tuning['method']
        print(f"🔍 Starting hyperparameter tuning with {tuning_method.upper()}...")
        
        if tuning_method == 'optuna' and OPTUNA_AVAILABLE:
            return self._tune_with_optuna(train_df, test_df)
        elif tuning_method == 'spark_cv':
            return self._tune_with_spark_cv(train_df)
        else:
            if tuning_method == 'optuna' and not OPTUNA_AVAILABLE:
                print("⚠️ Optuna not available, falling back to Spark CrossValidator")
            else:
                print(f"⚠️ Unknown tuning method '{tuning_method}', falling back to Spark CrossValidator")
            return self._tune_with_spark_cv(train_df)
    
    def _tune_with_optuna(self, train_df: DataFrame, test_df: DataFrame) -> Dict[str, Any]:
        """Hyperparameter tuning using Optuna (optimized version)"""
        
        print("Using Optuna for hyperparameter optimization...")
        
        # Cache the DataFrames to avoid re-computation
        train_df.cache()
        test_df.cache()
        
        # Force evaluation to cache data in memory
        print(f"Training samples: {train_df.count()}")
        print(f"Test samples: {test_df.count()}")

        def objective(trial):
            try:
                # Define hyperparameter search space
                params = {
                    'maxIter': trial.suggest_categorical('maxIter', 
                                                       self.config.model.hyperparameter_ranges['maxIter']),
                    'maxDepth': trial.suggest_categorical('maxDepth', 
                                                        self.config.model.hyperparameter_ranges['maxDepth']),
                    'stepSize': trial.suggest_categorical('stepSize', 
                                                        self.config.model.hyperparameter_ranges['stepSize']),
                    'subsamplingRate': trial.suggest_categorical('subsamplingRate', 
                                                               self.config.model.hyperparameter_ranges['subsamplingRate']),
                    'featureSubsetStrategy': trial.suggest_categorical('featureSubsetStrategy', 
                                                                     self.config.model.hyperparameter_ranges['featureSubsetStrategy']),
                    'seed': 42
                }
                
                # Train model with current parameters
                gbt = GBTRegressor(
                    featuresCol="features",
                    labelCol="target",
                    **params
                )
                
                # Use a smaller subset for validation during tuning to reduce memory usage
                train_sample = train_df.sample(0.3, seed=42)  # Use 30% of training data
                
                model = gbt.fit(train_sample)
                predictions = model.transform(test_df)
                
                # Calculate evaluation metric
                evaluator = RegressionEvaluator(
                    labelCol="target", 
                    predictionCol="prediction", 
                    metricName=self.config.hyperparameter_tuning['metric']
                )
                
                metric_value = evaluator.evaluate(predictions)
                
                # Clean up intermediate results
                predictions.unpersist()
                
                return metric_value
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')  # Return worst possible score for failed trials
        
        # Create study with pruning for early stopping
        study = optuna.create_study(
            direction=self.config.hyperparameter_tuning['direction'],
            study_name=self.config.hyperparameter_tuning['optuna']['study_name'],
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize with configured number of trials
        n_trials = self.config.hyperparameter_tuning['n_trials']
        study.optimize(objective, n_trials=n_trials)
        
        # Clean up cached DataFrames
        train_df.unpersist()
        test_df.unpersist()

        # Log best parameters
        best_params = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_{self.config.hyperparameter_tuning['metric']}", study.best_value)
        
        print(f"✅ Best parameters found: {best_params}")
        print(f"✅ Best {self.config.hyperparameter_tuning['metric']}: {study.best_value:.6f}")
        
        return best_params
    
    def _tune_with_spark_cv(self, train_df: DataFrame) -> Dict[str, Any]:
        """Hyperparameter tuning using Spark CrossValidator"""
        
        print("Using Spark CrossValidator for hyperparameter optimization...")
        
        # Create base estimator
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="target",
            seed=42
        )
        
        # Create parameter grid
        param_grid = ParamGridBuilder()
        
        for param_name, param_values in self.config.model.hyperparameter_ranges.items():
            if param_name != 'seed':
                param_attr = getattr(gbt, param_name)
                param_grid = param_grid.addGrid(param_attr, param_values[:3])  # Limit for speed
        
        param_grid = param_grid.build()
        
        # Create evaluator
        evaluator = RegressionEvaluator(
            labelCol="target",
            predictionCol="prediction",
            metricName=self.config.hyperparameter_tuning['metric']
        )
        
        # Create cross validator
        cv = CrossValidator(
            estimator=gbt,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=self.config.hyperparameter_tuning['cv_folds']
        )
        
        # Fit cross validator
        cv_model = cv.fit(train_df)
        
        # Get best parameters
        best_model = cv_model.bestModel
        best_params = {
            'maxIter': best_model.getMaxIter(),
            'maxDepth': best_model.getMaxDepth(),
            'stepSize': best_model.getStepSize(),
            'subsamplingRate': best_model.getSubsamplingRate(),
            'featureSubsetStrategy': best_model.getFeatureSubsetStrategy()
        }
        
        # Log best parameters
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        
        print(f"✅ Best parameters found: {best_params}")
        
        return best_params
    
    def _train_final_model(self, train_df: DataFrame, params: Dict[str, Any]) -> object:
        """Train final model with best parameters"""
        
        print("🔧 Training final model with parameters:")
        for param, value in params.items():
            print(f"   {param}: {value}")
        
        # Create GBT with best parameters
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="target",
            **params
        )
        
        # Train model
        start_time = time.time()
        model = gbt.fit(train_df)
        training_time = time.time() - start_time
        
        # Log training time
        mlflow.log_metric("training_time_seconds", training_time)
        
        print(f"✅ Model training completed in {training_time:.2f} seconds")
        
        return model
    
    def _evaluate_model(self, model: object, train_df: DataFrame, test_df: DataFrame, 
                       feature_names: List[str]) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        print("📊 Evaluating model performance...")
        
        # Make predictions
        train_predictions = model.transform(train_df)
        test_predictions = model.transform(test_df)
        
        # Collect results
        train_results = train_predictions.select("stock_symbol", "target", "prediction", "Date").collect()
        test_results = test_predictions.select("stock_symbol", "target", "prediction", "Date").collect()
        
        # Calculate metrics
        evaluation_results = {
            'train_metrics': self._calculate_metrics(train_results, "Training"),
            'test_metrics': self._calculate_metrics(test_results, "Test"),
            'feature_importance': self._get_feature_importance(model, feature_names),
            'per_stock_metrics': self._calculate_per_stock_metrics(test_results),
            'predictions': {
                'train': train_results,
                'test': test_results
            }
        }
        
        return evaluation_results
    
    def _calculate_metrics(self, results: List, data_type: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        true_values = [row.target for row in results]
        predicted_values = [row.prediction for row in results]
        
        metrics = {
            'mae': mean_absolute_error(true_values, predicted_values),
            'mse': mean_squared_error(true_values, predicted_values),
            'rmse': np.sqrt(mean_squared_error(true_values, predicted_values)),
            'r2': r2_score(true_values, predicted_values),
            'sample_count': len(results)
        }
        
        print(f"\n{data_type} Metrics:")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  Samples: {metrics['sample_count']}")
        
        return metrics
    
    def _calculate_per_stock_metrics(self, results: List) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per stock"""
        
        stock_data = {}
        for row in results:
            stock = row.stock_symbol
            if stock not in stock_data:
                stock_data[stock] = {'true': [], 'pred': []}
            stock_data[stock]['true'].append(row.target)
            stock_data[stock]['pred'].append(row.prediction)
        
        stock_metrics = {}
        print("\nPer-Stock Test Metrics:")
        
        for stock, data in stock_data.items():
            metrics = {
                'mae': mean_absolute_error(data['true'], data['pred']),
                'mse': mean_squared_error(data['true'], data['pred']),
                'r2': r2_score(data['true'], data['pred']),
                'sample_count': len(data['true'])
            }
            stock_metrics[stock] = metrics
            
            print(f"  {stock}: MAE={metrics['mae']:.6f}, MSE={metrics['mse']:.6f}, "
                  f"R²={metrics['r2']:.4f}, N={metrics['sample_count']}")
        
        return stock_metrics
    
    def _get_feature_importance(self, model: object, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from trained model"""
        
        feature_importance = model.featureImportances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance.toArray()
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def _log_metrics(self, evaluation_results: Dict[str, Any]) -> None:
        """Log metrics to MLflow"""
        
        if not self.config.mlflow_config['log_metrics']:
            return
        
        # Log overall metrics
        for metric_name, metric_value in evaluation_results['train_metrics'].items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)
        
        for metric_name, metric_value in evaluation_results['test_metrics'].items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Log per-stock metrics
        for stock, metrics in evaluation_results['per_stock_metrics'].items():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{stock}_{metric_name}", metric_value)
        
        # Log feature importance
        for idx, row in evaluation_results['feature_importance'].head(10).iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
    
    def _save_evaluation_results(self, evaluation_results: Dict[str, Any]) -> None:
        """Save evaluation results to CSV files"""
        
        print("💾 Saving evaluation results...")
        
        # Create results directory
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.path_manager.results_dir / "training" / f"run_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature importance
        importance_path = results_dir / "feature_importance.csv"
        evaluation_results['feature_importance'].to_csv(importance_path, index=False)
        
        # Save per-stock metrics
        stock_metrics_df = pd.DataFrame(evaluation_results['per_stock_metrics']).T
        stock_metrics_path = results_dir / "per_stock_metrics.csv"
        stock_metrics_df.to_csv(stock_metrics_path)
        
        # Save predictions
        test_pred_df = pd.DataFrame([
            {
                'stock_symbol': row.stock_symbol,
                'date': row.Date,
                'true_value': row.target,
                'predicted_value': row.prediction
            }
            for row in evaluation_results['predictions']['test']
        ])
        
        predictions_path = results_dir / "test_predictions.csv"
        test_pred_df.to_csv(predictions_path, index=False)
        
        print(f"✅ Results saved to: {results_dir}")
    
    def _create_visualizations(self, model: object, train_df: DataFrame, test_df: DataFrame,
                              feature_names: List[str], evaluation_results: Dict[str, Any]) -> None:
        """Create and save visualizations"""
        
        print("📊 Creating visualizations...")
        
        # Create visualizations
        self.visualizer.plot_feature_importance(evaluation_results['feature_importance'])
        self.visualizer.plot_predictions_vs_actual(evaluation_results['predictions']['test'])
        self.visualizer.plot_per_stock_performance(evaluation_results['per_stock_metrics'])
        self.visualizer.plot_residuals(evaluation_results['predictions']['test'])
        
        print("✅ Visualizations created successfully!")
        
        # Log plots to MLflow
        if self.config.mlflow_config['log_artifacts']:
            plots_dir = self.path_manager.plots_dir
            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file))