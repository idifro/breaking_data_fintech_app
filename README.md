# 🚀 Spark ML Stock Price Prediction Pipeline with Scalability Monitoring

A comprehensive, production-ready stock price prediction pipeline using Apache Spark MLlib, Delta Lake, and MLflow with integrated scalability monitoring and performance tracking. This implementation provides enterprise-level monitoring capabilities for Data Engineering at Scale projects.

## 🏗️ Architecture

```
spark_ml_pipeline/
├── src/                          # Core source modules
│   ├── utils.py                  # Configuration management and data models
│   ├── data_processing.py        # Delta Lake integration and preprocessing
│   ├── feature_engineering.py   # 20+ optimized features for GBT
│   ├── model_training.py         # GBT training with hyperparameter tuning
│   ├── visualization.py          # Comprehensive plotting and analysis
│   ├── scalability_monitor.py   # Real-time performance monitoring
│   └── monitoring_config_manager.py # MLflow monitoring integration
├── scripts/                      # Execution scripts
│   ├── train_model.py           # Basic training pipeline
│   ├── train_model_with_monitoring.py # Enhanced training with monitoring
│   ├── train_model_with_modes.py       # Multi-mode training (GBT, RF, Hybrid)
│   ├── inference.py             # Basic inference pipeline
│   ├── inference_with_monitoring.py   # Enhanced inference with monitoring
│   ├── inference_batch.py       # Batch inference for 25 stocks (RF model)
│   ├── create_predictions_table.py    # Create stock predictions Delta table
│   └── load_testing_scenarios.py      # Automated performance testing
├── config/                       # Configuration management
│   └── config.yaml              # Centralized YAML configuration
├── docs/                         # Comprehensive documentation
│   ├── MONITORING_GUIDE.md      # Scalability monitoring guide
│   ├── PERFORMANCE_BENCHMARKS.md # Performance analysis guide
│   └── QUICK_REFERENCE.md       # Quick command reference
├── PROJECT_OVERVIEW.md          # 📘 Master architecture & setup guide
├── TRAINING_PIPELINE.md         # 🎓 3 training modes comparison (GBT, RF, Hybrid)
├── INFERENCE_PIPELINE.md        # 🔮 Airflow-triggered daily predictions
├── SCALABILITY_PERFORMANCE.md   # 📊 Volume tests (5/10/20/29 stocks)
├── STREAMING_PIPELINE.md        # 📡 Kafka + GPT sentiment analysis
├── AIRFLOW_SETUP.md            # ⚙️  4-stage DAG workflow & monitoring
├── ASTRADB_CONFIG.md           # 🗄️  NoSQL sentiment storage setup
├── data_csv/                     # Input data (5 stock symbols)
├── delta_tables/                 # Delta Lake storage with ACID properties
├── results/                      # Training, inference, and evaluation results
├── plots/                        # Generated visualizations
├── mlflow_tracking/              # MLflow experiments and model registry
├── models/                       # Saved model artifacts
└── logs/                         # Application and monitoring logs
```

## 🚀 Features

### **Core ML Pipeline**
- ✅ **Delta Lake Integration**: ACID transactions with efficient data storage
- ✅ **Advanced Feature Engineering**: 20+ optimized features for financial time series
- ✅ **Gradient Boosted Trees**: Native Spark MLlib with hyperparameter tuning
- ✅ **MLflow Integration**: Complete experiment tracking and model registry
- ✅ **Cross-Validation**: Robust model evaluation with k-fold validation
- ✅ **Comprehensive Visualization**: Feature importance, predictions, residuals analysis

### **Scalability Monitoring System** 🔧
- ✅ **Real-time Performance Tracking**: CPU, memory, throughput metrics
- ✅ **Data Volume Scalability**: Partition efficiency and data size analysis
- ✅ **MLflow Monitoring Integration**: Automatic metrics logging
- ✅ **Resource Utilization Analysis**: Spark executor and driver monitoring

### **Enhanced Training Features**
- ✅ **Monitored Training**: `train_model_with_monitoring.py` with performance tracking
- ✅ **Hyperparameter Optimization**: Optuna-based intelligent tuning
- ✅ **Feature Selection**: Correlation-based feature filtering
- ✅ **Model Validation**: Time-series aware train/test splitting
- ✅ **Artifact Management**: Comprehensive model and plot storage

### **Automated Testing & Benchmarking**
- ✅ **Load Testing Scenarios**: Varying data volumes and concurrency
- ✅ **Resource Profiling**: Memory and CPU usage analysis

## 📊 Monitoring Capabilities

### **Scalability Metrics**
- **Data Volume Metrics**: Rows processed, data size, partition efficiency
- **Performance Metrics**: Processing time, throughput, latency percentiles
- **Resource Metrics**: Memory usage, CPU utilization, executor efficiency
- **Pipeline Metrics**: Feature engineering time, model inference time
- **Quality Metrics**: Data consistency, missing values, outlier detection

### **Real-time Tracking**
- **MLflow Integration**: Automatic experiment tracking for all monitoring runs
- **JSON Reports**: Structured performance reports with detailed metrics
- **Log Analysis**: Comprehensive application and performance logging
- **Trend Analysis**: Historical performance tracking and comparison

## 🛠️ Setup Instructions

### 1. Prerequisites
```bash
# Ensure you're in the breaking_data conda environment
conda activate breaking_data

# Navigate to the pipeline directory
cd /home/mha2cob/Downloads/breaking_data/spark_ml_pipeline
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Environment Setup
```bash
# Load environment variables and verify setup
python -c "from src.utils import load_environment; load_environment(); print('✅ Environment loaded')"
```

### 4. Verify Installation
```bash
# Test basic functionality
python -c "
from src.utils import get_config
from src.scalability_monitor import ScalabilityMonitor
print('✅ All core modules imported successfully')
"
```

## 🚂 Training Pipeline

### 1. Basic Training
```bash
# Run standard training pipeline
python scripts/train_model.py
```

### 2. Enhanced Training with Monitoring
```bash
# Run training with comprehensive monitoring
python scripts/train_model_with_monitoring.py

# Train specific stocks with monitoring
python scripts/train_model_with_monitoring.py --stocks AAPL,GOOG,NVDA

# Disable monitoring for faster training
python scripts/train_model_with_monitoring.py --no-monitoring

# Run with load testing
python scripts/train_model_with_monitoring.py --load-test
```

### 3. Monitor Training Progress
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# View logs
tail -f logs/pipeline.log
```

## 🔮 Inference Pipeline

### 1. Basic Inference
```bash
# Run standard inference
python scripts/inference.py
```

### 2. Enhanced Inference with Monitoring
```bash
# Run inference with performance monitoring
python scripts/inference_with_monitoring.py

# Inference for specific stocks
python scripts/inference_with_monitoring.py --stocks AAPL,TSLA

# Disable monitoring for faster inference
python scripts/inference_with_monitoring.py --no-monitoring

# Run concurrent load testing
python scripts/inference_with_monitoring.py --load-test
```

### 3. Example Monitored Output
```
🚀 ENHANCED STOCK PRICE PREDICTIONS WITH MONITORING
================================================================================
📊 Monitoring Configuration:
   ✅ Scalability Monitoring: ENABLED
   📈 Resource Tracking: ENABLED
   🔧 Performance Analysis: ENABLED
   📋 MLflow Integration: ENABLED

📈 AAPL:
   Current Price: $150.25
   Predicted Price: $152.30 (+1.36%)
   Confidence: 92.4%

📊 PERFORMANCE METRICS:
   Total Processing Time: 2.34s
   Throughput: 8,245 records/sec
   Peak Memory Usage: 1,247 MB
   CPU Utilization: 67.3%
   Scalability Score: 0.847

🔧 MONITORING SUMMARY:
   ✅ Experiment logged: stock_forecasting_gbt_exp_unified_monitoring
   📊 Run ID: abc123def456
   📁 Artifacts saved: results/inference/
```

## 🎯 Batch Inference Pipeline

### 1. Setup Predictions Table
```bash
# Create the stock predictions Delta table with proper schema
python scripts/create_predictions_table.py
```

### 2. Run Batch Inference
```bash
# Execute batch inference for 25 stocks using Spark RF model
python scripts/inference_batch.py
```

The batch inference pipeline will:
1. **Load Spark RF model** and feature scaler from MLflow registry
2. **Prepare data** for 25 stocks (last 50 rows from each Delta table)
3. **Apply feature engineering** using the same pipeline as training
4. **Scale features** using the trained scaler
5. **Make predictions** using the loaded Spark RF model
6. **Calculate predicted close prices** using close gain formula
7. **Save predictions** to Delta table and optionally CSV

### 3. Example Batch Output
```
🚀 Starting Batch Inference for Stock Price Prediction
================================================================================
🔧 Configuration loaded successfully
📊 Inference stocks: 25
🔧 Spark session created successfully
🚀 Inference Batch Processor initialized
🔮 Starting batch inference for 25 stocks
📥 Loading model and scaler from MLflow...
✅ Model loaded successfully: stock_predictor_spark_rf v3
📊 Preparing inference data...
✅ Prepared 25 records for prediction
🔧 Applying feature engineering...
⚖️ Applying feature scaling...
✅ Feature scaling applied successfully
🎯 Making predictions...
✅ Predictions generated successfully
💰 Calculating predicted close prices...
✅ Predicted close prices calculated successfully
💾 Saving predictions to Delta table...
✅ Predictions saved to Delta table: delta_tables/stock_predictions

🎯 BATCH INFERENCE PERFORMANCE SUMMARY
================================================================================
📊 Stocks Processed: 25
⏱️  Total Time: 45.67 seconds (0.76 minutes)
⚡ Average Time per Stock: 1.83 seconds
🚀 Throughput: 0.55 predictions/second
================================================================================

🎉 Batch Inference Completed Successfully!
================================================================================
✅ Status: success
📊 Predictions Generated: 25
🏢 Stocks Processed: 25
🤖 Model: stock_predictor_spark_rf v3
⏱️  Total Time: 45.67 seconds
🚀 Throughput: 0.55 predictions/second
```

### 4. Prediction Schema
The batch inference saves predictions to `delta_tables/stock_predictions` with the following schema:
```
prediction_date         date      # Next business day
prediction_timestamp    timestamp # Next business day at 9:30 AM
predicted_close         double    # Calculated using: last_close * (1 + predicted_gain)
model_confidence        double    # Fixed at 0.85
model_version           string    # MLflow model version
features_used           string    # Comma-separated feature names
created_at              timestamp # When prediction was made
days_ahead              integer   # Always 1
stock_symbol            string    # Stock symbol
```

## 🧪 Load Testing & Benchmarking

### 1. Run Load Testing Scenarios
```bash
# Baseline performance test
python scripts/load_testing_scenarios.py --scenario baseline --duration 60

# Stress testing
python scripts/load_testing_scenarios.py --scenario stress --duration 300

# Volume scaling test
python scripts/load_testing_scenarios.py --scenario volume --duration 120

# Comprehensive testing
python scripts/load_testing_scenarios.py --scenario all --duration 600
```


## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# Data Configuration
data:
  available_stocks: [AAPL, BABA, GOOG, NVDA, TSLA]
  sequence_length: 50
  train_split: 0.85

# Feature Engineering
features:
  max_features: 25
  correlation_threshold: 0.95
  enable_interactions: true
  enable_rsi: true
  enable_volatility: true

# Model Configuration
model:
  algorithm: GBTRegressor
  default_params:
    maxIter: 150
    maxDepth: 8
    stepSize: 0.05

# Monitoring Configuration
monitoring:
  enabled: true
  detailed_metrics: true
  resource_monitoring_interval: 1.0
  generate_reports: true
  thresholds:
    max_cpu_usage_percent: 80
    max_memory_usage_percent: 85
    min_throughput_records_per_second: 100

# MLflow Configuration
mlflow:
  experiment_name: stock_forecasting_gbt_exp
  log_artifacts: true
  log_metrics: true
  log_models: true
```

## 📊 Monitoring Results & Analysis

### MLflow Integration
- **Experiment Tracking**: `stock_forecasting_gbt_exp_unified_monitoring`
- **Model Registry**: Automatic model versioning and stage management
- **Metrics Logging**: Performance, scalability, and accuracy metrics
- **Artifact Storage**: Models, plots, monitoring reports


### Visualization Outputs
- **Feature Importance**: Interactive and static plots
- **Performance Trends**: Time-series monitoring charts
- **Resource Utilization**: CPU and memory usage over time
- **Scalability Analysis**: Throughput vs data volume charts

## 📈 Performance Benchmarks

### Expected Performance (Local 32GB System)
- **Training Time**: 10-20 minutes 
- **Inference Latency**: <100ms per prediction
- **Throughput**: 200 records/second

### Monitoring Thresholds
- **CPU Usage**: Alert at 85%, critical at 95%
- **Memory Usage**: Alert at 85% of allocated memory
- **Throughput**: Minimum 100 records/second


## 🔍 Monitoring Results Access

### MLflow UI
```bash
# Start MLflow tracking server
mlflow ui --host 0.0.0.0 --port 5000
# Access: http://localhost:5000
```


## 📚 Documentation

### **Academic Project Documentation** (Data Engineering at Scale)
- **� [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**: Master architecture & complete system overview
- **🎓 [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)**: 3 training modes comparison (Spark GBT, Spark RF, Hybrid Sklearn)
- **🔮 [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md)**: Airflow-triggered daily predictions (4-stage DAG)
- **�📊 [SCALABILITY_PERFORMANCE.md](SCALABILITY_PERFORMANCE.md)**: Volume tests (5/10/20/29 stocks) with metrics
- **📡 [STREAMING_PIPELINE.md](STREAMING_PIPELINE.md)**: Kafka + GPT sentiment analysis + AstraDB
- **⚙️  [AIRFLOW_SETUP.md](AIRFLOW_SETUP.md)**: 4-stage DAG workflow, scheduling, monitoring
- **🗄️  [ASTRADB_CONFIG.md](ASTRADB_CONFIG.md)**: NoSQL sentiment storage setup & integration

### **Developer Documentation**
- **📊 [docs/MONITORING_GUIDE.md](docs/MONITORING_GUIDE.md)**: Comprehensive monitoring documentation
- **🚀 [docs/PERFORMANCE_BENCHMARKS.md](docs/PERFORMANCE_BENCHMARKS.md)**: Performance analysis and optimization
- **⚡ [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**: Quick command reference and troubleshooting


### Monitoring Benefits
- **🔍 Bottleneck Detection**: Automated performance constraint identification
- **📊 Resource Optimization**: Intelligent resource allocation recommendations
- **🚨 Early Warning**: Threshold-based alerting for performance degradation
- **📈 Trend Analysis**: Historical performance tracking and comparison
- **🎯 Quality Assurance**: Comprehensive data and model validation

## 🚀 Future Enhancements

- [ ] **Model Retraining**: Drift based Model retraining Pipeline
- [ ] **Automated Scaling**: Dynamic resource allocation based on monitoring
- [ ] **Recursive Prediction**: currently prediction close gain, extend it to high, open, close, volume to enable recursive prediction
- [ ] **Model Drift Detection**: Automated model performance degradation alerts
- [ ] **Custom Dashboards**: Real-time monitoring dashboard interfaces
- [ ] **Cloud Deployment**: Cloud Deployment of entire streaming, train and inference pipeline.

---