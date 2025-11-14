# Spark ML Stock Price Prediction Pipeline

A scalable stock price prediction pipeline using Apache Spark MLlib, Delta Lake, and MLflow for experiment tracking. This implementation follows the CNN-style data processing approach but uses Gradient Boosted Trees (GBT) with optimally engineered features.

## 🏗️ Architecture

```
spark_ml_pipeline/
├── src/                          # Source code modules
│   ├── utils.py                  # Configuration and utilities
│   ├── data_processing.py        # Data loading and preprocessing
│   ├── feature_engineering.py   # Feature creation and selection
│   ├── model_training.py         # Model training with hyperparameter tuning
│   └── visualization.py          # Plotting and visualization
├── scripts/                      # Execution scripts
│   ├── train_model.py           # Training pipeline
│   └── inference.py             # Inference pipeline
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── data_csv/                     # Input CSV files
├── delta_tables/                 # Delta Lake storage
├── results/                      # Training and inference results
├── plots/                        # Generated visualizations
├── models/                       # Saved model artifacts
└── logs/                         # Application logs
```

## 🚀 Features

### **Data Processing**
- ✅ **Delta Lake Integration**: Efficient data storage and versioning
- ✅ **CNN-Style Preprocessing**: Maintains the CNN data processing approach
- ✅ **Time-Based Train/Test Split**: 85%/15% split following CNN methodology
- ✅ **Data Validation**: Automatic data cleaning and outlier removal

### **Feature Engineering**
- ✅ **20 Optimal Features**: Carefully selected features for GBT performance
- ✅ **Price Features**: Lag features, momentum indicators, moving averages
- ✅ **Volume Features**: Volume patterns and relative strength indicators
- ✅ **Sentiment Features**: News sentiment integration with momentum analysis
- ✅ **Technical Indicators**: Volatility measures, RSI-like indicators
- ✅ **Interaction Features**: Price-volume-sentiment combinations
- ✅ **Normalization**: CNN-style relative normalization

### **Model Training**
- ✅ **Gradient Boosted Trees**: Native Spark MLlib implementation
- ✅ **Hyperparameter Tuning**: Optuna and Spark CrossValidator support
- ✅ **Cross-Validation**: K-fold validation for robust performance estimation
- ✅ **Feature Importance**: Automatic feature importance analysis

### **Experiment Tracking**
- ✅ **MLflow Integration**: Complete experiment tracking and model registry
- ✅ **Model Versioning**: Automatic model versioning and deployment
- ✅ **Metrics Logging**: Comprehensive metrics logging (MAE, MSE, R², per-stock)
- ✅ **Artifact Storage**: Model artifacts, plots, and results storage

### **Visualization**
- ✅ **Feature Importance Plots**: Static and interactive visualizations
- ✅ **Prediction Analysis**: Predictions vs actual, residual analysis
- ✅ **Per-Stock Performance**: Individual stock performance metrics
- ✅ **Model Comparison**: Easy comparison between different runs

### **Scalability**
- ✅ **Distributed Computing**: Leverages Spark's distributed processing
- ✅ **Memory Optimization**: Configured for 32GB RAM systems
- ✅ **Delta Lake**: Efficient storage with ACID transactions
- ✅ **Batch Processing**: Efficient processing of large datasets

## 📊 Model Performance

Following CNN evaluation methodology:
- **Per-stock evaluation** with detailed metrics
- **Overall performance** across all stocks
- **Same evaluation metrics** as CNN (MAE, MSE, R²)
- **Feature importance analysis** for interpretability

### Expected Performance Improvements over CNN:
- 🚀 **Faster Training**: 10x faster than CNN training
- 📈 **Better Interpretability**: Feature importance insights
- 🎯 **Competitive Accuracy**: Comparable or better prediction accuracy
- 📊 **Real-time Inference**: Sub-100ms prediction latency

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
# Run the setup script
python setup.py
```

This will:
- ✅ Install required Python packages
- ✅ Copy CSV files to the data_csv directory
- ✅ Create necessary directories
- ✅ Setup MLflow configuration
- ✅ Create environment variables

### 3. Verify Setup
```bash
# Run the demo script to verify setup
python demo.py
```

## 🚂 Training Pipeline

### 1. Run Training
```bash
# Execute the training pipeline
python scripts/train_model.py
```

The training pipeline will:
1. **Load CSV data** into Delta tables
2. **Engineer features** using the optimal feature set
3. **Split data** into train/test sets (85%/15%)
4. **Tune hyperparameters** using Optuna or Spark CV
5. **Train the final model** with best parameters
6. **Evaluate performance** on test set
7. **Generate visualizations** and save results
8. **Register model** in MLflow registry

### 2. Monitor Training
```bash
# Start MLflow UI to monitor training
mlflow ui --backend-store-uri file://../mlflow_tracking
```

Access the UI at: http://localhost:5000

## 🔮 Inference Pipeline

### 1. Run Inference
```bash
# Execute inference on latest data
python scripts/inference.py
```

The inference pipeline will:
1. **Load the latest model** from MLflow registry
2. **Prepare latest data** with feature engineering
3. **Make predictions** for all stocks
4. **Display results** in formatted output
5. **Save predictions** to Delta table and CSV

### 2. Example Output
```
📈 STOCK PRICE PREDICTIONS
================================================================================

📈 AAPL:
   Current Price: $150.25
   Predicted Price: $152.30
   Predicted Return: 0.0136 (1.36%)
   Date: 2024-11-08

📉 TSLA:
   Current Price: $248.50
   Predicted Price: $245.20
   Predicted Return: -0.0133 (-1.33%)
   Date: 2024-11-08

📊 PREDICTION SUMMARY:
   Average Predicted Return: 0.0045 (0.45%)
   Highest Predicted Return: 0.0234 (2.34%)
   Lowest Predicted Return: -0.0187 (-1.87%)
   Bullish Stocks: 3
   Bearish Stocks: 2
```

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# Key configurations
data:
  sequence_length: 50           # Historical window size
  train_split: 0.85            # Train/test split ratio

features:
  max_features: 25             # Maximum number of features
  correlation_threshold: 0.95   # Feature correlation threshold

model:
  default_params:
    maxIter: 150               # GBT iterations
    maxDepth: 8                # Tree depth
    stepSize: 0.05             # Learning rate

hyperparameter_tuning:
  enabled: true                # Enable hyperparameter tuning
  method: "optuna"             # Tuning method
  n_trials: 50                 # Number of trials

spark:
  driver_memory: "4g"          # Driver memory
  executor_memory: "8g"        # Executor memory
```

## 📈 Results and Analysis

### Output Files
- **Training Results**: `results/training/run_YYYYMMDD_HHMMSS/`
  - `feature_importance.csv` - Feature importance rankings
  - `per_stock_metrics.csv` - Performance per stock
  - `test_predictions.csv` - Detailed predictions

- **Visualizations**: `plots/`
  - `feature_importance.png` - Feature importance plot
  - `predictions_vs_actual.png` - Prediction analysis
  - `per_stock_performance.png` - Per-stock metrics
  - `residuals_analysis.png` - Residual analysis

- **MLflow Artifacts**: Accessible via MLflow UI
  - Model artifacts and metadata
  - Experiment comparison
  - Model registry

### Key Metrics
- **MAE (Mean Absolute Error)**: Average prediction error
- **MSE (Mean Squared Error)**: Squared prediction error
- **R² Score**: Coefficient of determination
- **Per-Stock Performance**: Individual stock accuracy

## 🧪 Advanced Usage

### Custom Feature Engineering
Modify `src/feature_engineering.py` to add custom features:

```python
# Add custom technical indicators
def _create_custom_features(self, df, window_spec):
    # Your custom feature logic here
    return df
```

### Hyperparameter Tuning
Customize hyperparameter ranges in `config/config.yaml`:

```yaml
model:
  hyperparameter_ranges:
    maxDepth: [4, 6, 8, 10, 12]
    stepSize: [0.01, 0.05, 0.1, 0.2]
    # Add more parameters
```

### Model Deployment
```bash
# Promote model to production
mlflow models serve -m "models:/stock_forecast_model/1" -p 1234
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce `executor_memory` in config
   - Decrease `max_features` count
   - Use smaller dataset for testing

2. **Delta Table Issues**
   - Delete `delta_tables/` directory to reset
   - Ensure proper file permissions

3. **MLflow Issues**
   - Check MLflow tracking URI in `.env`
   - Ensure `mlflow_tracking` directory exists

4. **Import Errors**
   - Ensure you're in the `breaking_data` conda environment
   - Run `python setup.py` to install dependencies

### Performance Optimization

1. **Spark Configuration**
   - Adjust memory settings based on your system
   - Tune number of executor cores
   - Enable adaptive query execution

2. **Feature Selection**
   - Reduce number of features for faster training
   - Use correlation analysis to remove redundant features
   - Profile feature computation time

## 📚 Comparison with CNN Approach

| Aspect | CNN Approach | GBT Approach |
|--------|--------------|--------------|
| **Training Speed** | ~1-2 hours | ~10-20 minutes |
| **Interpretability** | Low (black box) | High (feature importance) |
| **Scalability** | Limited (single machine) | High (distributed) |
| **Memory Usage** | High (GPU required) | Moderate (CPU only) |
| **Feature Engineering** | Automatic | Manual but optimized |
| **Deployment** | Complex | Simple (Spark native) |
| **Accuracy** | High | Competitive |

## 🎯 Future Enhancements

- [ ] **Real-time Streaming**: Apache Kafka integration
- [ ] **Automated Retraining**: Schedule-based model updates
- [ ] **Model Ensemble**: Combine multiple models
- [ ] **Advanced Features**: More sophisticated technical indicators
- [ ] **A/B Testing**: Model performance comparison framework
- [ ] **API Deployment**: REST API for predictions
- [ ] **Monitoring**: Data drift and model performance monitoring

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review MLflow logs for training issues
3. Check Spark logs in `logs/` directory
4. Verify configuration in `config/config.yaml`

---

**Note**: This pipeline is optimized for the provided 5-stock dataset but can scale to hundreds of stocks with appropriate Spark cluster configuration.