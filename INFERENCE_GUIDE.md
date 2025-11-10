# Enhanced Stock Price Inference Pipeline Guide

## Overview
The enhanced inference script (`scripts/inference.py`) provides production-ready stock price prediction capabilities with comprehensive feature engineering, data validation, and prediction persistence.

## Key Enhancements Implemented

### ✅ **Proper Feature Engineering Pipeline**
- Uses the same `FeatureEngineer` class as training
- Creates all 25 features including price lags, moving averages, technical indicators, and interactions
- Ensures feature consistency between training and inference

### ✅ **Production-Ready Architecture**
- Comprehensive error handling and validation
- MLflow experiment tracking for inference runs
- System performance monitoring
- Proper logging with timestamps

### ✅ **Data Quality Validation**
- Minimum data requirements checking
- Missing value validation
- Data continuity checks (gap detection)
- Prediction reasonableness validation

### ✅ **Prediction Persistence**
- Automatic saving to Delta tables (`delta_tables/predictions/`)
- Schema compatibility with existing tables
- Batch saving optimization
- Configurable via `config.yaml`

### ✅ **Flexible Usage Options**
- **Batch Processing**: Run all stocks with optimized performance
- **Individual Stocks**: Target specific stocks via command line
- **Command Line Interface**: Easy stock selection

### ✅ **Enhanced Error Handling**
- Graceful failure handling per stock
- Detailed error reporting
- Continues processing even if individual stocks fail
- Comprehensive logging and debugging information

## Usage Examples

### 1. Run All Stocks (Optimized Batch Processing)
```bash
conda activate breaking_data
python scripts/inference.py
```
**Output**: Processes all stocks listed in `config.data.inference_stocks` using optimized batch processing.

### 2. Run Specific Stocks
```bash
# Single stock
python scripts/inference.py AAPL

# Multiple stocks
python scripts/inference.py AAPL GOOG NVDA TSLA

# Any combination
python scripts/inference.py NVDA BABA
```

### 3. Configuration Options
Edit `config/config.yaml` to customize inference behavior:

```yaml
inference:
  # Data requirements
  lookback_days: 49
  min_required_rows: 49
  
  # Model configuration  
  model_name: "stock_predictor_unified_5stocks"
  model_version: "latest"
  model_stage: "None"
  
  # Prediction settings
  predict_next_trading_day: true
  skip_weekends_holidays: true
  save_predictions: true  # Set to false to disable persistence
  
  # Quality validation
  validate_data_quality: true
  quality_checks:
    check_missing_values: true
    check_data_continuity: true
    max_missing_ratio: 0.1
```

## Technical Implementation Details

### **Feature Engineering Process**
1. **Load Recent Data**: Gets last 59 rows from stock Delta tables (49 for features + buffer)
2. **Create Features**: Uses `FeatureEngineer.create_all_features()` - same as training
3. **Feature Vector**: Assembles features using `VectorAssembler`
4. **Scaling**: Applies saved `StandardScaler` from training
5. **Latest Row**: Selects most recent row for prediction

### **Prediction Pipeline**
1. **Data Loading**: Load from individual stock tables (`delta_tables/stock_AAPL/`)
2. **Validation**: Check data quality and completeness
3. **Feature Engineering**: Create same 25 features as training
4. **Model Prediction**: Load model from MLflow and predict return/gain
5. **Price Calculation**: `predicted_close = current_close * (1 + predicted_gain)`
6. **Persistence**: Save to `delta_tables/predictions/` (optional)

### **Batch Processing Optimization**
- Reuses loaded model and scaler across all stocks
- Individual error handling per stock
- Batch prediction saving
- Comprehensive performance metrics

## Performance Metrics

From recent test run (5 stocks):
- **Total Time**: 31.35 seconds  
- **Average per Stock**: 6.27 seconds
- **Successful Predictions**: 5/5 (100%)
- **Features Created**: 25 per stock
- **Memory Efficient**: Optimized Spark configurations

## Output Schema

### **Predictions Table** (`delta_tables/predictions/`)
```
stock_symbol: string
prediction_date: timestamp  
predicted_for_date: string (YYYY-MM-DD)
predicted_gain: double (decimal return, e.g., 0.0341 = 3.41%)
predicted_close: double (predicted closing price)
last_close: double (current closing price)
model_used: string (MLflow model name)
model_version: string 
rows_used: string (data rows used for prediction)
created_at: timestamp
```

### **Console Output Example**
```
📈 PREDICTIONS:
--------------------------------------------------------------------------------
AAPL   | Last Close: $197.57 | Predicted Close: $190.83 | Gain: -0.0341 (-3.41%) | Date: 2023-12-18
GOOG   | Last Close: $133.84 | Predicted Close: $132.94 | Gain: -0.0068 (-0.68%) | Date: 2023-12-18
NVDA   | Last Close: $488.90 | Predicted Close: $485.08 | Gain: -0.0078 (-0.78%) | Date: 2023-12-18
TSLA   | Last Close: $253.50 | Predicted Close: $256.43 | Gain: 0.0116 (1.16%) | Date: 2023-12-18
BABA   | Last Close: $74.51 | Predicted Close: $74.71 | Gain: 0.0027 (0.27%) | Date: 2023-12-18
--------------------------------------------------------------------------------
```

## Error Handling

### **Common Issues & Solutions**

1. **"Insufficient data"**: Check if stock table has at least 49 recent rows
2. **"Model not found"**: Verify MLflow model exists and version is correct
3. **"Feature scaler not loaded"**: Ensure `models/feature_scaler/` exists from training
4. **"Schema mismatch"**: Script automatically handles different prediction table schemas

### **Validation Features**
- **Data Quality**: Checks for missing values and data gaps
- **Prediction Bounds**: Validates predictions within reasonable ranges (±50%)
- **Model Compatibility**: Ensures feature names match training expectations
- **Resource Monitoring**: Tracks memory, CPU, and disk usage

## Integration with Existing Pipeline

### **Follows Training Pipeline Exactly**
- Same feature engineering (25 features)
- Same data preprocessing steps
- Same scaling methodology
- Same target variable calculation (next-day return)

### **MLflow Integration**
- Tracks inference runs as experiments
- Logs performance metrics
- Records system resource usage
- Maintains prediction history

### **Delta Lake Benefits**
- ACID transactions for prediction storage
- Schema evolution support
- Time travel capabilities
- Optimized storage and querying

## Best Practices

1. **Regular Runs**: Schedule daily inference after market close
2. **Data Freshness**: Ensure stock tables are updated with latest market data
3. **Model Versioning**: Use specific model versions for production consistency
4. **Resource Monitoring**: Monitor memory usage for large batch runs
5. **Prediction Validation**: Review predictions for reasonableness before using
6. **Backup Strategy**: Regularly backup MLflow models and prediction history

## Future Enhancements Roadmap

- [ ] **Multi-Model Support**: Per-stock model selection capability
- [ ] **Real-Time Processing**: Streaming inference for intraday predictions  
- [ ] **Advanced Validation**: Statistical validation against historical accuracy
- [ ] **Alert System**: Notifications for extreme predictions or failures
- [ ] **Performance Optimization**: Further Spark tuning for large-scale inference
- [ ] **Model Ensemble**: Combine multiple models for improved accuracy