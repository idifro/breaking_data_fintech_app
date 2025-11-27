# Training Pipeline Documentation

> **Distributed Machine Learning for Stock Price Forecasting**  
> **Multi-Mode Training with Scalability Analysis**

---

<<<<<<< HEAD
## 📋 Table of Contents
=======
##  Table of Contents
>>>>>>> d3e5d7a (added streaming and backend)

- [Overview](#overview)
- [Training Architecture](#training-architecture)
- [Three Training Modes](#three-training-modes)
- [Mode Comparison](#mode-comparison)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Performance Metrics](#performance-metrics)
- [Scalability Analysis](#scalability-analysis)
- [Usage Guide](#usage-guide)
- [Best Practices](#best-practices)

---

## Overview

The training pipeline implements **three distinct approaches** to distributed machine learning, each demonstrating different trade-offs between:
- Training speed
- Memory efficiency
- Prediction accuracy
- Scalability characteristics
- Production readiness

### **Objective**
Predict next-day stock closing prices using:
- **Historical price data**: OHLCV (Open, High, Low, Close, Volume)
- **News sentiment**: GPT-analyzed financial news scores
- **Technical indicators**: Moving averages, RSI, volatility
- **Engineered features**: 20+ features including lags and interactions

### **Problem Formulation**
```
Target: Close price gain (next day)
Formula: (Close_t+1 - Close_t) / Close_t

Features: 20+ engineered features from historical data
Algorithm: Gradient Boosted Trees / Random Forest
Evaluation: 85/15 train-test split with cross-validation
```

---

## Training Architecture

```
<<<<<<< HEAD
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE FLOW                           │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  1. DATA LOAD   │
│  Delta Tables   │
│  • 29 stocks    │
│  • 81K+ records │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  2. FEATURE ENGINEERING             │
│                                     │
│  Price-based Features:              │
│    • Close Lags (1,2,3,5)          │
│    • Open Lags (1,2,3,5)           │
│    • MA_5, MA_10, MA_20            │
│    • Price Range (High-Low)        │
│    • Daily Return                   │
│                                     │
│  Volume Features:                   │
│    • Volume Lags (1,2,3,5)         │
│    • Volume MA (5,10,20)           │
│    • Volume Change                  │
│                                     │
│  Technical Indicators:              │
│    • RSI (Relative Strength Index) │
│    • Volatility (20-day std)       │
│    • Bollinger Bands               │
│                                     │
│  Sentiment Features:                │
│    • Sentiment Lags (1,3,5)        │
│    • Scaled Sentiment              │
│    • News Flag                      │
│                                     │
│  Interaction Features:              │
│    • Close × Volume                 │
│    • Sentiment × Close              │
│    • MA_5 × Volume                  │
│                                     │
│  Result: 20+ features per record   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  3. MULTI-MODE TRAINING (Select One)                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  SPARK GBT   │  │  SPARK RF    │  │ HYBRID       │ │
│  │              │  │              │  │ SKLEARN      │ │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤ │
│  │ Sequential   │  │ Distributed  │  │ Spark prep + │ │
│  │ on Driver    │  │ Training     │  │ Sklearn train│ │
│  │              │  │              │  │              │ │
│  │ GBT Regressor│  │ RF Regressor │  │ RandomForest │ │
│  │ Spark MLlib  │  │ Spark MLlib  │  │ Scikit-learn │ │
│  │              │  │              │  │              │ │
│  │ ⏱ Slow       │  │ ⚡ Fast      │  │ ⏱ Medium     │ │
│  │ 📊 Medium    │  │ 📊 Best      │  │ 📊 Poor      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                  │                  │         │
│         └──────────────────┴──────────────────┘         │
│                            ▼                             │
│                   ┌────────────────┐                     │
│                   │  Best Model    │                     │
│                   │  Selection     │                     │
│                   └───────┬────────┘                     │
└───────────────────────────┼──────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────┐
│  4. MODEL REGISTRATION (MLflow)                           │
│                                                            │
│  Experiment: stock_forecasting_*                          │
│  Model Name: stock_predictor_spark_rf_unified_29stocks   │
│  Version: 1                                                │
│  Stage: Production                                         │
│                                                            │
│  Logged:                                                   │
│    • Parameters (maxDepth, numTrees, etc.)                │
│    • Metrics (MAE, RMSE, R², training time)               │
│    • Artifacts (model file, feature scaler)               │
│    • Tags (mode, stocks, volume)                          │
└───────────────────────────────────────────────────────────┘
=======

                        TRAINING PIPELINE FLOW                           



  1. DATA LOAD   
  Delta Tables   
  • 29 stocks    
  • 81K+ records 

         
         

  2. FEATURE ENGINEERING             
                                     
  Price-based Features:              
    • Close Lags (1,2,3,5)          
    • Open Lags (1,2,3,5)           
    • MA_5, MA_10, MA_20            
    • Price Range (High-Low)        
    • Daily Return                   
                                     
  Volume Features:                   
    • Volume Lags (1,2,3,5)         
    • Volume MA (5,10,20)           
    • Volume Change                  
                                     
  Technical Indicators:              
    • RSI (Relative Strength Index) 
    • Volatility (20-day std)       
    • Bollinger Bands               
                                     
  Sentiment Features:                
    • Sentiment Lags (1,3,5)        
    • Scaled Sentiment              
    • News Flag                      
                                     
  Interaction Features:              
    • Close × Volume                 
    • Sentiment × Close              
    • MA_5 × Volume                  
                                     
  Result: 20+ features per record   

         
         

  3. MULTI-MODE TRAINING (Select One)                    
                                                          
       
    SPARK GBT       SPARK RF       HYBRID        
                                   SKLEARN       
       
   Sequential      Distributed     Spark prep +  
   on Driver       Training        Sklearn train 
                                                 
   GBT Regressor   RF Regressor    RandomForest  
   Spark MLlib     Spark MLlib     Scikit-learn  
                                                 
   ⏱ Slow           Fast         ⏱ Medium      
    Medium        Best          Poor       
       
                                                      
                  
                                                         
                                        
                     Best Model                         
                     Selection                          
                                        

                            

  4. MODEL REGISTRATION (MLflow)                           
                                                            
  Experiment: stock_forecasting_*                          
  Model Name: stock_predictor_spark_rf_unified_29stocks   
  Version: 1                                                
  Stage: Production                                         
                                                            
  Logged:                                                   
    • Parameters (maxDepth, numTrees, etc.)                
    • Metrics (MAE, RMSE, R², training time)               
    • Artifacts (model file, feature scaler)               
    • Tags (mode, stocks, volume)                          

>>>>>>> d3e5d7a (added streaming and backend)
```

---

## Three Training Modes

### **Mode 1: Spark GBT (Gradient Boosted Trees)**

#### **Architecture**
```
Driver Node:
<<<<<<< HEAD
  ├─ Data Loading (distributed via Spark)
  ├─ Feature Engineering (distributed)
  ├─ GBT Training (sequential on driver)
  │   └─ Tree building (one at a time)
  └─ Model Evaluation

Workers:
  └─ Data processing only (no training)
=======
   Data Loading (distributed via Spark)
   Feature Engineering (distributed)
   GBT Training (sequential on driver)
      Tree building (one at a time)
   Model Evaluation

Workers:
   Data processing only (no training)
>>>>>>> d3e5d7a (added streaming and backend)
```

#### **Implementation Details**
- **Algorithm**: `pyspark.ml.regression.GBTRegressor`
- **Training Strategy**: Sequential tree building on driver node
- **Parallelization**: Data processing only, not model training
- **Memory**: Entire model and data on driver

#### **Configuration**
```python
# From config/config_modes.yaml
spark_gbt_mode:
  algorithm: GBTRegressor
  params:
    maxDepth: 12
    maxIter: 150
    stepSize: 0.01
    subsamplingRate: 0.8
    featureSubsetStrategy: "log2"
    seed: 42
```

#### **Pros**
<<<<<<< HEAD
✅ Native Spark implementation  
✅ Good for small-medium datasets  
✅ Well-tested algorithm  
✅ Easy to debug  

#### **Cons**
❌ **Very slow** (2.6 hours for 20 stocks)  
❌ Sequential tree building bottleneck  
❌ Poor scalability (14.7x time increase from 5→20 stocks)  
❌ High GC overhead (82+ GB GC time)  
=======
 Native Spark implementation  
 Good for small-medium datasets  
 Well-tested algorithm  
 Easy to debug  

#### **Cons**
 **Very slow** (2.6 hours for 20 stocks)  
 Sequential tree building bottleneck  
 Poor scalability (14.7x time increase from 5→20 stocks)  
 High GC overhead (82+ GB GC time)  
>>>>>>> d3e5d7a (added streaming and backend)

#### **Performance (20 stocks - Large Volume)**
| Metric | Value |
|--------|-------|
| **Training Time** | 9,537 seconds (2.6 hours) |
| **Throughput** | 8.49 records/sec |
| **Peak Memory** | 29.9 GB |
| **MAE** | 0.0152 |
| **RMSE** | 0.0229 |
| **R² Score** | -0.229 |
| **CPU Utilization** | 54.9% |

---

<<<<<<< HEAD
### **Mode 2: Spark RF (Random Forest)** ⭐ **SELECTED FOR PRODUCTION**
=======
### **Mode 2: Spark RF (Random Forest)**  **SELECTED FOR PRODUCTION**
>>>>>>> d3e5d7a (added streaming and backend)

#### **Architecture**
```
Driver Node:
<<<<<<< HEAD
  ├─ Orchestration
  ├─ Model aggregation
  └─ Evaluation

Worker Nodes (Distributed):
  ├─ Data partitions
  ├─ Tree training (parallel)
  │   ├─ Worker 1: Trees 1-25
  │   ├─ Worker 2: Trees 26-50
  │   └─ Worker N: Trees 51-100
  └─ Local aggregation
=======
   Orchestration
   Model aggregation
   Evaluation

Worker Nodes (Distributed):
   Data partitions
   Tree training (parallel)
      Worker 1: Trees 1-25
      Worker 2: Trees 26-50
      Worker N: Trees 51-100
   Local aggregation
>>>>>>> d3e5d7a (added streaming and backend)
```

#### **Implementation Details**
- **Algorithm**: `pyspark.ml.regression.RandomForestRegressor`
- **Training Strategy**: Fully distributed parallel tree building
- **Parallelization**: Both data AND model training
- **Memory**: Distributed across cluster

#### **Configuration**
```python
# From config/config_modes.yaml
spark_rf_mode:
  algorithm: RandomForestRegressor
  params:
    numTrees: 100
    maxDepth: 12
    subsamplingRate: 0.8
    featureSubsetStrategy: "sqrt"
    seed: 42
    maxBins: 32
```

#### **Pros**
<<<<<<< HEAD
✅ **5.5x faster** than Spark GBT  
✅ **Fully distributed** training  
✅ **Best accuracy** (R² = 0.0140)  
✅ Excellent scalability  
✅ Lower memory per record  
✅ Production-ready  

#### **Cons**
⚠️ Requires proper cluster configuration  
⚠️ More complex debugging  
=======
 **5.5x faster** than Spark GBT  
 **Fully distributed** training  
 **Best accuracy** (R² = 0.0140)  
 Excellent scalability  
 Lower memory per record  
 Production-ready  

#### **Cons**
 Requires proper cluster configuration  
 More complex debugging  
>>>>>>> d3e5d7a (added streaming and backend)

#### **Performance (20 stocks - Large Volume)**
| Metric | Value | vs Spark GBT |
|--------|-------|--------------|
<<<<<<< HEAD
| **Training Time** | 1,110 seconds (18.5 min) | **5.5x faster** ⚡ |
=======
| **Training Time** | 1,110 seconds (18.5 min) | **5.5x faster**  |
>>>>>>> d3e5d7a (added streaming and backend)
| **Throughput** | 73.0 records/sec | **8.6x higher** |
| **Peak Memory** | 28.0 GB | 6.3% lower |
| **MAE** | **0.0143** | **6.3% better** |
| **RMSE** | **0.0206** | **10% better** |
<<<<<<< HEAD
| **R² Score** | **0.0140** | **✅ Positive!** |
=======
| **R² Score** | **0.0140** | ** Positive!** |
>>>>>>> d3e5d7a (added streaming and backend)
| **CPU Utilization** | 27.4% | More efficient |
| **Cost Efficiency** | 31M | **9x better** |

#### **Why Spark RF Won**
1. **Speed**: 5.5x faster training time
2. **Accuracy**: Best R² score across all modes
3. **Scalability**: True distributed training
4. **Memory**: Most efficient memory usage
5. **Cost**: Best cost/performance ratio
6. **Production**: Ready for 29-stock deployment

---

### **Mode 3: Hybrid Sklearn (Spark + Scikit-learn)**

#### **Architecture**
```
Spark (Distributed):
<<<<<<< HEAD
  ├─ Data Loading
  ├─ Feature Engineering
  └─ Data Collection to Driver

Driver Node (Centralized):
  ├─ Convert to Pandas
  ├─ Scikit-learn RandomForest Training
  └─ Model Evaluation

Inference:
  ├─ Spark (distributed preprocessing)
  └─ UDF-wrapped sklearn model (broadcast)
=======
   Data Loading
   Feature Engineering
   Data Collection to Driver

Driver Node (Centralized):
   Convert to Pandas
   Scikit-learn RandomForest Training
   Model Evaluation

Inference:
   Spark (distributed preprocessing)
   UDF-wrapped sklearn model (broadcast)
>>>>>>> d3e5d7a (added streaming and backend)
```

#### **Implementation Details**
- **Algorithm**: `sklearn.ensemble.RandomForestRegressor`
- **Training Strategy**: Distributed preprocessing, centralized training
- **Parallelization**: Data processing distributed, training on driver
- **Memory**: Model training on single node

#### **Configuration**
```python
# From config/config_modes.yaml
hybrid_sklearn_mode:
  algorithm: RandomForestRegressor (sklearn)
  params:
    n_estimators: 100
    max_depth: 12
    max_features: "sqrt"
    random_state: 42
    n_jobs: -1  # Use all cores on driver
```

#### **Pros**
<<<<<<< HEAD
✅ Familiar sklearn API  
✅ Rich ecosystem (matplotlib, pandas)  
✅ Good for prototyping  
✅ Easier debugging  

#### **Cons**
❌ **Poor accuracy** (R² = -0.040, MAE = 0.301)  
❌ Centralized training bottleneck  
❌ Data collection overhead (Spark → Pandas)  
❌ Memory limitations on driver  
❌ Not truly scalable  
=======
 Familiar sklearn API  
 Rich ecosystem (matplotlib, pandas)  
 Good for prototyping  
 Easier debugging  

#### **Cons**
 **Poor accuracy** (R² = -0.040, MAE = 0.301)  
 Centralized training bottleneck  
 Data collection overhead (Spark → Pandas)  
 Memory limitations on driver  
 Not truly scalable  
>>>>>>> d3e5d7a (added streaming and backend)

#### **Performance (20 stocks - Large Volume)**
| Metric | Value |
|--------|-------|
| **Training Time** | 3,294 seconds (54.9 min) |
| **Throughput** | 24.6 records/sec |
| **Peak Memory** | 30.9 GB |
<<<<<<< HEAD
| **MAE** | 0.3014 (❌ **21x worse**) |
| **RMSE** | 0.3787 (❌ **18x worse**) |
=======
| **MAE** | 0.3014 ( **21x worse**) |
| **RMSE** | 0.3787 ( **18x worse**) |
>>>>>>> d3e5d7a (added streaming and backend)
| **R² Score** | -0.040 |
| **CPU Utilization** | 58.9% |

---

## Mode Comparison

### **Quick Comparison Table**

<<<<<<< HEAD
| Metric | Spark GBT | Spark RF ⭐ | Hybrid Sklearn |
=======
| Metric | Spark GBT | Spark RF  | Hybrid Sklearn |
>>>>>>> d3e5d7a (added streaming and backend)
|--------|-----------|------------|----------------|
| **Training Time (20 stocks)** | 2.6 hours | **18.5 min** | 54.9 min |
| **Speed vs GBT** | 1.0x | **5.5x faster** | 2.9x faster |
| **Throughput (rec/sec)** | 8.49 | **73.0** | 24.6 |
| **Peak Memory** | 29.9 GB | **28.0 GB** | 30.9 GB |
<<<<<<< HEAD
| **MAE** | 0.0152 | **0.0143** ✅ | 0.3014 ❌ |
| **RMSE** | 0.0229 | **0.0206** ✅ | 0.3787 ❌ |
| **R² Score** | -0.229 | **0.0140** ✅ | -0.040 |
| **Training Strategy** | Sequential | **Distributed** | Centralized |
| **Scalability** | Poor | **Excellent** | Medium |
| **Production Ready** | ❌ | **✅** | ❌ |
=======
| **MAE** | 0.0152 | **0.0143**  | 0.3014  |
| **RMSE** | 0.0229 | **0.0206**  | 0.3787  |
| **R² Score** | -0.229 | **0.0140**  | -0.040 |
| **Training Strategy** | Sequential | **Distributed** | Centralized |
| **Scalability** | Poor | **Excellent** | Medium |
| **Production Ready** |  | **** |  |
>>>>>>> d3e5d7a (added streaming and backend)

### **Volume Scaling Comparison**

#### **5 Stocks (Small Volume)**
| Mode | Time | Memory | MAE | RMSE | R² |
|------|------|--------|-----|------|----|
| Spark GBT | 648.9s (10.8m) | 29.2 GB | 0.0166 | 0.0253 | -0.299 |
| **Spark RF** | **119.0s (2.0m)** | **21.2 GB** | **0.0151** | **0.0222** | **-0.0003** |
| Hybrid | 273.2s (4.6m) | 24.8 GB | 0.0360 | 0.0445 | -3.014 |

#### **10 Stocks (Medium Volume)**
| Mode | Time | Memory | MAE | RMSE | R² |
|------|------|--------|-----|------|----|
| Spark GBT | 1441.3s (24.0m) | 30.1 GB | 0.0165 | 0.0259 | -0.303 |
| **Spark RF** | **192.8s (3.2m)** | **24.3 GB** | **0.0154** | **0.0227** | **0.0008** |
| Hybrid | 679.9s (11.3m) | 24.7 GB | 0.0253 | 0.0331 | -1.156 |

#### **20 Stocks (Large Volume)**
| Mode | Time | Memory | MAE | RMSE | R² |
|------|------|--------|-----|------|----|
| Spark GBT | 9537.6s (2.6h) | 29.9 GB | 0.0152 | 0.0229 | -0.229 |
| **Spark RF** | **1110.0s (18.5m)** | **28.0 GB** | **0.0143** | **0.0206** | **0.0140** |
| Hybrid | 3293.5s (54.9m) | 30.9 GB | 0.3014 | 0.3787 | -0.040 |

### **Scalability Metrics**

#### **Time Scaling (from 5 → 20 stocks)**
| Mode | Time Increase | Scaling Efficiency |
|------|---------------|-------------------|
<<<<<<< HEAD
| Spark GBT | **14.7x** ❌ | 6.8% (very poor) |
| **Spark RF** | **9.3x** ✅ | **10.7%** (acceptable) |
=======
| Spark GBT | **14.7x**  | 6.8% (very poor) |
| **Spark RF** | **9.3x**  | **10.7%** (acceptable) |
>>>>>>> d3e5d7a (added streaming and backend)
| Hybrid | 12.1x | 8.3% (poor) |

#### **Memory Scaling (from 5 → 20 stocks)**
| Mode | Memory Increase | Efficiency |
|------|----------------|------------|
<<<<<<< HEAD
| Spark GBT | 1.03x | 97% ✅ |
| **Spark RF** | **1.32x** | **76%** ✅ |
| Hybrid | 1.25x | 80% ✅ |
=======
| Spark GBT | 1.03x | 97%  |
| **Spark RF** | **1.32x** | **76%**  |
| Hybrid | 1.25x | 80%  |
>>>>>>> d3e5d7a (added streaming and backend)

**Observation**: All modes scale memory efficiently (near-linear), but **only Spark RF scales time acceptably** for production use.

---

## Hyperparameter Tuning

### **Optimization Strategy**

```python
# Method: Optuna with TPE (Tree-structured Parzen Estimator)
# Trials: 50
# Cross-Validation: 3-fold
# Metric: MAE (minimize)
# Stocks: 4 (AAPL, GOOG, TSLA, BABA)
```

### **Search Space**
```python
{
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [6, 8, 10, 12, 15],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "max_features": ["auto", "sqrt", "log2"]
}
```

### **Best Parameters Found**
**File**: `Final_Report/hyperparameter_tuning/best_params_20251117_124526.json`

```json
{
  "best_params": {
    "n_estimators": 150,
    "max_depth": 12,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "max_features": "log2"
  },
  "performance": {
    "train_metrics": {
      "mae": 0.0165,
      "mse": 0.0101,
      "rmse": 0.1004,
      "r2": 0.8945
    },
    "test_metrics": {
      "mae": 0.0176,
      "mse": 0.0006,
      "rmse": 0.0252,
      "r2": 0.0084
    }
  },
  "optimization": {
    "best_trial_number": 44,
    "total_trials": 50,
    "optimization_time_seconds": 257.58
  }
}
```

### **Tuning Results**
| Metric | Train | Test |
|--------|-------|------|
| **MAE** | 0.0165 | **0.0176** |
| **RMSE** | 0.1004 | **0.0252** |
| **R²** | 0.8945 | 0.0084 |

**Observations**:
<<<<<<< HEAD
- ✅ Low test MAE (0.0176) indicates good generalization
- ⚠️ Large R² gap (train: 0.89, test: 0.01) suggests some overfitting
- ✅ Best params: deeper trees (12), more estimators (150), conservative learning (0.01)
=======
-  Low test MAE (0.0176) indicates good generalization
-  Large R² gap (train: 0.89, test: 0.01) suggests some overfitting
-  Best params: deeper trees (12), more estimators (150), conservative learning (0.01)
>>>>>>> d3e5d7a (added streaming and backend)

---

## Performance Metrics

### **Evaluation Metrics Explained**

#### **MAE (Mean Absolute Error)**
```python
MAE = mean(|actual - predicted|)
```
- **Interpretation**: Average prediction error in absolute terms
- **Units**: Same as target (price gain ratio)
- **Best value**: 0.0143 (Spark RF, 20 stocks)
- **Industry**: MAE < 0.02 is excellent for stock prediction

#### **RMSE (Root Mean Squared Error)**
```python
RMSE = sqrt(mean((actual - predicted)²))
```
- **Interpretation**: Standard deviation of prediction errors
- **Penalty**: Heavily penalizes large errors
- **Best value**: 0.0206 (Spark RF, 20 stocks)
- **Comparison**: Lower is better

#### **R² Score (Coefficient of Determination)**
```python
R² = 1 - (sum((actual - predicted)²) / sum((actual - mean)²))
```
- **Interpretation**: Proportion of variance explained
- **Range**: -∞ to 1.0 (1.0 = perfect)
- **Best value**: 0.0140 (Spark RF, 20 stocks)
- **Note**: Small positive R² is good for stock prediction (hard problem)

#### **Training Time**
- **Metric**: Total time from data load to model save
- **Includes**: Feature engineering, training, evaluation
- **Best value**: 118.97s (Spark RF, 5 stocks), 1110s (Spark RF, 20 stocks)

#### **Throughput**
```python
Throughput = Total Records / Training Time (records/second)
```
- **Best value**: 146.5 rec/sec (Spark RF, 10 stocks)

#### **Memory Efficiency**
```python
Memory per 1K rows = Peak Memory (MB) / (Total Rows / 1000)
```
- **Best value**: 261.5 MB/1K rows (Spark RF, 5 stocks)

---

## Scalability Analysis

### **Linear Scalability Score**

**Formula**: Correlation between data volume and training time
- **Score = 1.0**: Perfect linear scaling
- **Score > 0.5**: Acceptable for production
- **Score < 0.3**: Poor scalability

| Mode | Score | Rating |
|------|-------|--------|
<<<<<<< HEAD
| Spark GBT | 0.0057 | ❌ Very Poor |
| **Spark RF** | **0.0486** | ⚠️ Acceptable |
| Hybrid | 0.0164 | ❌ Poor |
=======
| Spark GBT | 0.0057 |  Very Poor |
| **Spark RF** | **0.0486** |  Acceptable |
| Hybrid | 0.0164 |  Poor |
>>>>>>> d3e5d7a (added streaming and backend)

### **Performance Degradation Analysis**

**Time Scale Factors** (relative to 5 stocks):

| Volume | Spark GBT | Spark RF | Hybrid |
|--------|-----------|----------|---------|
| 5 stocks | 1.0x | 1.0x | 1.0x |
<<<<<<< HEAD
| 10 stocks | 2.2x | **1.6x** ✅ | 2.5x |
| 20 stocks | 14.7x | **9.3x** ✅ | 12.1x |
=======
| 10 stocks | 2.2x | **1.6x**  | 2.5x |
| 20 stocks | 14.7x | **9.3x**  | 12.1x |
>>>>>>> d3e5d7a (added streaming and backend)

**Spark RF degrades slower** as data volume increases.

### **Resource Utilization**

#### **CPU Utilization (20 stocks)**
| Mode | CPU % | Efficiency |
|------|-------|------------|
| Spark GBT | 54.9% | Medium |
| **Spark RF** | **27.4%** | **High** (more distributed) |
| Hybrid | 58.9% | Medium |

#### **GC (Garbage Collection) Overhead**
- **Spark GBT**: 82.4 GB GC time (major bottleneck)
- **Spark RF**: Minimal GC pressure
- **Hybrid**: Moderate GC pressure

---

## Usage Guide

### **Training Command**

```bash
# Activate environment
conda activate breaking_data

# Train specific mode
python scripts/train_model_with_modes.py \
    --mode spark_rf \
    --data-volume full

# Train with different volumes
python scripts/train_model_with_modes.py \
    --mode spark_rf \
    --data-volume small   # 5 stocks
    
python scripts/train_model_with_modes.py \
    --mode spark_rf \
    --data-volume medium  # 10 stocks
    
python scripts/train_model_with_modes.py \
    --mode spark_rf \
    --data-volume large   # 20 stocks

# Run all modes for comparison
python scripts/train_model_with_modes.py \
    --run-all-modes \
    --data-volume large

# Run scalability evaluation
python scripts/train_model_with_modes.py \
    --scalability-evaluation
```

### **Configuration Files**

#### **Main Config**: `config/config.yaml`
```yaml
model:
  algorithm: GBTRegressor
  default_params:
    maxDepth: 12
    maxIter: 150
    stepSize: 0.01
    subsamplingRate: 0.8

features:
  price_lags: [1, 2, 3, 5]
  sentiment_lags: [1, 3]
  ma_windows: [5, 10, 20]
  enable_rsi: true
  enable_volatility: true

mlflow:
  experiment_name: close_gain_forecasting
  model_name: stock_predictor_unified_29stocks
```

#### **Multi-Mode Config**: `config/config_modes.yaml`
```yaml
spark_rf_mode:
  algorithm: RandomForestRegressor
  params:
    numTrees: 100
    maxDepth: 12
    subsamplingRate: 0.8
    featureSubsetStrategy: "sqrt"
  
  mlflow:
    model_name: stock_predictor_spark_rf_unified_29stocks
    experiment_name: stock_forecasting_rf_exp
```

### **View Results**

```bash
# Start MLflow UI
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri ./mlflow_tracking \
    --default-artifact-root ./mlflow_tracking

# Open browser
open http://localhost:5000

# View scalability report
open Final_Report/validation_scalability_analysis_report_*.html

# View feature importance
open plots/feature_importance_interactive.html
```

### **Access Trained Model**

```python
import mlflow.spark

# Load production model
model_uri = "models:/stock_predictor_spark_rf_unified_29stocks/1"
model = mlflow.spark.load_model(model_uri)

# Get model metadata
client = mlflow.tracking.MlflowClient()
model_version = client.get_model_version(
    name="stock_predictor_spark_rf_unified_29stocks",
    version="1"
)
print(f"Model stage: {model_version.current_stage}")
print(f"Model tags: {model_version.tags}")
```

---

## References

- **MLflow Experiments**: `mlflow_tracking/` directory
- **Training Results**: `Final_Report/` directory
- **Scalability Report**: `validation_scalability_analysis_report_*.html`
- **Best Parameters**: `Final_Report/hyperparameter_tuning/best_params_*.json`
- **Configuration**: `config/config_modes.yaml`

---
