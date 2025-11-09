# KMRF Batch Training Guide

## Overview

The `train_KMRF.ipynb` notebook enables batch training of KMRF models across multiple assets, similar to the `fit_KAMA_MSR.ipynb` workflow but for the KMRF (Kalman-Filter Market Regime Factor) model.

## Quick Start

1. Open `train_KMRF.ipynb`
2. Configure the batch training parameters in the Configuration section
3. Run all cells to train models for all selected asset classes
4. Models are automatically saved with organized directory structure

## Configuration Options

### Asset Selection
```python
ASSET_CLASSES_TO_TRAIN = ['us_equity']  # Which asset classes to train
```

Available asset classes:
- `'us_equity'` - US equity ETFs and indices
- `'us_treasury'` - US Treasury bond ETFs
- `'int_equity'` - International equity ETFs
- `'commodity'` - Commodity futures

### Model Parameters
```python
CLASSIFICATION_TYPE = 'adapted'  # 'original' (4 regimes) or 'adapted' (3 regimes)
USE_BORUTA = True  # Enable Boruta feature selection (slow but effective)
BORUTA_MAX_ITER = 100  # Max iterations for Boruta
INCLUDE_MACRO = True  # Include macroeconomic features
```

### Time Periods
```python
END_DATE = '20190101'  # Training data end date
VALIDATION_START = '2019-04-01'
VALIDATION_END = '2019-09-30'
TEST_START = '2020-01-01'
```

### Error Handling
```python
CONTINUE_ON_ERROR = True  # Continue if one asset fails
SAVE_SUCCESSFUL_ONLY = True  # Only save successful models
```

## Features

### 1. Automated Batch Processing
- Processes multiple assets sequentially
- Progress tracking with asset counter
- Estimated time display

### 2. Error Handling
- Continue training even if one asset fails
- Detailed error logging
- Full traceback capture for debugging

### 3. Model Persistence
- Automatic saving after successful training
- Organized directory structure by asset class and classification type
- Models saved with descriptive names

### 4. Training Summary
- JSON summary file with training metadata
- Success/failure status for each asset
- Training duration and feature counts
- Error messages for failed models

### 5. Statistics and Reporting
- Training time per asset
- Feature selection statistics
- Success rate calculation
- Model size information

## Output Structure

```
saved_models/KMRF/
├── adapted_labels/
│   ├── us_equity/
│   │   ├── KMRF_SPDR-S&P-500-ETF_20190101_boruta.pkl
│   │   ├── KMRF_Vanguard-S&P-500-ETF_20190101_boruta.pkl
│   │   ├── KMRF_Invesco-QQQ-Trust_20190101_boruta.pkl
│   │   └── training_summary_20190101.json
│   ├── us_treasury/
│   ├── int_equity/
│   └── commodity/
└── original_labels/
    └── ...
```

## Training Summary JSON

Each training run produces a JSON summary file containing:

```json
{
  "SPDR S&P 500 ETF": {
    "status": "success",
    "model_path": "saved_models/KMRF/adapted_labels/us_equity/KMRF_SPDR-S&P-500-ETF_20190101_boruta.pkl",
    "n_features": 150,
    "n_selected_features": 45,
    "n_train_samples": 5234,
    "n_test_samples": 252,
    "boruta_used": true,
    "duration_seconds": 1845.2,
    "timestamp": "2025-11-04T10:30:45"
  },
  "Failed Asset": {
    "status": "failed",
    "error": "Error message here",
    "error_trace": "Full traceback...",
    "duration_seconds": 120.5,
    "timestamp": "2025-11-04T10:32:45"
  }
}
```