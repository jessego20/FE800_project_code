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

## Typical Training Times

**Without Boruta:**
- Per asset: 2-5 minutes
- 10 assets: 20-50 minutes

**With Boruta (recommended):**
- Per asset: 30-60 minutes
- 10 assets: 5-10 hours

**Tips for faster training:**
- Train overnight for large batches with Boruta
- Use `USE_BORUTA = False` for quick testing
- Reduce `BORUTA_MAX_ITER` for faster (but less thorough) feature selection

## Asset Lists

### US Equity (19 assets)
- Major index ETFs: SPY, VOO, IVV, QQQ, IWM, VTI
- Sector ETFs: XLE, XLF, XLU, XLI, XLV, XLK, XLB, XLY, XLP
- Style ETFs: IVW, IVE, IWF, IWD

### US Treasury (3 assets)
- BIL, SHY, IEF

### International Equity (8 assets)
- VXUS, VEA, VWO, VGK, VPL, FXI, EWJ, INDA

### Commodity (5 assets)
- Gold, Silver, Crude Oil, Natural Gas, Copper

**Total: 35 assets across all classes**

## Usage Examples

### Train All US Equity Assets
```python
ASSET_CLASSES_TO_TRAIN = ['us_equity']
USE_BORUTA = True
# Run notebook
```

### Train Multiple Asset Classes
```python
ASSET_CLASSES_TO_TRAIN = ['us_equity', 'us_treasury', 'int_equity']
USE_BORUTA = True
# Run notebook
```

### Quick Training (No Boruta)
```python
ASSET_CLASSES_TO_TRAIN = ['us_equity']
USE_BORUTA = False
# Run notebook - much faster!
```

### Train Specific Assets Only
Edit the asset lists in the "Asset Lists" section:
```python
us_equity_assets = [
    'SPDR S&P 500 ETF',
    'Invesco QQQ Trust',
    # Add/remove assets as needed
]
```

## Workflow Comparison

### Single Asset (test_kmrf.ipynb)
1. Configure one asset
2. Load data and features
3. Load labels
4. Prepare training data
5. Train model
6. Save model
7. Generate predictions

### Batch Training (train_KMRF.ipynb)
1. Configure asset classes and parameters
2. For each asset:
   - Initialize model
   - Load data and features
   - Load labels
   - Prepare training data
   - Train model
   - Save model
3. Generate training summary
4. Display statistics

## After Training

### Load a Trained Model
```python
import kmrf

model = kmrf.KMRF.load_model(
    'saved_models/KMRF/adapted_labels/us_equity/KMRF_SPDR-S&P-500-ETF_20190101_boruta.pkl'
)

predictions = model.predict(model.X_test, return_proba=True)
```

### Use in test_kmrf.ipynb
```python
LOAD_EXISTING_MODEL = True
MODEL_PATH = 'saved_models/KMRF/adapted_labels/us_equity/KMRF_SPDR-S&P-500-ETF_20190101_boruta.pkl'
# Run notebook for detailed analysis and visualization
```

## Troubleshooting

### Common Issues

**1. Memory Error**
- Train fewer assets at once
- Close other applications
- Restart kernel between runs

**2. Asset Data Not Found**
- Check that asset exists in ready data files
- Verify asset name matches exactly
- Check `data/ready/` directory

**3. KAMA+MSR Labels Missing**
- Ensure KAMA+MSR models are fitted first using `fit_KAMA_MSR.ipynb`
- Check `saved_models/KAMA_MSR/` directory

**4. Training Takes Too Long**
- Set `USE_BORUTA = False` for testing
- Reduce `BORUTA_MAX_ITER` to 50
- Train smaller batches

**5. Some Assets Fail**
- Check training summary JSON for error details
- Review error traces in notebook output
- Verify data availability for failed assets

## Best Practices

1. **Start Small**: Test with 2-3 assets before running full batch
2. **Use Boruta**: Feature selection significantly improves model performance
3. **Monitor Progress**: Check training summary after each run
4. **Save Intermediate Results**: Training summary JSON allows resume if interrupted
5. **Version Control**: Include date in END_DATE for model versioning
6. **Document**: Add notes about training configuration in summary JSON

## Advanced Usage

### Custom Asset Lists
Add your own assets by editing the asset lists:
```python
custom_assets = [
    'Your Asset Name 1',
    'Your Asset Name 2',
]

ASSET_GROUPS['custom'] = custom_assets
ASSET_CLASSES_TO_TRAIN = ['custom']
```

### Parallel Training
For very large batches, consider splitting into multiple notebooks:
- Notebook 1: Train assets 1-10
- Notebook 2: Train assets 11-20
- etc.

### Re-train Failed Models
Extract failed assets from training summary and re-run:
```python
# Load training summary
import json
with open('saved_models/KMRF/adapted_labels/us_equity/training_summary_20190101.json') as f:
    summary = json.load(f)

# Get failed assets
failed_assets = [name for name, result in summary.items() if result['status'] == 'failed']

# Re-train
us_equity_assets = failed_assets
ASSET_CLASSES_TO_TRAIN = ['us_equity']
```

## Performance Tips

1. **Boruta Settings**: Adjust `BORUTA_MAX_ITER` based on dataset size
   - Small datasets (< 1000 samples): 50 iterations
   - Medium datasets (1000-5000): 100 iterations  
   - Large datasets (> 5000): 150 iterations

2. **Macro Features**: If training is very slow, try `INCLUDE_MACRO = False`

3. **Random Seed**: Use consistent seed for reproducibility

4. **Validation Periods**: Shorter validation periods = more training data

## Integration with Analysis Pipeline

```
1. fit_KAMA_MSR.ipynb      → Generate regime labels
2. train_KMRF.ipynb        → Train prediction models (batch)
3. test_kmrf.ipynb         → Analyze individual models
4. Your analysis notebook  → Use predictions for trading strategies
```

## Summary

The `train_KMRF.ipynb` notebook provides an efficient way to train KMRF models across multiple assets with:
- Automated batch processing
- Robust error handling
- Comprehensive logging
- Training statistics
- Easy integration with single-asset analysis

Perfect for building a complete library of trained KMRF models for portfolio-wide regime prediction!
