# PM2.5 Time Series Prediction Model

Deep learning model for predicting PM2.5 air quality levels during wildfire events in California.

## Model Architecture

**LSTM (Long Short-Term Memory)** neural network chosen for the following reasons:

1. **Sequential Nature**: PM2.5 readings are time-dependent with temporal patterns
2. **Long-term Dependencies**: Wildfire smoke can persist and evolve over days
3. **Variable Dynamics**: LSTM handles both gradual changes and sudden spikes from fire events
4. **Proven Track Record**: Established success in air quality forecasting

### Architecture Details

- **Input**: 24-hour sequence of PM2.5 readings
- **Output**: 24-hour forecast of future PM2.5 levels
- **Hidden Layers**: 2 LSTM layers with 128 hidden units each
- **Regularization**: Dropout (0.2) to prevent overfitting
- **Normalization**: StandardScaler (z-score normalization)

```
Input (24 hours × 1 feature)
    ↓
LSTM Layer 1 (128 units)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (128 units)
    ↓
Dropout (0.2)
    ↓
Fully Connected Layer
    ↓
Output (24 hours × 1 feature)
```

## Data Preparation

The model uses sliding windows created by `matrix.py`:
- **Window Size**: 24 hours (input sequence)
- **Forecast Horizon**: 24 hours (prediction range)
- **Stride**: 1 hour (windows overlap)
- **Site Handling**: Each monitoring site processed independently
- **Train/Test Split**: By site (prevents data leakage)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_dl.txt
```

For CUDA support (if you have an NVIDIA GPU):
```bash
# Check CUDA version first
nvidia-smi

# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## Usage

### Training

```bash
cd src
python train_pm25_model.py
```

**Configuration** (edit in `train_pm25_model.py`):
```python
CONFIG = {
    # Data
    'window_size': 24,           # Hours of historical data
    'forecast_horizon': 24,      # Hours to predict
    'test_size': 0.2,            # 20% sites for testing

    # Model
    'hidden_size': 128,          # LSTM hidden units
    'num_layers': 2,             # LSTM layers
    'dropout': 0.2,              # Dropout rate

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'early_stopping_patience': 15,
}
```

**Outputs** (saved to `models/`):
- `pm25_lstm_best.pth` - Model checkpoint
- `pm25_lstm_scaler.pkl` - Data normalization scaler
- `pm25_lstm_results.json` - Metrics and configuration
- `pm25_lstm_training_history.png` - Loss curves
- `pm25_lstm_predictions.png` - Sample predictions

### Inference

```bash
cd src
python predict_pm25.py
```

**Programmatic Usage**:

```python
from predict_pm25 import load_model_artifacts, predict_from_sequence

# Load trained model
model, scaler, config, device = load_model_artifacts('../models')

# Option 1: Predict from CSV file
from predict_pm25 import predict_from_csv
predictions_df, y_pred, y_actual, metadata = predict_from_csv(
    'data.csv', model, scaler, config, device
)

# Option 2: Predict from array
pm25_sequence = [45.2, 52.1, 48.3, ...]  # 24 hourly values
predictions = predict_from_sequence(
    pm25_sequence, model, scaler, config, device
)
```

## Evaluation Metrics

The model is evaluated using:

- **RMSE** (Root Mean Squared Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average absolute error
- **R²** (R-squared): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Relative error percentage

## Model Performance

Expected performance (will vary based on data):
- RMSE: ~10-20 µg/m³
- MAE: ~8-15 µg/m³
- R²: 0.70-0.85

Performance varies by:
- Forecast horizon (earlier hours more accurate)
- Fire intensity (extreme events harder to predict)
- Site characteristics (terrain, weather patterns)

## File Structure

```
wildfire-air-quality/
├── data/
│   └── final_merged_data.csv          # Input data
├── src/
│   ├── matrix.py                       # Data preparation utilities
│   ├── train_pm25_model.py            # Training script
│   └── predict_pm25.py                # Inference script
├── models/                             # Saved models (created during training)
│   ├── pm25_lstm_best.pth
│   ├── pm25_lstm_scaler.pkl
│   └── pm25_lstm_results.json
├── predictions/                        # Inference outputs (created during inference)
│   ├── predictions.csv
│   └── predictions_site_*.png
├── requirements_dl.txt                 # Python dependencies
└── MODEL_README.md                     # This file
```

## Extending the Model

### Adding More Features

Currently uses only PM2.5. To include weather variables:

1. Update `feature_cols` in training script:
```python
feature_cols = ['pm25', 'temperature', 'humidity', 'wind_speed']
```

2. Model will automatically adjust input size
3. Retrain from scratch with new features

### Hyperparameter Tuning

Key parameters to experiment with:
- `hidden_size`: 64, 128, 256 (larger = more capacity)
- `num_layers`: 1, 2, 3 (deeper = more complex patterns)
- `window_size`: 12, 24, 48 (longer = more context)
- `forecast_horizon`: 6, 12, 24, 48 (shorter = easier)
- `learning_rate`: 0.0001, 0.001, 0.01

### Alternative Architectures

Consider these if LSTM doesn't perform well:

1. **GRU** (simpler than LSTM):
   - Replace `nn.LSTM` with `nn.GRU`
   - Faster training, similar performance

2. **Bidirectional LSTM**:
   - Set `bidirectional=True` in LSTM
   - Better for fixed sequences, not real-time

3. **CNN-LSTM Hybrid**:
   - Add Conv1D layers before LSTM
   - Extract local patterns, then temporal

4. **Transformer**:
   - Use attention mechanism
   - Better for very long sequences

## Training Tips

1. **Monitor Training**:
   - Watch for overfitting (val_loss >> train_loss)
   - Use early stopping to prevent overfitting
   - Reduce learning rate if loss plateaus

2. **Data Quality**:
   - Handle missing values carefully
   - Remove outliers if necessary
   - Ensure consistent time intervals

3. **Computational Resources**:
   - Use GPU for faster training
   - Larger batch sizes on GPU (32-128)
   - Smaller batch sizes on CPU (8-32)

4. **Experiment Tracking**:
   - Results saved automatically to JSON
   - Compare different runs by timestamp
   - Keep notes on configuration changes

## Troubleshooting

**CUDA Out of Memory**:
- Reduce `batch_size`
- Reduce `hidden_size` or `num_layers`
- Clear cache: `torch.cuda.empty_cache()`

**Poor Performance**:
- Check data quality and normalization
- Try different hyperparameters
- Increase model capacity (hidden_size, num_layers)
- Add more training data if available

**Training Too Slow**:
- Ensure GPU is being used
- Increase `batch_size` (if memory allows)
- Reduce `window_size` or `forecast_horizon`

## References

- LSTM for Air Quality: [Zhang et al., 2020]
- Time Series Forecasting: [Hewamalage et al., 2021]
- Wildfire Smoke Prediction: [Reid et al., 2016]

## Future Improvements

- [ ] Add attention mechanism for interpretability
- [ ] Incorporate spatial features (nearby sites)
- [ ] Multi-task learning (predict multiple pollutants)
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Real-time inference pipeline
- [ ] Model compression for edge deployment
