import pandas as pd
import numpy as np
import torch
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import your model class and data preparation functions
from train_pm25_model import PM25LSTM
from matrix import create_multivariate_windows


def load_model_artifacts(model_dir, model_name='pm25_lstm'):
    """Load trained model, scaler, and configuration."""

    model_dir = Path(model_dir)

    # Load configuration and results
    with open(model_dir / f"{model_name}_results.json", 'r') as f:
        results = json.load(f)

    config = results['config']

    # Load scaler
    scaler = joblib.load(model_dir / f"{model_name}_scaler.pkl")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PM25LSTM(
        input_size=len(config['feature_cols']),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        forecast_horizon=config['forecast_horizon'],
        output_size=len(config['feature_cols'])
    ).to(device)

    # Load model weights
    checkpoint = torch.load(
        model_dir / f"{model_name}_best.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully from {model_dir}")
    print(f"Device: {device}")
    print(f"Test metrics: RMSE={results['test_metrics']['rmse']:.2f}, "
          f"MAE={results['test_metrics']['mae']:.2f}, "
          f"R²={results['test_metrics']['r2']:.4f}")

    return model, scaler, config, device


def predict_from_csv(csv_path, model, scaler, config, device, site_id=None):
    """
    Make predictions from a CSV file.

    Args:
        csv_path: Path to CSV file with PM2.5 data
        model: Trained model
        scaler: Fitted scaler
        config: Model configuration
        device: PyTorch device
        site_id: Optional specific site to predict for (if None, predicts for all sites)

    Returns:
        predictions_df: DataFrame with predictions and metadata
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Filter by site if specified
    if site_id is not None:
        df = df[df['site'] == site_id]
        if len(df) == 0:
            raise ValueError(f"No data found for site {site_id}")

    print(f"Loaded {len(df)} records from {df['site'].nunique()} sites")

    # Drop rows with missing PM2.5
    df = df.dropna(subset=['pm25'])

    # Create windows
    X, y, metadata = create_multivariate_windows(
        df,
        window_size=config['window_size'],
        forecast_horizon=config['forecast_horizon'],
        feature_cols=config['feature_cols'],
        stride=config['stride']
    )

    print(f"Created {len(X)} prediction windows")

    # Normalize
    X_2d = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_2d).reshape(X.shape)

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # Make predictions
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    # Inverse transform predictions
    y_pred_2d = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
    y_pred = scaler.inverse_transform(y_pred_2d).reshape(y_pred_scaled.shape)

    # Also inverse transform actual values if available
    y_2d = y.reshape(-1, y.shape[-1])
    y_actual = scaler.inverse_transform(y_2d).reshape(y.shape)

    # Create results dataframe
    results = []

    for i in range(len(y_pred)):
        for h in range(config['forecast_horizon']):
            results.append({
                'site': metadata.iloc[i]['site'],
                'window_idx': i,
                'input_start_date': metadata.iloc[i]['X_start_date'],
                'input_start_hour': metadata.iloc[i]['X_start_hour'],
                'input_end_date': metadata.iloc[i]['X_end_date'],
                'input_end_hour': metadata.iloc[i]['X_end_hour'],
                'forecast_hour': h + 1,
                'predicted_pm25': y_pred[i, h, 0],
                'actual_pm25': y_actual[i, h, 0] if y is not None else None
            })

    predictions_df = pd.DataFrame(results)

    return predictions_df, y_pred, y_actual, metadata


def predict_from_sequence(pm25_sequence, model, scaler, config, device):
    """
    Make predictions from a PM2.5 sequence.

    Args:
        pm25_sequence: Array or list of PM2.5 values (length = window_size)
        model: Trained model
        scaler: Fitted scaler
        config: Model configuration
        device: PyTorch device

    Returns:
        predictions: Array of predicted PM2.5 values (length = forecast_horizon)
    """

    if len(pm25_sequence) != config['window_size']:
        raise ValueError(
            f"Sequence length ({len(pm25_sequence)}) must match "
            f"window_size ({config['window_size']})"
        )

    # Reshape to (1, window_size, n_features)
    X = np.array(pm25_sequence).reshape(1, -1, 1)

    # Normalize
    X_2d = X.reshape(-1, 1)
    X_scaled = scaler.transform(X_2d).reshape(X.shape)

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # Make prediction
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    # Inverse transform
    y_pred_2d = y_pred_scaled.reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_2d).reshape(y_pred_scaled.shape)

    return y_pred[0, :, 0]  # Return 1D array


def plot_site_predictions(predictions_df, site_id, n_windows=5, save_path=None):
    """Plot predictions for a specific site."""

    site_data = predictions_df[predictions_df['site'] == site_id]

    if len(site_data) == 0:
        print(f"No predictions found for site {site_id}")
        return

    # Get unique windows
    unique_windows = site_data['window_idx'].unique()[:n_windows]

    fig, axes = plt.subplots(len(unique_windows), 1, figsize=(14, 4 * len(unique_windows)))

    if len(unique_windows) == 1:
        axes = [axes]

    for idx, window_idx in enumerate(unique_windows):
        window_data = site_data[site_data['window_idx'] == window_idx].sort_values('forecast_hour')

        axes[idx].plot(
            window_data['forecast_hour'],
            window_data['actual_pm25'],
            label='Actual',
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=6,
            alpha=0.7
        )
        axes[idx].plot(
            window_data['forecast_hour'],
            window_data['predicted_pm25'],
            label='Predicted',
            marker='x',
            linestyle='--',
            linewidth=2,
            markersize=8,
            alpha=0.7
        )

        # Calculate error metrics for this window
        mse = np.mean((window_data['predicted_pm25'] - window_data['actual_pm25']) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(window_data['predicted_pm25'] - window_data['actual_pm25']))

        axes[idx].set_xlabel('Forecast Hour', fontsize=10)
        axes[idx].set_ylabel('PM2.5 (µg/m³)', fontsize=10)
        axes[idx].set_title(
            f"Site {site_id} - Window {window_idx} | "
            f"Input: {window_data.iloc[0]['input_start_date']} "
            f"{window_data.iloc[0]['input_start_hour']}:00 to "
            f"{window_data.iloc[0]['input_end_date']} "
            f"{window_data.iloc[0]['input_end_hour']}:00 | "
            f"RMSE: {rmse:.2f}, MAE: {mae:.2f}",
            fontsize=10
        )
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def example_usage():
    """Example usage of the prediction functions."""

    # Configuration
    MODEL_DIR = '../models'
    DATA_PATH = '../data/final_merged_data.csv'
    OUTPUT_DIR = '../predictions'

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print("=" * 80)
    print("PM2.5 Prediction - Inference")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, scaler, config, device = load_model_artifacts(MODEL_DIR)

    # ============================================================================
    # Example 1: Predict from CSV file
    # ============================================================================
    print("\n" + "=" * 80)
    print("Example 1: Predictions from CSV")
    print("=" * 80)

    predictions_df, y_pred, y_actual, metadata = predict_from_csv(
        DATA_PATH, model, scaler, config, device
    )

    print(f"\nGenerated {len(predictions_df)} predictions")
    print("\nFirst few predictions:")
    print(predictions_df.head(10))

    # Save predictions
    output_file = Path(OUTPUT_DIR) / 'predictions.csv'
    predictions_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

    # Calculate overall metrics
    errors = predictions_df['predicted_pm25'] - predictions_df['actual_pm25']
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    print(f"\nOverall Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    # Plot predictions for each site
    print("\nGenerating plots for each site...")
    for site in predictions_df['site'].unique():
        plot_site_predictions(
            predictions_df,
            site,
            n_windows=3,
            save_path=Path(OUTPUT_DIR) / f'predictions_site_{site}.png'
        )

    # ============================================================================
    # Example 2: Predict from a custom sequence
    # ============================================================================
    print("\n" + "=" * 80)
    print("Example 2: Prediction from custom PM2.5 sequence")
    print("=" * 80)

    # Create a sample sequence (e.g., from the first 24 hours of data)
    df = pd.read_csv(DATA_PATH)
    sample_sequence = df['pm25'].dropna().values[:config['window_size']]

    print(f"\nInput sequence (first {config['window_size']} hours):")
    print(sample_sequence)

    # Make prediction
    predictions = predict_from_sequence(sample_sequence, model, scaler, config, device)

    print(f"\nPredicted PM2.5 for next {config['forecast_horizon']} hours:")
    print(predictions)

    # Plot
    plt.figure(figsize=(12, 6))

    # Plot input sequence
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(sample_sequence) + 1), sample_sequence, marker='o')
    plt.xlabel('Hour')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.title('Input Sequence')
    plt.grid(True, alpha=0.3)

    # Plot predictions
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(predictions) + 1), predictions, marker='x', color='red')
    plt.xlabel('Forecast Hour')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.title('Predictions')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'custom_sequence_prediction.png', dpi=300)
    print(f"\nPlot saved to {Path(OUTPUT_DIR) / 'custom_sequence_prediction.png'}")
    plt.close()

    print("\n" + "=" * 80)
    print("Inference completed!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
