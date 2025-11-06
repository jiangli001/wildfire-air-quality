import pandas as pd
import numpy as np


def create_multivariate_windows(
    df,
    window_size=24,
    forecast_horizon=24,
    feature_cols=None,
    stride=1,
):
    """
    Create 2D sliding windows for multivariate time series modeling.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with columns: site, date, start_hour, and feature columns
    window_size : int
        Size of the sliding window (default: 24 for hours in a day)
    forecast_horizon : int
        Number of time steps to predict ahead (default: 2)
    feature_cols : list
        List of column names to include as features
    stride : int
        Step size for sliding the window (default: 1)

    Returns:
    --------
    X : np.ndarray
        Array of shape (n_windows, window_size, n_features) containing input sequences
    y : np.ndarray
        Array of shape (n_windows, forecast_horizon, n_features) containing target sequences
    metadata : pd.DataFrame
        Dataframe containing metadata for each window
    """

    if feature_cols is None:
        feature_cols = []
    # Verify all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    # Sort data by site, date, and start_hour
    df = df.sort_values(["site", "date", "start_hour"]).reset_index(drop=True)

    X = []
    y = []
    metadata = []

    # Process each site separately to ensure no cross-site windows
    for site in df["site"].unique():
        site_data = df[df["site"] == site].reset_index(drop=True)

        # Calculate the maximum valid starting index
        # We need window_size + forecast_horizon rows to create a valid sample
        max_start_idx = len(site_data) - window_size - forecast_horizon + 1

        # Create sliding windows with specified stride
        for i in range(0, max_start_idx, stride):
            # Extract input window (window_size × n_features)
            window_end_idx = i + window_size
            X_window = site_data[feature_cols].iloc[i:window_end_idx].values

            # Extract target window (forecast_horizon × n_features)
            target_start_idx = window_end_idx
            target_end_idx = target_start_idx + forecast_horizon
            y_window = (
                site_data[feature_cols].iloc[target_start_idx:target_end_idx].values
            )

            # Verify we got the correct shapes
            if (
                X_window.shape[0] == window_size
                and y_window.shape[0] == forecast_horizon
            ):
                X.append(X_window)
                y.append(y_window)

                # Store metadata
                metadata.append(
                    {
                        "site": site,
                        "window_start_idx": i,
                        "X_start_date": site_data["date"].iloc[i],
                        "X_start_hour": site_data["start_hour"].iloc[i],
                        "X_end_date": site_data["date"].iloc[window_end_idx - 1],
                        "X_end_hour": site_data["start_hour"].iloc[window_end_idx - 1],
                        "y_start_date": site_data["date"].iloc[target_start_idx],
                        "y_start_hour": site_data["start_hour"].iloc[target_start_idx],
                        "y_end_date": site_data["date"].iloc[target_end_idx - 1],
                        "y_end_hour": site_data["start_hour"].iloc[target_end_idx - 1],
                    }
                )

    # Convert to numpy arrays
    X = np.array(X)  # Shape: (n_samples, window_size, n_features)
    y = np.array(y)  # Shape: (n_samples, forecast_horizon, n_features)
    metadata_df = pd.DataFrame(metadata)

    return X, y, metadata_df


def split_train_test_by_site(X, y, metadata, test_size=0.2, random_state=42):
    """
    Split data into train and test sets, keeping all windows from each site together.

    Parameters:
    -----------
    X : np.ndarray
        Input windows
    y : np.ndarray
        Target windows
    metadata : pd.DataFrame
        Metadata with site information
    test_size : float
        Proportion of sites to use for testing
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_train, X_test, y_train, y_test, metadata_train, metadata_test
    """
    np.random.seed(random_state)

    # Get unique sites
    unique_sites = metadata["site"].unique()
    n_test_sites = max(1, int(len(unique_sites) * test_size))

    # Randomly select test sites
    test_sites = np.random.choice(unique_sites, size=n_test_sites, replace=False)

    # Create train/test masks
    test_mask = metadata["site"].isin(test_sites)
    train_mask = ~test_mask

    return (
        X[train_mask],
        X[test_mask],
        y[train_mask],
        y[test_mask],
        metadata[train_mask].reset_index(drop=True),
        metadata[test_mask].reset_index(drop=True),
    )


# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("./combined_pm25_data_sample.csv")

    # Define your feature columns
    feature_cols = ["value"]

    print("=" * 70)
    print("Creating Multivariate Sliding Windows")
    print("=" * 70)

    # Create windows
    X, y, metadata = create_multivariate_windows(
        df,
        window_size=24,  # 24 hours of input
        forecast_horizon=2,  # Predict next 2 hours
        feature_cols=feature_cols,
        stride=1,  # Slide by 1 hour each time
    )

    print(f"\nInput (X) shape: {X.shape}")
    print(f"  - {X.shape[0]} samples")
    print(f"  - {X.shape[1]} time steps (hours)")
    print(f"  - {X.shape[2]} features")

    print(f"\nTarget (y) shape: {y.shape}")
    print(f"  - {y.shape[0]} samples")
    print(f"  - {y.shape[1]} time steps (hours to predict)")
    print(f"  - {y.shape[2]} features")

    print(f"\nFeatures: {feature_cols}")

    print("\n" + "=" * 70)
    print("First Sample Example")
    print("=" * 70)
    print("\nFirst input window (X[0]):")
    print(f"Shape: {X[0].shape} (24 hours × {len(feature_cols)} features)")
    print("First 5 rows:")
    print(X[0][:5])

    print("\nFirst target window (y[0]):")
    print(f"Shape: {y[0].shape} (2 hours × {len(feature_cols)} features)")
    print(y[0])

    print("\n" + "=" * 70)
    print("Metadata for First Few Windows")
    print("=" * 70)
    print(metadata.head(10))

    print("\n" + "=" * 70)
    print("Windows per Site")
    print("=" * 70)
    print(metadata["site"].value_counts().sort_index())

    # Optional: Split into train/test
    print("\n" + "=" * 70)
    print("Train/Test Split (by site)")
    print("=" * 70)
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_train_test_by_site(
        X, y, metadata, test_size=0.2
    )

    print(f"\nTraining set:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Sites: {sorted(meta_train['site'].unique())}")

    print(f"\nTest set:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Sites: {sorted(meta_test['site'].unique())}")
