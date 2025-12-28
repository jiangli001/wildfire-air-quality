import numpy as np
from tqdm.auto import tqdm

def create_multivariate_windows(
    df,
    window_size=24,
    forecast_horizon=24,
    feature_cols=None,
    stride=1,
):
    """
    Create 2D sliding windows for multivariate time series modeling.
    Optimized for performance using NumPy operations.
    
    IMPORTANT: Windows are only created within contiguous time segments.
    Gaps in time (e.g., 8/2 to 10/1) are treated as segment boundaries.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with columns: site, date, start_hour, and feature columns
    window_size : int
        Size of the sliding window (default: 24 for hours in a day)
    forecast_horizon : int
        Number of time steps to predict ahead (default: 24)
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
    unique_sites = df["site"].unique()
    print(f"Processing {len(unique_sites)} sites to create sliding windows...")

    total_segments = 0
    skipped_segments = 0

    for site in tqdm(unique_sites, desc="Sites"):
        site_data = df[df["site"] == site].reset_index(drop=True)
        
        # ================================================================
        # NEW: Detect contiguous segments within each site
        # ================================================================
        # Create datetime for each row
        site_data = site_data.copy()
        site_data['datetime'] = pd.to_datetime(site_data['date']) + pd.to_timedelta(site_data['start_hour'], unit='h')
        
        # Calculate time difference between consecutive rows
        time_diff = site_data['datetime'].diff()
        
        # A new segment starts where gap > 1 hour (or first row)
        # Mark segment boundaries
        segment_starts = (time_diff != pd.Timedelta(hours=1)) | (time_diff.isna())
        
        # Assign segment IDs
        site_data['segment_id'] = segment_starts.cumsum()
        
        # Process each contiguous segment separately
        for segment_id in site_data['segment_id'].unique():
            segment_data = site_data[site_data['segment_id'] == segment_id].reset_index(drop=True)
            total_segments += 1
            
            # Convert to numpy arrays for fast slicing
            data_values = segment_data[feature_cols].values
            dates = segment_data["date"].values
            hours = segment_data["start_hour"].values

            # Calculate the maximum valid starting index
            max_start_idx = len(segment_data) - window_size - forecast_horizon + 1

            if max_start_idx <= 0:
                skipped_segments += 1
                continue

            # Create sliding windows with specified stride
            for i in range(0, max_start_idx, stride):
                window_end_idx = i + window_size
                target_start_idx = window_end_idx
                target_end_idx = target_start_idx + forecast_horizon

                # Extract windows using numpy slicing
                X_window = data_values[i:window_end_idx]
                y_window = data_values[target_start_idx:target_end_idx]

                X.append(X_window)
                y.append(y_window)

                # Store metadata
                metadata.append(
                    {
                        "site": site,
                        "segment_id": segment_id,
                        "window_start_idx": i,
                        "X_start_date": dates[i],
                        "X_start_hour": hours[i],
                        "X_end_date": dates[window_end_idx - 1],
                        "X_end_hour": hours[window_end_idx - 1],
                        "y_start_date": dates[target_start_idx],
                        "y_start_hour": hours[target_start_idx],
                        "y_end_date": dates[target_end_idx - 1],
                        "y_end_hour": hours[target_end_idx - 1],
                    }
                )

    print(f"Found {total_segments} contiguous segments across all sites")
    print(f"Skipped {skipped_segments} segments (too short for window_size={window_size} + forecast_horizon={forecast_horizon})")

    # Convert to numpy arrays
    X = np.array(X)  # Shape: (n_samples, window_size, n_features)
    y = np.array(y)  # Shape: (n_samples, forecast_horizon, n_features)
    metadata_df = pd.DataFrame(metadata)

    return X, y, metadata_df