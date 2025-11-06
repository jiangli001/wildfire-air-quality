"""Combine multiple CSV files into a single dataframe with start and end dates."""

from pathlib import Path
from sys import argv
import pandas as pd

# Get all files in the folder
FILE_DIR = argv[1]
data_folder = Path(FILE_DIR)
files = list(data_folder.glob('*.csv'))

# List to store individual dataframes
dfs = []

for file in files:
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        print(f"Warning: {file} is empty and will be skipped.")
        continue

    # Extract start and end dates from filename
    filename = file.stem
    date_parts = filename.split('_')

    if len(date_parts) == 4:
        start_date_str = date_parts[1]
        end_date_str = date_parts[3]

        # Parse dates to datetime
        df['start_date'] = pd.to_datetime(start_date_str)
        df['end_date'] = pd.to_datetime(end_date_str)

    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

site_name_mapping = pd.read_csv('site_name_mapping.csv')
combined_df = combined_df.merge(site_name_mapping, on="site", how="left")


print(f"Combined {len(dfs)} files into a dataframe with {len(combined_df)} rows")
print(combined_df.head())
combined_df.to_csv(f'combined_{data_folder.name}.csv', index=False)
