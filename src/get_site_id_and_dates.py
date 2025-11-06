import pandas as pd

df = pd.read_csv("./data/pm25_data_20220101_to_20220114.csv")

# df['incident_dateonly_created'] = pd.to_datetime(df['incident_dateonly_created'], errors='raise')
# df['incident_dateonly_extinguished'] = pd.to_datetime(
#     df['incident_dateonly_extinguished'], errors='raise'
# )

# # Filter out rows where any of the three columns is empty
# df = df.dropna(subset=['SiteName', 'incident_dateonly_created', 'incident_dateonly_extinguished'])

# Convert rows to list of tuples (site_id, start_date, end_date)
# result = list(
#     df[[
#         'SiteName',
#         'incident_dateonly_created',
#         'incident_dateonly_extinguished'
#     ]].itertuples(index=False, name=None))
# print(result)

# print(site_name_mapping)

for col in ['site', 'name']:
    df[col] = df[col].astype('string').str.strip()
# convert to a dictionary mapping name to site
# site_name_mapping = dict(df[['site', 'name']].drop_duplicates().itertuples(index=False, name=None))
# site_name_mapping = {name: site for site, name in site_name_mapping.items()}
# print(site_name_mapping)

df[['site', 'name']].drop_duplicates().to_csv("./site_name_mapping.csv", index=False)
