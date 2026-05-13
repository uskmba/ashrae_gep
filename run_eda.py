import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

print("Loading Data...")
train = pd.read_csv('data/train.csv')
train = reduce_mem_usage(train)
building_meta = pd.read_csv('data/building_metadata.csv')
building_meta = reduce_mem_usage(building_meta)

print("Merging...")
train_site = train.merge(building_meta[['building_id', 'site_id']], on='building_id', how='left')

print("Analyzing Site 0 Anomaly...")
# Filter to site 0, meter 0
site0_meter0 = train_site[(train_site['site_id'] == 0) & (train_site['meter'] == 0)].copy()
site0_meter0['timestamp'] = pd.to_datetime(site0_meter0['timestamp'])

# Calculate daily average meter reading for site 0 meter 0
daily_avg = site0_meter0.groupby(site0_meter0['timestamp'].dt.date)['meter_reading'].mean()

plt.figure(figsize=(15, 6))
daily_avg.plot()
plt.title('Site 0, Meter 0 (Electricity) Average Daily Reading in 2016')
plt.ylabel('Average Meter Reading')
plt.xlabel('Date')
plt.axvline(pd.to_datetime('2016-05-20'), color='red', linestyle='--', label='End of Zero Anomaly (~May 20)')
plt.legend()
plt.tight_layout()
plt.savefig('site0_anomaly.png')
print("Saved site0_anomaly.png")

# Checking proportion of exactly 0 readings before May 20
before_may_20 = site0_meter0[site0_meter0['timestamp'] < '2016-05-20']
zeros_proportion = (before_may_20['meter_reading'] == 0).mean() * 100
print(f"Percentage of exactly zero readings for Site 0, Meter 0 before May 20: {zeros_proportion:.2f}%")
