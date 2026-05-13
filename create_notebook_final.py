import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# ASHRAE - Great Energy Predictor III

This notebook implements the end-to-end pipeline for the ASHRAE Energy Predictor competition:
1. **Memory Optimization**
2. **Pre-processing:** Statistical pruning & Site 0 analysis
3. **Feature Engineering:** Time/Weather features, Lagged trajectories, Target encoding
4. **Modeling:** LightGBM & XGBoost ensemble with `log1p` transformation
"""

code_imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold
import gc
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
"""

text_mem = """\
## 1. Memory Optimization
"""

code_mem = """\
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
"""

text_load = "## 2. Data Loading & Joining"

code_load = """\
import os
data_dir = './data'

print("Loading data...")
train = pd.read_csv(f'{data_dir}/train.csv')
train = reduce_mem_usage(train)
building_meta = pd.read_csv(f'{data_dir}/building_metadata.csv')
building_meta = reduce_mem_usage(building_meta)
weather_train = pd.read_csv(f'{data_dir}/weather_train.csv')
weather_train = reduce_mem_usage(weather_train)

# Join building metadata
train = train.merge(building_meta, on='building_id', how='left')
"""

text_prep = "## 3. Pre-processing: Statistical Pruning"

code_prep = """\
def prune_outliers(df):
    print("Initial shape:", df.shape)
    
    # 1. Drop Site 0 electrical anomaly before May 20
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    idx_drop = df[(df['site_id'] == 0) & (df['meter'] == 0) & (df['timestamp'] < '2016-05-20')].index
    df.drop(idx_drop, inplace=True)
    print("Shape after Site 0 anomaly drop:", df.shape)
    
    # 2. Statistical Pruning (mean +/- 3*std) grouped by building_id and meter
    df['log_meter_reading'] = np.log1p(df['meter_reading'])
    
    group_stats = df.groupby(['building_id', 'meter'])['log_meter_reading'].agg(['mean', 'std']).reset_index()
    df = df.merge(group_stats, on=['building_id', 'meter'], how='left')
    
    # Filter out rows outside 3 std
    df['std'] = df['std'].fillna(0)
    
    lower_bound = df['mean'] - 3 * df['std']
    upper_bound = df['mean'] + 3 * df['std']
    
    idx_keep = (df['log_meter_reading'] >= lower_bound) & (df['log_meter_reading'] <= upper_bound)
    df = df[idx_keep].copy()
    print("Shape after statistical pruning:", df.shape)
    
    df.drop(['mean', 'std', 'log_meter_reading'], axis=1, inplace=True)
    return df

train = prune_outliers(train)
"""

text_feat = "## 4. Feature Engineering"

code_feat = """\
def add_time_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    return df

def add_weather_features(weather_df):
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    # Forward fill missing weather data per site
    weather_df = weather_df.groupby('site_id').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
    
    # Lagged features
    weather_df = weather_df.sort_values(by=['site_id', 'timestamp'])
    for lag in [3, 24]:
        weather_df[f'air_temp_lag_{lag}'] = weather_df.groupby('site_id')['air_temperature'].shift(lag)
        
    # Rolling features
    weather_df['air_temp_rolling_24'] = weather_df.groupby('site_id')['air_temperature'].transform(lambda x: x.rolling(24, min_periods=1).mean())
    return weather_df

def target_encode(train_df, columns, target='meter_reading'):
    train_df['target_log'] = np.log1p(train_df[target])
    for col in columns:
        train_df[f'{col}_target_enc'] = 0.0
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            mean_enc = X_tr.groupby(col)['target_log'].mean()
            train_df.loc[train_df.index[val_idx], f'{col}_target_enc'] = X_val[col].map(mean_enc)
    train_df.drop(['target_log'], axis=1, inplace=True)
    return train_df

print("Engineering features...")
weather_train = add_weather_features(weather_train)
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
train = add_time_features(train)
train = target_encode(train, columns=['primary_use', 'site_id'])

print("Final dataset shape:", train.shape)
"""

text_model = "## 5. Model Training: LightGBM & XGBoost Ensemble"

code_model = """\
# TARGET TRANSFORMATION
train['target'] = np.log1p(train['meter_reading'])

features = [c for c in train.columns if c not in ['timestamp', 'meter_reading', 'target', 'building_id', 'primary_use']]

models = []

for meter_type in range(4): # 0: electricity, 1: chilled water, 2: steam, 3: hot water
    print(f"\\n--- Training models for meter type {meter_type} ---")
    df_m = train[train['meter'] == meter_type].reset_index(drop=True)
    if df_m.empty: continue
        
    X = df_m[features]
    y = df_m['target']
    groups = df_m['month']
    
    gkf = GroupKFold(n_splits=3)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"Fold {fold}")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # LightGBM
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_valid = lgb.Dataset(X_va, y_va)
        params_lgb = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'gpu',
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'random_state': 42
        }
        model_lgb = lgb.train(params_lgb, lgb_train, valid_sets=[lgb_train, lgb_valid],
                              num_boost_round=1000, callbacks=[lgb.early_stopping(50)])
        
        # XGBoost
        xgb_train = xgb.DMatrix(X_tr, label=y_tr)
        xgb_valid = xgb.DMatrix(X_va, label=y_va)
        params_xgb = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist',
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        model_xgb = xgb.train(params_xgb, xgb_train, evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
                              num_boost_round=1000, early_stopping_rounds=50)
        
        models.append({'meter': meter_type, 'lgb': model_lgb, 'xgb': model_xgb})
        break # Train only 1 fold for demonstration. Remove 'break' for full training
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_mem),
    nbf.v4.new_code_cell(code_mem),
    nbf.v4.new_markdown_cell(text_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(text_prep),
    nbf.v4.new_code_cell(code_prep),
    nbf.v4.new_markdown_cell(text_feat),
    nbf.v4.new_code_cell(code_feat),
    nbf.v4.new_markdown_cell(text_model),
    nbf.v4.new_code_cell(code_model)
]

with open('ashrae_energy_predictor.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
