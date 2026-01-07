import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SECTORS = ["industry", "transport", "services", "households", "agriculture"]
TARGET = "total"

def load_features_csv(path="dataset_features.csv"):
    df = pd.read_csv(path)
    return df

def temporal_split(df, train_end=2018, val_end=2021):
    train = df[df["year"] <= train_end].copy()
    val = df[(df["year"] > train_end) & (df["year"] <= val_end)].copy()
    test = df[df["year"] > val_end].copy()
    return train, val, test

def feature_cols_anti_leakage(df):
    # Scoatem categorice + target + valorile curente brute (sector) ca sa fie mai “corect” pe forecast
    drop_cols = ["country", "year", TARGET] + SECTORS

    # In plus, scoatem orice feature care incepe cu total_ (anti-leakage)
    cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if c.startswith("total_"):
            continue
        cols.append(c)
    return cols

def replace_inf_and_impute_median(dset, skip_cols=("country", "year")):
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in dset.columns:
        if col in skip_cols:
            continue
        if dset[col].isna().any():
            dset[col] = dset[col].fillna(dset[col].median())
    return dset

def build_scaled_matrices(train, val, test, feature_cols):
    X_train = train[feature_cols].values
    y_train = train[TARGET].values

    X_val = val[feature_cols].values
    y_val = val[TARGET].values

    X_test = test[feature_cols].values
    y_test = test[TARGET].values

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test), scaler
