# FILE: model_probabilistic_bayesian_ridge.py
# Probabilistic model (Bayesian Ridge) for regression + uncertainty
# Uses dataset_features.csv, temporal split, strict anti-leakage preprocessing.

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def temporal_split(df_feat: pd.DataFrame, train_end=2018, val_end=2021):
    train = df_feat[df_feat["year"] <= train_end].copy()
    val = df_feat[(df_feat["year"] > train_end) & (df_feat["year"] <= val_end)].copy()
    test = df_feat[df_feat["year"] > val_end].copy()
    return train, val, test


def replace_inf_with_nan_inplace(dset: pd.DataFrame):
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)


def compute_train_medians(train: pd.DataFrame, skip_cols=("country", "year")) -> dict:
    med = {}
    for col in train.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            med[col] = float(train[col].median())
    return med


def apply_train_medians_inplace(dset: pd.DataFrame, medians: dict, skip_cols=("country", "year")):
    for col, m in medians.items():
        if col in skip_cols:
            continue
        if col in dset.columns:
            dset[col] = dset[col].fillna(m)


def select_features_anti_leakage(df_feat: pd.DataFrame):
    # Drop identifiers + targets and direct components of total (to avoid leakage)
    drop_cols = [
        "country", "year",
        "total", "industry", "transport", "services", "households", "agriculture"
    ]
    # Also drop any engineered "total_*" columns (lag/rolling of target), if they exist
    feature_cols = [
        c for c in df_feat.columns
        if (c not in drop_cols) and (not c.startswith("total_"))
    ]
    return feature_cols


def eval_metrics(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name} -> RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.3f}")
    return rmse, mae, r2


def main():
    t0 = time.time()
    print("=== MODEL: BayesianRidge (probabilistic) ===")

    df = pd.read_csv("dataset_features.csv")

    # Split temporal
    train, val, test = temporal_split(df, train_end=2018, val_end=2021)

    # Replace inf with nan in all splits
    for d in (train, val, test):
        replace_inf_with_nan_inplace(d)

    # Strict imputation: compute medians on TRAIN only, apply to VAL and TEST
    train_medians = compute_train_medians(train, skip_cols=("country", "year"))
    apply_train_medians_inplace(train, train_medians, skip_cols=("country", "year"))
    apply_train_medians_inplace(val, train_medians, skip_cols=("country", "year"))
    apply_train_medians_inplace(test, train_medians, skip_cols=("country", "year"))

    # Select features (anti-leakage)
    feature_cols = select_features_anti_leakage(df)

    # Build matrices
    X_train = train[feature_cols].to_numpy()
    y_train = train["total"].to_numpy()

    X_val = val[feature_cols].to_numpy()
    y_val = val["total"].to_numpy()

    X_test = test[feature_cols].to_numpy()
    y_test = test["total"].to_numpy()

    # Scale features using TRAIN only
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train model
    model = BayesianRidge()
    model.fit(X_train_s, y_train)

    # Predict with uncertainty
    pred_val, std_val = model.predict(X_val_s, return_std=True)
    pred_test, std_test = model.predict(X_test_s, return_std=True)

    print("\n=== EVALUARE ===")
    rmse_v, mae_v, r2_v = eval_metrics(y_val, pred_val, "VAL")
    rmse_t, mae_t, r2_t = eval_metrics(y_test, pred_test, "TEST")

    # Save metrics
    pd.DataFrame([{
        "model": "BayesianRidge_prob",
        "rmse_val": rmse_v, "mae_val": mae_v, "r2_val": r2_v,
        "rmse_test": rmse_t, "mae_test": mae_t, "r2_test": r2_t,
        "runtime_sec": round(time.time() - t0, 3),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_cols))
    }]).to_csv("metrics_bayesian_ridge.csv", index=False)

    # Plot 1: actual vs predicted (test) + y=x reference
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_test)

    lo = float(min(y_test.min(), pred_test.min()))
    hi = float(max(y_test.max(), pred_test.max()))
    plt.plot([lo, hi], [lo, hi])

    plt.title("BayesianRidge: Actual vs Predicted (Test)")
    plt.xlabel("Actual total (TEST)")
    plt.ylabel("Predicted total (TEST)")
    plt.tight_layout()
    plt.savefig("bayes_actual_vs_pred_test.png", dpi=150)
    plt.show()

    # Plot 2: uncertainty (error bars) on test, sorted by actual
    order = np.argsort(y_test)
    plt.figure(figsize=(9, 4))
    plt.errorbar(
        np.arange(len(y_test)),
        pred_test[order],
        yerr=2.0 * std_test[order],
        fmt="o",
        capsize=2
    )
    plt.title("BayesianRidge: Prediction uncertainty (Test, +-2*std)")
    plt.xlabel("Test sample (sorted by actual)")
    plt.ylabel("Predicted total")
    plt.tight_layout()
    plt.savefig("bayes_uncertainty_test.png", dpi=150)
    plt.show()

    print("Saved: metrics_bayesian_ridge.csv, bayes_actual_vs_pred_test.png, bayes_uncertainty_test.png")


if __name__ == "__main__":
    main()
