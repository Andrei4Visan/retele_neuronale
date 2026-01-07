# FILE: model_rbf_scratch.py
# RBF Network FROM SCRATCH (centers + Phi matrix + closed-form ridge)
# Uses dataset_features.csv generated earlier.

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def temporal_split(df_feat: pd.DataFrame, train_end=2018, val_end=2021):
    train = df_feat[df_feat["year"] <= train_end].copy()
    val = df_feat[(df_feat["year"] > train_end) & (df_feat["year"] <= val_end)].copy()
    test = df_feat[df_feat["year"] > val_end].copy()
    return train, val, test


def fix_inf_nan_inplace(dset: pd.DataFrame, skip_cols=("country", "year")):
    # Replace inf with NaN
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)

    nan_before = int(dset.isna().sum().sum())

    # Fill NaN with median for numeric columns
    for col in dset.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(dset[col]):
            med = dset[col].median()
            dset[col] = dset[col].fillna(med)

    nan_after = int(dset.isna().sum().sum())
    return nan_before, nan_after


def select_features_anti_leakage(df_feat: pd.DataFrame):
    # Target + identifiers
    drop_cols = [
        "country",
        "year",
        "total",
        # Raw sectors (optional drop to avoid trivial reconstruction)
        "industry",
        "transport",
        "services",
        "households",
        "agriculture",
    ]

    feature_cols = [
        c for c in df_feat.columns
        if (c not in drop_cols) and (not c.startswith("total_"))
    ]

    return feature_cols


def rbf_phi(X: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    # Phi[i, j] = exp(-gamma * ||x_i - c_j||^2)
    Phi = np.zeros((X.shape[0], centers.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(centers.shape[0]):
            diff = X[i] - centers[j]
            Phi[i, j] = np.exp(-gamma * float(np.dot(diff, diff)))
    return Phi


def eval_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)  # fara squared=
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name} -> RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.3f}")
    return rmse, mae, r2



def main():
    t0 = time.time()

    print("START model_rbf_scratch.py")
    print("=== MODEL: RBF FROM SCRATCH ===")

    # 1) Load features dataset
    path = "dataset_features.csv"
    df_feat = pd.read_csv(path)
    print("Citit:", path)
    print("Dimensiuni:", df_feat.shape)

    # 2) Split temporal
    print("\n=== SPLIT TEMPORAL ===")
    train, val, test = temporal_split(df_feat, train_end=2018, val_end=2021)
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    # 3) Feature selection (anti-leakage)
    print("\n=== FEATURE SELECTION (anti-leakage) ===")
    feature_cols = select_features_anti_leakage(df_feat)
    print("Nr features:", len(feature_cols))
    print("Features care incep cu total_ (trebuie 0):", sum(c.startswith("total_") for c in feature_cols))

    # 4) Fix inf/nan on each split (IMPORTANT: after split)
    print("\n=== TRATARE INF / NaN (pe train/val/test) ===")
    for name, dset in [("train", train), ("val", val), ("test", test)]:
        nb, na = fix_inf_nan_inplace(dset, skip_cols=("country", "year"))
        print(f"{name}: NaN inainte={nb}, dupa={na}")

    # 5) Build X/y
    X_train = train[feature_cols].values
    y_train = train["total"].values

    X_val = val[feature_cols].values
    y_val = val["total"].values

    X_test = test[feature_cols].values
    y_test = test["total"].values

    # 6) Scale features (fit only on train)
    print("\n=== SCALARE (MinMax) ===")
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("X_train_s:", X_train_s.shape, "y_train:", y_train.shape)

    # ============================
    # RBF FROM SCRATCH
    # ============================
    print("\n=== RBF FROM SCRATCH ===")

    # Hyperparametri simpli
    K = 25          # numar centre RBF
    gamma = 10.0    # latimea functiei RBF
    lam = 1e-3      # regularizare ridge

    # 1) Selectie centre (random din train)
    rng = np.random.default_rng(42)
    if K > len(X_train_s):
        K = len(X_train_s)

    idx = rng.choice(len(X_train_s), size=K, replace=False)
    centers = X_train_s[idx]
    print("Nr centre RBF:", centers.shape)

    # 2) Matrice Phi
    Phi_train = rbf_phi(X_train_s, centers, gamma)
    Phi_val = rbf_phi(X_val_s, centers, gamma)
    Phi_test = rbf_phi(X_test_s, centers, gamma)

    # Bias term
    Phi_train = np.hstack([np.ones((Phi_train.shape[0], 1)), Phi_train])
    Phi_val = np.hstack([np.ones((Phi_val.shape[0], 1)), Phi_val])
    Phi_test = np.hstack([np.ones((Phi_test.shape[0], 1)), Phi_test])

    # 3) Closed-form ridge regression: W = (Phi^T Phi + lam I)^(-1) Phi^T y
    I = np.eye(Phi_train.shape[1])
    W = np.linalg.inv(Phi_train.T @ Phi_train + lam * I) @ (Phi_train.T @ y_train)

    # Predictii
    pred_val = Phi_val @ W
    pred_test = Phi_test @ W

    # Evaluare
    print("\n=== EVALUARE RBF SCRATCH ===")
    rmse_v, mae_v, r2_v = eval_metrics(y_val, pred_val, "VAL")
    rmse_t, mae_t, r2_t = eval_metrics(y_test, pred_test, "TEST")

    # Salvare metrici
    metrics_rbf = pd.DataFrame([{
        "model": "RBF_scratch",
        "K": K,
        "gamma": gamma,
        "lambda": lam,
        "rmse_val": rmse_v,
        "mae_val": mae_v,
        "r2_val": r2_v,
        "rmse_test": rmse_t,
        "mae_test": mae_t,
        "r2_test": r2_t
    }])

    metrics_rbf.to_csv("metrics_rbf_scratch.csv", index=False)
    print("Metrici salvate in metrics_rbf_scratch.csv")

    # 4) Grafice simple (optional, dar utile)
    plt.figure()
    plt.scatter(y_test, pred_test)
    plt.title("RBF Scratch: Actual vs Predicted (Test)")
    plt.xlabel("Actual total (TEST)")
    plt.ylabel("Predicted total (TEST)")
    plt.tight_layout()
    plt.savefig("rbf_scratch_actual_vs_pred_test.png", dpi=150)
    plt.show()

    residuals = y_test - pred_test
    plt.figure()
    plt.scatter(pred_test, residuals)
    plt.axhline(0)
    plt.title("RBF Scratch: Residuals vs Predicted (Test)")
    plt.xlabel("Predicted total (TEST)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig("rbf_scratch_residuals_vs_pred_test.png", dpi=150)
    plt.show()

    print("\nRuntime sec:", round(time.time() - t0, 3))


if __name__ == "__main__":
    main()
