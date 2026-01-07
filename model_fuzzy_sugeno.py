# FILE: model_fuzzy_sugeno.py
# Fuzzy Sugeno (conceptual neuro-fuzzy) - simplu si explicabil
# Alege automat 2 feature-uri si invata consequentii (ridge) cu membership fixe.

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def temporal_split(df_feat: pd.DataFrame, train_end=2018, val_end=2021):
    train = df_feat[df_feat["year"] <= train_end].copy()
    val = df_feat[(df_feat["year"] > train_end) & (df_feat["year"] <= val_end)].copy()
    test = df_feat[df_feat["year"] > val_end].copy()
    return train, val, test


def fix_inf_nan_inplace(dset: pd.DataFrame, skip_cols=("country", "year")):
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in dset.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(dset[col]):
            dset[col] = dset[col].fillna(dset[col].median())


def select_features_anti_leakage(df_feat: pd.DataFrame):
    drop_cols = ["country", "year", "total", "industry", "transport", "services", "households", "agriculture"]
    feature_cols = [c for c in df_feat.columns if (c not in drop_cols) and (not c.startswith("total_"))]
    return feature_cols


def gauss_mf(x, c, s):
    # Gaussian membership
    return np.exp(-0.5 * ((x - c) / (s + 1e-9)) ** 2)


def eval_metrics(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name} -> RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.3f}")
    return rmse, mae, r2


def main():
    t0 = time.time()
    print("=== MODEL: Fuzzy Sugeno (concept neuro-fuzzy) ===")

    df = pd.read_csv("dataset_features.csv")
    train, val, test = temporal_split(df)

    for d in (train, val, test):
        fix_inf_nan_inplace(d)

    features = select_features_anti_leakage(df)
    if len(features) < 2:
        raise ValueError("Nu am destule feature-uri pentru fuzzy (minim 2).")

    # luam 2 feature-uri pentru fuzzy
    f1, f2 = features[0], features[1]
    print("Fuzzy inputs:", f1, f2)

    X_train = train[[f1, f2]].to_numpy()
    y_train = train["total"].to_numpy()

    X_val = val[[f1, f2]].to_numpy()
    y_val = val["total"].to_numpy()

    X_test = test[[f1, f2]].to_numpy()
    y_test = test["total"].to_numpy()

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 3 membership per input: low, mid, high (centre fixe)
    centers = np.array([0.2, 0.5, 0.8])
    sigma = 0.18

    # reguli = produs cartezian (3x3 = 9 reguli)
    # fiecare regula are consequent: y = a0 + a1*x1 + a2*x2
    R = 9

    def compute_rule_weights(X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        mu1 = np.stack([gauss_mf(x1, c, sigma) for c in centers], axis=1)  # (N,3)
        mu2 = np.stack([gauss_mf(x2, c, sigma) for c in centers], axis=1)  # (N,3)

        w = []
        for i in range(3):
            for j in range(3):
                w.append(mu1[:, i] * mu2[:, j])
        W = np.stack(w, axis=1)  # (N,9)
        W = W / (W.sum(axis=1, keepdims=True) + 1e-9)  # normalize
        return W

    Wtr = compute_rule_weights(X_train_s)
    Wva = compute_rule_weights(X_val_s)
    Wte = compute_rule_weights(X_test_s)

    # construim design matrix pentru consequentii liniari pe fiecare regula
    # y_hat = sum_r w_r * (a0_r + a1_r*x1 + a2_r*x2)
    def build_design(X, W):
        N = X.shape[0]
        x1 = X[:, 0:1]
        x2 = X[:, 1:2]
        base = np.concatenate([np.ones((N, 1)), x1, x2], axis=1)  # (N,3)

        blocks = []
        for r in range(R):
            blocks.append(W[:, r:r+1] * base)  # (N,3)
        return np.concatenate(blocks, axis=1)  # (N, 27)

    Dtr = build_design(X_train_s, Wtr)
    Dva = build_design(X_val_s, Wva)
    Dte = build_design(X_test_s, Wte)

    # ridge ca "invatare" a consequentilor
    reg = Ridge(alpha=1.0)
    reg.fit(Dtr, y_train)

    pred_val = reg.predict(Dva)
    pred_test = reg.predict(Dte)

    print("\n=== EVALUARE ===")
    rmse_v, mae_v, r2_v = eval_metrics(y_val, pred_val, "VAL")
    rmse_t, mae_t, r2_t = eval_metrics(y_test, pred_test, "TEST")

    pd.DataFrame([{
        "model": "FuzzySugeno_concept",
        "inputs": f"{f1},{f2}",
        "rules": R,
        "rmse_val": rmse_v, "mae_val": mae_v, "r2_val": r2_v,
        "rmse_test": rmse_t, "mae_test": mae_t, "r2_test": r2_t,
        "runtime_sec": round(time.time() - t0, 3)
    }]).to_csv("metrics_fuzzy_sugeno.csv", index=False)

    plt.figure()
    plt.scatter(y_test, pred_test)
    plt.title("Fuzzy Sugeno: Actual vs Predicted (Test)")
    plt.xlabel("Actual total (TEST)")
    plt.ylabel("Predicted total (TEST)")
    plt.tight_layout()
    plt.savefig("fuzzy_actual_vs_pred_test.png", dpi=150)
    plt.show()

    residuals = y_test - pred_test
    plt.figure()
    plt.scatter(pred_test, residuals)
    plt.axhline(0)
    plt.title("Fuzzy Sugeno: Residuals vs Predicted (Test)")
    plt.xlabel("Predicted total (TEST)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig("fuzzy_residuals_test.png", dpi=150)
    plt.show()

    print("Saved: metrics_fuzzy_sugeno.csv, fuzzy_actual_vs_pred_test.png, fuzzy_residuals_test.png")


if __name__ == "__main__":
    main()
