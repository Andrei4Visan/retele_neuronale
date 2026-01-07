import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans

from utils_data import (
    load_features_csv,
    temporal_split,
    feature_cols_anti_leakage,
    replace_inf_and_impute_median,
    build_scaled_matrices
)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_block(name, split, y_true, y_pred):
    return {
        "model": name,
        "split": split,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def rbf_design_matrix(X, centers, gamma):
    # Phi[i, j] = exp(-gamma * ||x_i - c_j||^2)
    X2 = np.sum(X**2, axis=1, keepdims=True)           # (n,1)
    C2 = np.sum(centers**2, axis=1, keepdims=True).T   # (1,k)
    XC = X @ centers.T                                  # (n,k)
    d2 = X2 - 2 * XC + C2                               # (n,k)
    Phi = np.exp(-gamma * d2)
    return Phi

def ridge_closed_form(Phi, y, lam):
    # w = (Phi^T Phi + lam I)^-1 Phi^T y
    k = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(k)
    b = Phi.T @ y
    w = np.linalg.solve(A, b)
    return w

def main():
    print("=== MODEL: RBF Network (from scratch) ===")
    t0 = time.time()

    df = load_features_csv("dataset_features.csv")
    train, val, test = temporal_split(df, train_end=2018, val_end=2021)

    # tratare inf/nan pe split-uri
    replace_inf_and_impute_median(train)
    replace_inf_and_impute_median(val)
    replace_inf_and_impute_median(test)

    feat_cols = feature_cols_anti_leakage(df)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = build_scaled_matrices(
        train, val, test, feat_cols
    )

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("Nr features:", len(feat_cols))

    # 1) Alegem centrele cu KMeans (simplu si standard)
    k = 20
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train)
    centers = km.cluster_centers_

    # 2) Setam gamma pe baza unei reguli simple (mediana distantelor intre centre)
    # gamma = 1 / (2 * sigma^2)
    # sigma estimat din distantele dintre centre
    from itertools import combinations
    dists = []
    for i, j in combinations(range(k), 2):
        dists.append(np.sum((centers[i] - centers[j]) ** 2))
    med = np.median(dists) if len(dists) else 1.0
    sigma2 = med if med > 1e-12 else 1.0
    gamma = 1.0 / (2.0 * sigma2)

    lam = 1e-2  # ridge regularization

    # 3) Construim Phi si rezolvam closed-form
    Phi_train = rbf_design_matrix(X_train, centers, gamma)
    Phi_val = rbf_design_matrix(X_val, centers, gamma)
    Phi_test = rbf_design_matrix(X_test, centers, gamma)

    w = ridge_closed_form(Phi_train, y_train, lam)

    pred_val = Phi_val @ w
    pred_test = Phi_test @ w

    # 4) Metrici
    m_val = metrics_block("RBF_from_scratch", "val", y_val, pred_val)
    m_test = metrics_block("RBF_from_scratch", "test", y_test, pred_test)

    print("VAL  ->", m_val)
    print("TEST ->", m_test)

    # Salvam metrici
    out = pd.DataFrame([m_val, m_test])
    out.to_csv("metrics_rbf.csv", index=False)
    print("Metrici salvate in metrics_rbf.csv")

    # 5) Grafice (2 simple, obligatorii)
    plt.figure()
    plt.scatter(y_test, pred_test)
    plt.title("RBF: Actual vs Predicted (Test)")
    plt.xlabel("Actual total (TEST)")
    plt.ylabel("Predicted total (TEST)")
    plt.tight_layout()
    plt.savefig("rbf_actual_vs_pred_test.png", dpi=150)
    plt.show()

    residuals = y_test - pred_test
    plt.figure()
    plt.scatter(pred_test, residuals)
    plt.axhline(0)
    plt.title("RBF: Residuals vs Predicted (Test)")
    plt.xlabel("Predicted total (TEST)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig("rbf_residuals_vs_pred_test.png", dpi=150)
    plt.show()

    print("Runtime sec:", round(time.time() - t0, 3))

if __name__ == "__main__":
    main()
