# FILE: model_som_scratch.py
# SOM (Self-Organizing Map) FROM SCRATCH + U-Matrix
# Dataset: dataset_features.csv (generat anterior)
# Output: som_umatrix.png, som_bmu_map.png, som_pca_clusters.png, metrics_som.csv

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# 1) Helpers: split + cleaning
# -----------------------------
def temporal_split(df_feat: pd.DataFrame, train_end=2018, val_end=2021):
    train = df_feat[df_feat["year"] <= train_end].copy()
    val = df_feat[(df_feat["year"] > train_end) & (df_feat["year"] <= val_end)].copy()
    test = df_feat[df_feat["year"] > val_end].copy()
    return train, val, test


def fix_inf_nan_inplace(dset: pd.DataFrame, skip_cols=("country", "year")):
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_before = int(dset.isna().sum().sum())

    for col in dset.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(dset[col]):
            dset[col] = dset[col].fillna(dset[col].median())

    nan_after = int(dset.isna().sum().sum())
    return nan_before, nan_after


def select_features_anti_leakage(df_feat: pd.DataFrame):
    drop_cols = [
        "country",
        "year",
        "total",
        # drop raw sectors to avoid trivial reconstruction
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


# -----------------------------
# 2) SOM from scratch
# -----------------------------
class SOM2D:
    def __init__(self, m, n, input_dim, sigma=2.0, lr=0.3, seed=42):
        self.m = m
        self.n = n
        self.input_dim = input_dim
        self.sigma0 = float(sigma)
        self.lr0 = float(lr)
        self.rng = np.random.default_rng(seed)

        # weights: (m, n, input_dim) in [0,1] because we use MinMax-scaled X
        self.W = self.rng.random((m, n, input_dim), dtype=float)

        # grid coordinates
        xs, ys = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
        self.grid = np.stack([xs, ys], axis=-1)  # (m, n, 2)

    def _bmu(self, x):
        # Euclidean distance to all neurons
        d = np.linalg.norm(self.W - x.reshape(1, 1, -1), axis=2)
        idx = np.unravel_index(np.argmin(d), d.shape)
        return idx  # (i, j)

    def fit(self, X, n_epochs=2000):
        X = np.asarray(X, dtype=float)
        T = int(n_epochs)

        for t in range(T):
            # exponential decay
            lr_t = self.lr0 * np.exp(-t / T)
            sigma_t = self.sigma0 * np.exp(-t / T)
            sigma_t = max(sigma_t, 1e-3)

            # pick random sample
            x = X[self.rng.integers(0, len(X))]

            bi, bj = self._bmu(x)
            bmu_xy = np.array([bi, bj], dtype=float)

            # neighborhood: gaussian over grid distance
            dist2 = np.sum((self.grid - bmu_xy) ** 2, axis=2)  # (m, n)
            h = np.exp(-dist2 / (2 * (sigma_t ** 2)))  # (m, n)

            # update weights
            # W += lr * h * (x - W)
            self.W += lr_t * h[..., None] * (x.reshape(1, 1, -1) - self.W)

    def transform_bmu_coords(self, X):
        coords = np.zeros((len(X), 2), dtype=int)
        for i, x in enumerate(X):
            bi, bj = self._bmu(x)
            coords[i] = [bi, bj]
        return coords

    def umatrix(self):
        # U-matrix: average distance to neighbors
        U = np.zeros((self.m, self.n), dtype=float)
        for i in range(self.m):
            for j in range(self.n):
                neighbors = []
                if i > 0:
                    neighbors.append(self.W[i - 1, j])
                if i < self.m - 1:
                    neighbors.append(self.W[i + 1, j])
                if j > 0:
                    neighbors.append(self.W[i, j - 1])
                if j < self.n - 1:
                    neighbors.append(self.W[i, j + 1])

                if neighbors:
                    neighbors = np.stack(neighbors, axis=0)
                    U[i, j] = np.mean(np.linalg.norm(neighbors - self.W[i, j], axis=1))
        return U


def eval_regression(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name} -> RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.3f}")
    return rmse, mae, r2


def main():
    t0 = time.time()

    print("START model_som_scratch.py")
    print("=== MODEL: SOM (SCRATCH) + U-MATRIX ===")

    # 1) Load data
    path = "dataset_features.csv"
    df_feat = pd.read_csv(path)
    print("Citit:", path, "Shape:", df_feat.shape)

    # 2) Split
    train, val, test = temporal_split(df_feat, train_end=2018, val_end=2021)
    print("Train:", train.shape, "Val:", val.shape, "Test:", test.shape)

    # 3) Feature selection
    feature_cols = select_features_anti_leakage(df_feat)
    print("Nr features:", len(feature_cols))
    print("Features care incep cu total_ (trebuie 0):", sum(c.startswith("total_") for c in feature_cols))

    # 4) Fix inf/nan per split (after split)
    print("\n=== TRATARE INF / NaN (pe train/val/test) ===")
    for name, dset in [("train", train), ("val", val), ("test", test)]:
        nb, na = fix_inf_nan_inplace(dset)
        print(f"{name}: NaN inainte={nb}, dupa={na}")

    # 5) Build arrays
    X_train = train[feature_cols].values
    y_train = train["total"].values

    X_val = val[feature_cols].values
    y_val = val["total"].values

    X_test = test[feature_cols].values
    y_test = test["total"].values

    # 6) Scale (fit on train only)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\n=== SOM TRAINING ===")
    # SOM hyperparams (simple)
    m, n = 10, 10
    som = SOM2D(m=m, n=n, input_dim=X_train_s.shape[1], sigma=3.0, lr=0.35, seed=42)
    som.fit(X_train_s, n_epochs=3000)

    # 7) U-matrix plot (mandatory style viz for SOM)
    U = som.umatrix()
    plt.figure()
    plt.imshow(U, origin="lower", aspect="auto")
    plt.title("SOM U-Matrix (train)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("som_umatrix.png", dpi=150)
    plt.show()

    # 8) BMU map for train points (just to see distribution)
    bmu_train = som.transform_bmu_coords(X_train_s)
    plt.figure()
    plt.scatter(bmu_train[:, 0] + 0.1, bmu_train[:, 1] + 0.1)
    plt.title("SOM BMU map (train points)")
    plt.xlabel("BMU i")
    plt.ylabel("BMU j")
    plt.tight_layout()
    plt.savefig("som_bmu_map.png", dpi=150)
    plt.show()

    # 9) Simple supervised mapping:
    #    Use BMU coords as features -> Ridge regression to predict total
    #    (keeps the project simple and explainable)
    print("\n=== SUPERVISED STEP: Ridge on BMU coords ===")
    bmu_val = som.transform_bmu_coords(X_val_s)
    bmu_test = som.transform_bmu_coords(X_test_s)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(bmu_train, y_train)

    pred_val = model.predict(bmu_val)
    pred_test = model.predict(bmu_test)

    print("\n=== EVALUARE (SOM + Ridge) ===")
    rmse_v, mae_v, r2_v = eval_regression(y_val, pred_val, "VAL")
    rmse_t, mae_t, r2_t = eval_regression(y_test, pred_test, "TEST")

    metrics = pd.DataFrame([
        {"model": "SOM_scratch_plus_Ridge", "split": "val", "RMSE": rmse_v, "MAE": mae_v, "R2": r2_v},
        {"model": "SOM_scratch_plus_Ridge", "split": "test", "RMSE": rmse_t, "MAE": mae_t, "R2": r2_t},
    ])
    metrics.to_csv("metrics_som.csv", index=False)
    print("Metrici salvate in metrics_som.csv")

    # 10) Optional but nice: PCA scatter colored by SOM "cluster id" (neuron index)
    # cluster_id = i*n + j
    cluster_train = bmu_train[:, 0] * n + bmu_train[:, 1]
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_train_s)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=cluster_train)
    plt.title("PCA (train) colored by SOM BMU cluster id")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("som_pca_clusters.png", dpi=150)
    plt.show()

    print("Runtime sec:", round(time.time() - t0, 3))


if __name__ == "__main__":
    main()
