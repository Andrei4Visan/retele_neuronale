# FILE: model_mlp.py
# Model MLP (retea neuronala feedforward) pentru predictia "total"
# folosind dataset_features.csv
#
# IMPORTANT:
# - Scrie metrics_mlp.csv in FORMAT STANDARD (wide) pentru compare_all_models.py
# - Nu mai scrie deloc formatul vechi cu split/RMSE/MAE/R2

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def temporal_split(df_feat: pd.DataFrame, train_end=2018, val_end=2021):
    train = df_feat[df_feat["year"] <= train_end].copy()
    val = df_feat[(df_feat["year"] > train_end) & (df_feat["year"] <= val_end)].copy()
    test = df_feat[df_feat["year"] > val_end].copy()
    return train, val, test


def clean_inf_nan_inplace(dset: pd.DataFrame, skip_cols=("country", "year")):
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
        "industry",
        "transport",
        "services",
        "households",
        "agriculture",
    ]
    feature_cols = [c for c in df_feat.columns if (c not in drop_cols) and (not c.startswith("total_"))]
    return feature_cols


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    t0 = time.time()
    print("=== MODEL: MLPRegressor ===")

    # =========================
    # PASUL 1: Citire date
    # =========================
    DATA_PATH = "dataset_features.csv"
    df_feat = pd.read_csv(DATA_PATH)

    print("Citit:", DATA_PATH)
    print("Dimensiuni:", df_feat.shape)
    print("Coloane:", len(df_feat.columns))

    # =========================
    # PASUL 2: Split temporal
    # =========================
    print("\n=== SPLIT TEMPORAL ===")
    train, val, test = temporal_split(df_feat, train_end=2018, val_end=2021)

    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    # =========================
    # PASUL 3: Feature selection (anti-leakage)
    # =========================
    print("\n=== FEATURE SELECTION (anti-leakage) ===")
    feature_cols = select_features_anti_leakage(df_feat)

    print("Nr features:", len(feature_cols))
    print("Features care incep cu total_ (trebuie 0):", sum(c.startswith("total_") for c in feature_cols))

    # =========================
    # PASUL 4: Tratare INF / NaN pe split-uri
    # =========================
    print("\n=== TRATARE INF / NaN (pe train/val/test) ===")
    nb, na = clean_inf_nan_inplace(train, skip_cols=("country", "year"))
    print(f"train: NaN inainte={nb}, dupa={na}")

    nb, na = clean_inf_nan_inplace(val, skip_cols=("country", "year"))
    print(f"val:   NaN inainte={nb}, dupa={na}")

    nb, na = clean_inf_nan_inplace(test, skip_cols=("country", "year"))
    print(f"test:  NaN inainte={nb}, dupa={na}")

    # =========================
    # PASUL 5: X/y + scalare (fit DOAR pe train)
    # =========================
    print("\n=== SCALARE (MinMax) ===")

    X_train = train[feature_cols].to_numpy()
    y_train = train["total"].to_numpy()

    X_val = val[feature_cols].to_numpy()
    y_val = val["total"].to_numpy()

    X_test = test[feature_cols].to_numpy()
    y_test = test["total"].to_numpy()

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("X_train_s:", X_train_s.shape, "y_train:", y_train.shape)

    # =========================
    # PASUL 6: Model MLP
    # =========================
    print("\n=== ANTRENARE MLP ===")

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
    )

    mlp.fit(X_train_s, y_train)

    print("Iteratii facute:", int(mlp.n_iter_))
    print("Loss final:", float(mlp.loss_))

    # =========================
    # PASUL 7: Evaluare (VAL + TEST)
    # =========================
    print("\n=== EVALUARE ===")

    pred_val = mlp.predict(X_val_s)
    pred_test = mlp.predict(X_test_s)

    rmse_val = rmse(y_val, pred_val)
    mae_val = float(mean_absolute_error(y_val, pred_val))
    r2_val = float(r2_score(y_val, pred_val))

    rmse_test = rmse(y_test, pred_test)
    mae_test = float(mean_absolute_error(y_test, pred_test))
    r2_test = float(r2_score(y_test, pred_test))

    print("VAL  -> RMSE:", rmse_val, "MAE:", mae_val, "R2:", r2_val)
    print("TEST -> RMSE:", rmse_test, "MAE:", mae_test, "R2:", r2_test)

    # =========================
    # GRAFICE: Loss + Validation + Actual vs Pred + Residuals
    # =========================

    # 1) Loss curve
    if hasattr(mlp, "loss_curve_") and mlp.loss_curve_ is not None:
        plt.figure()
        plt.plot(mlp.loss_curve_)
        plt.title("MLP - Training Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig("mlp_loss_curve.png", dpi=150)
        plt.close()

    # 2) Validation score curve
    if hasattr(mlp, "validation_scores_"):
        plt.figure()
        plt.plot(mlp.validation_scores_)
        plt.title("MLP - Validation Score Curve")
        plt.xlabel("Iteration")
        plt.ylabel("R2 (Validation)")
        plt.tight_layout()
        plt.savefig("mlp_validation_curve.png", dpi=150)
        plt.close()

    # 3) Actual vs Predicted (VAL)
    plt.figure()
    plt.scatter(y_val, pred_val)
    plt.xlabel("Actual total (VAL)")
    plt.ylabel("Predicted total (VAL)")
    plt.title("MLP: Actual vs Predicted (Validation)")
    plt.tight_layout()
    plt.savefig("plot_mlp_actual_vs_pred_val.png", dpi=200)
    plt.close()

    # 4) Actual vs Predicted (TEST)
    plt.figure()
    plt.scatter(y_test, pred_test)
    plt.xlabel("Actual total (TEST)")
    plt.ylabel("Predicted total (TEST)")
    plt.title("MLP: Actual vs Predicted (Test)")
    plt.tight_layout()
    plt.savefig("plot_mlp_actual_vs_pred_test.png", dpi=200)
    plt.close()

    # 5) Residuals (TEST)
    residuals_test = y_test - pred_test
    plt.figure()
    plt.scatter(pred_test, residuals_test)
    plt.axhline(0)
    plt.xlabel("Predicted total (TEST)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("MLP: Residuals vs Predicted (Test)")
    plt.tight_layout()
    plt.savefig("plot_mlp_residuals_test.png", dpi=200)
    plt.close()

    print("Salvat graficele MLP (png).")

    # =========================
    # PASUL 8: Salvare metrici (FORMAT STANDARD)
    # =========================
    runtime_sec = round(time.time() - t0, 3)

    metrics_std = {
        "model": "MLPRegressor",

        "rmse_train": np.nan,
        "mae_train": np.nan,
        "r2_train": np.nan,

        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "r2_val": r2_val,

        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test,

        "runtime_sec": runtime_sec,

        # metadate utile
        "hidden_layers": "64-32",
        "alpha": 1e-4,
        "lr": 1e-3,
    }

    pd.DataFrame([metrics_std]).to_csv("metrics_mlp.csv", index=False)
    print("\nMetrici salvate in metrics_mlp.csv (STANDARD)")
    print("Runtime sec:", runtime_sec)


if __name__ == "__main__":
    main()
