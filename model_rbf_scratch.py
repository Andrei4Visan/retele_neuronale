# FILE: model_rbf_scratch.py
# Retea RBF implementata de la zero (centre + matrice Phi + solutie inchisa ridge)
# Folosim dataset_features.csv generat anterior.
#
# Ideea generala:
# Folosim o retea RBF pentru a aproxima relatia dintre variabilele explicative
# si consumul total de energie. Activarile RBF sunt functii Gaussiene centrate
# in anumite puncte din spatiul de date, iar ponderile de iesire sunt calculate
# analitic prin ridge regression.

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def temporal_split(df_feat: pd.DataFrame, train_end=2018, val_end=2021):
    """
    Realizam un split temporal al datelor:
    - train: ani <= train_end
    - validation: train_end < ani <= val_end
    - test: ani > val_end

    Procedam astfel pentru a evita data leakage.
    In problemele cu componenta temporala, nu avem voie
    sa folosim informatii din viitor pentru a invata trecutul.
    """
    train = df_feat[df_feat["year"] <= train_end].copy()
    val = df_feat[(df_feat["year"] > train_end) & (df_feat["year"] <= val_end)].copy()
    test = df_feat[df_feat["year"] > val_end].copy()
    return train, val, test


def fix_inf_nan_inplace(dset: pd.DataFrame, skip_cols=("country", "year")):
    """
    Curatam valorile numerice problematice din dataset.

    Pasii sunt:
    1) Inlocuim valorile +inf si -inf cu NaN.
    2) Completam valorile NaN pentru coloanele numerice
       folosind mediana coloanei.

    Alegem mediana deoarece este mai robusta la valori extreme
    si nu distorsioneaza distributia la fel de mult ca media.
    """
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)

    nan_before = int(dset.isna().sum().sum())

    for col in dset.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(dset[col]):
            med = dset[col].median()
            dset[col] = dset[col].fillna(med)

    nan_after = int(dset.isna().sum().sum())
    return nan_before, nan_after


def select_features_anti_leakage(df_feat: pd.DataFrame):
    """
    Selectam variabilele explicative, eliminand coloanele
    care pot introduce leakage sau predictii triviale.

    Eliminam:
    - identificatori (country, year)
    - variabila tinta (total)
    - sectoarele brute, deoarece totalul este foarte corelat
      cu acestea si modelul ar invata aproape direct suma lor.
    """
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

    feature_cols = [
        c for c in df_feat.columns
        if (c not in drop_cols) and (not c.startswith("total_"))
    ]

    return feature_cols


def rbf_phi(X: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    """
    Construim matricea Phi pentru reteaua RBF.

    Formula folosita este:
        Phi[i, j] = exp(-gamma * ||x_i - c_j||^2)

    Interpretare:
    - masuram distanta dintre fiecare observatie si fiecare centru
    - aplicam o functie Gaussiana care produce activari mari
      pentru puncte apropiate si mici pentru puncte indepartate

    Parametrul gamma controleaza latimea functiei:
    - gamma mare -> functii inguste, comportament foarte local
    - gamma mic  -> functii late, comportament mai neted
    """
    Phi = np.zeros((X.shape[0], centers.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(centers.shape[0]):
            diff = X[i] - centers[j]
            Phi[i, j] = np.exp(-gamma * float(np.dot(diff, diff)))
    return Phi


def eval_metrics(y_true, y_pred, name):
    """
    Calculam metricile standard de evaluare pentru regresie.

    - RMSE: penalizeaza mai mult erorile mari
    - MAE: masura mai robusta la outlieri
    - R2: cat din variatia lui y este explicata de model
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name} -> RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.3f}")
    return rmse, mae, r2


def main():
    """
    In aceasta functie implementam intregul pipeline:

    1) Citim datele
    2) Realizam split temporal
    3) Selectam feature-urile
    4) Curatam valorile NaN si Inf
    5) Scalare MinMax (fit doar pe train)
    6) Construim si antrenam reteaua RBF
    7) Evaluam rezultatele pe validation si test

    Observatie:
    In acest model nu folosim gradient descent si learning rate.
    Ponderile sunt calculate direct printr-o solutie analitica
    de tip ridge regression.
    """
    t0 = time.time()

    print("START model_rbf_scratch.py")
    print("=== MODEL: RBF FROM SCRATCH ===")

    # Incarcam dataset-ul de feature-uri
    path = "dataset_features.csv"
    df_feat = pd.read_csv(path)
    print("Citit:", path)
    print("Dimensiuni:", df_feat.shape)

    # Split temporal
    train, val, test = temporal_split(df_feat, train_end=2018, val_end=2021)

    # Selectie feature-uri
    feature_cols = select_features_anti_leakage(df_feat)

    # Curatare NaN si Inf separat pe fiecare split
    for dset in [train, val, test]:
        fix_inf_nan_inplace(dset, skip_cols=("country", "year"))

    # Construim X si y
    X_train = train[feature_cols].values
    y_train = train["total"].values

    X_val = val[feature_cols].values
    y_val = val["total"].values

    X_test = test[feature_cols].values
    y_test = test["total"].values

    # Scalare MinMax
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Hiperparametri RBF
    K = 25
    gamma = 10.0
    lam = 1e-3

    # Alegem centrele RBF aleator din setul de antrenare
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train_s), size=K, replace=False)
    centers = X_train_s[idx]

    # Construim matricea Phi
    Phi_train = rbf_phi(X_train_s, centers, gamma)
    Phi_val = rbf_phi(X_val_s, centers, gamma)
    Phi_test = rbf_phi(X_test_s, centers, gamma)

    # Adaugam termen de bias
    Phi_train = np.hstack([np.ones((Phi_train.shape[0], 1)), Phi_train])
    Phi_val = np.hstack([np.ones((Phi_val.shape[0], 1)), Phi_val])
    Phi_test = np.hstack([np.ones((Phi_test.shape[0], 1)), Phi_test])

    # Calculam ponderile folosind solutia inchisa ridge
    I = np.eye(Phi_train.shape[1])
    W = np.linalg.inv(Phi_train.T @ Phi_train + lam * I) @ (Phi_train.T @ y_train)

    # Predictii
    pred_val = Phi_val @ W
    pred_test = Phi_test @ W

    # Evaluare
    eval_metrics(y_val, pred_val, "VAL")
    eval_metrics(y_test, pred_test, "TEST")

    print("Runtime:", round(time.time() - t0, 3), "sec")


if __name__ == "__main__":
    main()
