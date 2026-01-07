# FILE: ga_optimize_mlp.py
# Genetic Algorithm (simplu) pentru optimizarea hiperparametrilor MLPRegressor
# Stabilizat:
# - StandardScaler
# - max_iter mai mare + early_stopping
# - selectie best corecta (re-evaluare pe populatia finala)
# - evaluare finala pe TEST
# - scrie CSV compatibil cu compare_all_models.py (si cu varianta veche)

import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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
    # target + coloane derivate care ar induce leakage
    drop_cols = [
        "country", "year", "total",
        "industry", "transport", "services", "households", "agriculture",
    ]
    feature_cols = [c for c in df_feat.columns if (c not in drop_cols) and (not c.startswith("total_"))]
    return feature_cols


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def sample_individual(rng: np.random.Generator):
    # hidden units in [8..96], alpha in [1e-7..1e-2], lr in [1e-4..5e-2]
    h = int(rng.integers(8, 97))
    alpha = 10 ** rng.uniform(-7, -2)
    lr = 10 ** rng.uniform(-4, -1.3)
    return {"h": h, "alpha": float(alpha), "lr": float(lr)}


def mutate(ind, rng: np.random.Generator, p=0.35):
    out = dict(ind)
    if rng.random() < p:
        out["h"] = int(np.clip(out["h"] + rng.integers(-12, 13), 8, 96))
    if rng.random() < p:
        out["alpha"] = float(np.clip(out["alpha"] * (10 ** rng.uniform(-0.6, 0.6)), 1e-7, 1e-2))
    if rng.random() < p:
        out["lr"] = float(np.clip(out["lr"] * (10 ** rng.uniform(-0.6, 0.6)), 1e-4, 5e-2))
    return out


def crossover(a, b, rng: np.random.Generator):
    child = {}
    for k in a.keys():
        child[k] = a[k] if rng.random() < 0.5 else b[k]
    return child


def make_model(ind, seed=42):
    return MLPRegressor(
        hidden_layer_sizes=(ind["h"],),
        alpha=ind["alpha"],
        learning_rate_init=ind["lr"],
        solver="adam",
        learning_rate="adaptive",
        max_iter=2500,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.15,
    )


def fitness(ind, Xtr, ytr, Xva, yva, seed=42):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Stochastic Optimizer*")

        model = make_model(ind, seed=seed)
        model.fit(Xtr, ytr)

    pred = model.predict(Xva)
    return rmse(yva, pred)


def main():
    t0 = time.time()
    print("=== GA OPTIMIZE: MLPRegressor (stabilizat) ===")

    # IMPORTANT: foloseste fisierul tau corect
    df = pd.read_csv("dataset_features.csv")
    train, val, test = temporal_split(df)

    for d in (train, val, test):
        fix_inf_nan_inplace(d)

    feat = select_features_anti_leakage(df)

    Xtr = train[feat].to_numpy()
    ytr = train["total"].to_numpy()

    Xva = val[feat].to_numpy()
    yva = val["total"].to_numpy()

    Xte = test[feat].to_numpy()
    yte = test["total"].to_numpy()

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    rng = np.random.default_rng(42)

    pop_size = 12
    generations = 8
    elite_k = 4

    pop = [sample_individual(rng) for _ in range(pop_size)]

    # GA loop
    for g in range(1, generations + 1):
        scores = [fitness(ind, Xtr_s, ytr, Xva_s, yva, seed=42) for ind in pop]
        order = np.argsort(scores)
        pop = [pop[i] for i in order]
        scores = [scores[i] for i in order]

        best = pop[0]
        print(
            f"Gen {g}: best_RMSE_val={scores[0]:.2f}  "
            f"h={best['h']} alpha={best['alpha']:.2e} lr={best['lr']:.2e}"
        )

        # elitism + recombinare
        new_pop = pop[:elite_k]
        while len(new_pop) < pop_size:
            p1 = pop[int(rng.integers(0, pop_size // 2))]
            p2 = pop[int(rng.integers(0, pop_size // 2))]
            child = crossover(p1, p2, rng)
            child = mutate(child, rng, p=0.4)
            new_pop.append(child)
        pop = new_pop

    # Re-evaluare corecta pe populatia finala
    final_scores = [fitness(ind, Xtr_s, ytr, Xva_s, yva, seed=42) for ind in pop]
    order = np.argsort(final_scores)
    pop = [pop[i] for i in order]
    final_scores = [final_scores[i] for i in order]

    best = pop[0]
    best_rmse_val = float(final_scores[0])

    print("\nBEST after GA (sorted):")
    print(f"RMSE_val={best_rmse_val:.2f}  h={best['h']} alpha={best['alpha']:.2e} lr={best['lr']:.2e}")

    # Final train on train+val, then test
    Xtrva_s = np.vstack([Xtr_s, Xva_s])
    ytrva = np.concatenate([ytr, yva])

    final_model = make_model(best, seed=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Stochastic Optimizer*")
        final_model.fit(Xtrva_s, ytrva)

    pred_test = final_model.predict(Xte_s)
    test_m = eval_metrics(yte, pred_test)

    # Save artifacts
    joblib.dump(
        {"scaler": scaler, "features": feat, "model": final_model},
        "ga_best_mlp_model.joblib",
    )

    runtime_sec = round(time.time() - t0, 3)

    # CSV output (standard + compatibil vechi)
    out = {
        "model": "GA_opt_MLP",
        "h": int(best["h"]),
        "alpha": float(best["alpha"]),
        "lr": float(best["lr"]),
        "generations": int(generations),
        "pop_size": int(pop_size),
        "runtime_sec": float(runtime_sec),

        # standard (compare_all_models standard)
        "rmse_val": float(best_rmse_val),
        "mae_val": np.nan,
        "r2_val": np.nan,

        "rmse_test": float(test_m["rmse"]),
        "mae_test": float(test_m["mae"]),
        "r2_test": float(test_m["r2"]),

        # compatibil vechi (compare_all_models ga_best_mlp vechi)
        "best_rmse_val": float(best_rmse_val),
        "test_rmse": float(test_m["rmse"]),
        "test_mae": float(test_m["mae"]),
        "test_r2": float(test_m["r2"]),
    }

    pd.DataFrame([out]).to_csv("ga_best_mlp.csv", index=False)

    print("\nSaved: ga_best_mlp.csv")
    print("Saved: ga_best_mlp_model.joblib")
    print("TEST metrics:", out["rmse_test"], out["mae_test"], out["r2_test"])


if __name__ == "__main__":
    main()
