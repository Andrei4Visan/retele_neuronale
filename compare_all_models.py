# FILE: compare_all_models.py
# Unifica metricele din toate modelele si face comparatii (RMSE/MAE/R2 pe TEST).
# Output:
#  - metrics_all_models.csv
#  - plot_compare_rmse_test.png
#  - plot_compare_mae_test.png
#  - plot_compare_r2_test.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def load_metrics_file(path, kind):
    """
    kind:
      - "standard": fisiere de tip metrics_*.csv cu coloane rmse_val/rmse_test etc
      - "ga_best_mlp": fisier ga_best_mlp.csv cu best_rmse_val si test_rmse/test_mae/test_r2
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    row = df.iloc[0].to_dict()

    if kind == "standard":
        # incercam sa luam numele modelului
        model = row.get("model", os.path.splitext(os.path.basename(path))[0])

        out = {
            "model": str(model),

            "rmse_train": _safe_float(row.get("rmse_train", np.nan)),
            "mae_train": _safe_float(row.get("mae_train", np.nan)),
            "r2_train": _safe_float(row.get("r2_train", np.nan)),

            "rmse_val": _safe_float(row.get("rmse_val", np.nan)),
            "mae_val": _safe_float(row.get("mae_val", np.nan)),
            "r2_val": _safe_float(row.get("r2_val", np.nan)),

            "rmse_test": _safe_float(row.get("rmse_test", np.nan)),
            "mae_test": _safe_float(row.get("mae_test", np.nan)),
            "r2_test": _safe_float(row.get("r2_test", np.nan)),

            "runtime_sec": _safe_float(row.get("runtime_sec", np.nan)),
            "source_file": os.path.basename(path),
        }
        return out

    if kind == "ga_best_mlp":
        # ga_best_mlp.csv (din scriptul tau)
        model = row.get("model", "GA_opt_MLP")
        out = {
            "model": str(model),

            "rmse_train": np.nan,
            "mae_train": np.nan,
            "r2_train": np.nan,

            "rmse_val": _safe_float(row.get("best_rmse_val", np.nan)),
            "mae_val": np.nan,
            "r2_val": np.nan,

            "rmse_test": _safe_float(row.get("test_rmse", np.nan)),
            "mae_test": _safe_float(row.get("test_mae", np.nan)),
            "r2_test": _safe_float(row.get("test_r2", np.nan)),

            "runtime_sec": _safe_float(row.get("runtime_sec", np.nan)),
            "source_file": os.path.basename(path),
        }
        return out

    return None


def plot_bar(df, metric_col, title, out_png):
    dfp = df.dropna(subset=[metric_col]).copy()
    if dfp.empty:
        print(f"[WARN] Nu am valori pentru {metric_col}, sar peste plot.")
        return

    dfp = dfp.sort_values(metric_col, ascending=True)

    plt.figure(figsize=(10, 5))
    plt.bar(dfp["model"].astype(str), dfp[metric_col].astype(float))
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric_col)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    print("Saved:", out_png)


def main():
    # Lista de fisiere pe care le ai/trebuie sa le ai
    candidates = [
        ("metrics_mlp.csv", "standard"),
        ("metrics_rbf.csv", "standard"),
        ("metrics_rbf_scratch.csv", "standard"),
        ("metrics_bayesian_ridge.csv", "standard"),
        ("metrics_fuzzy_sugeno.csv", "standard"),
        ("metrics_rnn_elman_scratch.csv", "standard"),
        ("ga_best_mlp.csv", "ga_best_mlp"),
    ]

    rows = []
    for path, kind in candidates:
        r = load_metrics_file(path, kind)
        if r is None:
            print("[MISS]", path)
        else:
            print("[OK]  ", path, "->", r["model"])
            rows.append(r)

    if not rows:
        print("Nu am gasit niciun fisier de metrici. Verifica numele fisierelor CSV.")
        return

    allm = pd.DataFrame(rows)

    # Curatare: daca lipseste model, pune din source_file
    allm["model"] = allm["model"].fillna(allm["source_file"].astype(str))

    # Salveaza tabelul final
    allm.to_csv("metrics_all_models.csv", index=False)
    print("\nSaved: metrics_all_models.csv")

    # Afisare scurta in consola (top dupa RMSE_test)
    view = allm[["model", "rmse_val", "rmse_test", "mae_test", "r2_test", "runtime_sec", "source_file"]].copy()
    view = view.sort_values("rmse_test", ascending=True)
    print("\n=== RANK (best RMSE_test first) ===")
    print(view.to_string(index=False))

    # Ploturi
    plot_bar(allm, "rmse_test", "Comparatie RMSE (TEST) pe modele", "plot_compare_rmse_test.png")
    plot_bar(allm, "mae_test", "Comparatie MAE (TEST) pe modele", "plot_compare_mae_test.png")

    # La R2 mai mare e mai bine, dar plotul e ok oricum
    plot_bar(allm, "r2_test", "Comparatie R2 (TEST) pe modele", "plot_compare_r2_test.png")


if __name__ == "__main__":
    main()
