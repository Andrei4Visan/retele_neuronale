# FILE: model_rnn_jordan.py
# RNN Jordan FROM SCRATCH (numpy) for regression (predict total)
# Feedback: y_{t-1} -> hidden
#
# Fix-uri:
# 1) Curata df INAINTE sa construiesti secventele (altfel Inf/NaN ajung in Xtr si crapa scaler.fit)
# 2) Protectie extra cu np.nan_to_num pe secvente
# 3) Wyh chiar este antrenat (aproximatie simpla: gradient local pe fiecare pas, fara lant complet prin y_prev)

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
    """
    Inlocuieste Inf/-Inf cu NaN, apoi imputa NaN cu mediana pe coloane numerice.
    """
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in dset.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(dset[col]):
            med = dset[col].median()
            if pd.isna(med):
                med = 0.0
            dset[col] = dset[col].fillna(med)


def select_features_anti_leakage(df_feat: pd.DataFrame):
    """
    Eliminam coloanele care pot introduce leakage:
    - target-ul total
    - componentele sectoriale (industry, transport, etc.)
    - agregari de tip total_*
    - identificatori (country, year)
    """
    drop_cols = [
        "country", "year", "total",
        "industry", "transport", "services", "households", "agriculture",
    ]
    return [c for c in df_feat.columns if (c not in drop_cols) and (not c.startswith("total_"))]


def make_sequences_global(df, feature_cols, target_col="total", seq_len=3, train_end=2018, val_end=2021):
    """
    Construieste secvente pe fiecare tara, apoi imparte in train/val/test dupa anul ultimului element din secventa.
    Returneaza:
      Xtr (Ntr, seq_len, D), ytr (Ntr,)
      Xva (Nva, seq_len, D), yva (Nva,)
      Xte (Nte, seq_len, D), yte (Nte,)
    """
    X_tr, y_tr, X_va, y_va, X_te, y_te = [], [], [], [], [], []

    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    for _, g in df.groupby("country"):
        g = g.sort_values("year").reset_index(drop=True)

        Xg = g[feature_cols].to_numpy(dtype=float)
        yg = g[target_col].to_numpy(dtype=float)
        years = g["year"].to_numpy()

        if len(g) < seq_len:
            continue

        for t in range(seq_len - 1, len(g)):
            x_seq = Xg[t - (seq_len - 1): t + 1]  # (seq_len, D)
            y_t = yg[t]
            end_year = years[t]

            if end_year <= train_end:
                X_tr.append(x_seq)
                y_tr.append(y_t)
            elif end_year <= val_end:
                X_va.append(x_seq)
                y_va.append(y_t)
            else:
                X_te.append(x_seq)
                y_te.append(y_t)

    return (
        np.array(X_tr, dtype=float), np.array(y_tr, dtype=float),
        np.array(X_va, dtype=float), np.array(y_va, dtype=float),
        np.array(X_te, dtype=float), np.array(y_te, dtype=float),
    )


class JordanRNNRegressor:
    """
    Jordan RNN:
      h_t = tanh(x_t Wxh + h_{t-1} Whh + y_{t-1} Wyh + b_h)
      y_t = h_t Why + b_y

    Pentru o secventa, output = y_hat la ultimul pas.
    """

    def __init__(self, input_dim, hidden_dim=16, lr=0.005, seed=42):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        self.Wxh = rng.normal(0, 1 / np.sqrt(max(1, input_dim)), size=(input_dim, hidden_dim))
        self.Whh = rng.normal(0, 1 / np.sqrt(max(1, hidden_dim)), size=(hidden_dim, hidden_dim))
        self.Wyh = rng.normal(0, 1 / np.sqrt(1), size=(1, hidden_dim))  # y_prev scalar -> hidden
        self.bh = np.zeros((hidden_dim,), dtype=float)

        self.Why = rng.normal(0, 1 / np.sqrt(max(1, hidden_dim)), size=(hidden_dim, 1))
        self.by = np.zeros((1,), dtype=float)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _dtanh_from_h(h):
        # derivative tanh(pre) = 1 - tanh(pre)^2, iar h = tanh(pre)
        return 1.0 - h * h

    def forward_one(self, X_seq):
        """
        Returneaza:
          y_hat (scalar)
          cache pentru backprop (X_seq, hs, yprevs)
        """
        T = X_seq.shape[0]
        hs = np.zeros((T, self.hidden_dim), dtype=float)
        yprevs = np.zeros((T,), dtype=float)

        y_prev = 0.0
        h_prev = np.zeros((self.hidden_dim,), dtype=float)

        for t in range(T):
            pre = X_seq[t] @ self.Wxh + h_prev @ self.Whh + np.array([y_prev], dtype=float) @ self.Wyh + self.bh
            h = self._tanh(pre)
            y = float((h @ self.Why + self.by).item())

            hs[t] = h
            yprevs[t] = y_prev

            h_prev = h
            y_prev = y

        y_hat = y_prev
        cache = (X_seq, hs, yprevs)
        return y_hat, cache

    def predict(self, X):
        out = np.zeros((X.shape[0],), dtype=float)
        for i in range(X.shape[0]):
            out[i] = self.forward_one(X[i])[0]
        return out

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=250, print_every=50):
        hist = {"train_loss": [], "val_loss": []}
        rng = np.random.default_rng(42)

        if len(X_train) == 0:
            return hist

        for ep in range(1, epochs + 1):
            idx = rng.permutation(len(X_train))
            total_loss = 0.0

            for i in idx:
                y_hat, cache = self.forward_one(X_train[i])
                y_true = float(y_train[i])
                err = y_hat - y_true
                total_loss += 0.5 * err * err

                X_seq, hs, yprevs = cache
                T = X_seq.shape[0]

                # gradient output
                d_yhat = err
                dWhy = hs[-1].reshape(-1, 1) * d_yhat
                dby = np.array([d_yhat], dtype=float)

                # backprop into last hidden
                dh_next = self.Why.flatten() * d_yhat

                dWxh = np.zeros_like(self.Wxh)
                dWhh = np.zeros_like(self.Whh)
                dWyh = np.zeros_like(self.Wyh)
                dbh = np.zeros_like(self.bh)

                # BPTT simplificat (stabil):
                # - propagam doar prin h_{t-1} (Whh)
                # - folosim gradient local pentru Wyh (depinde de y_{t-1} ca input exogen la pasul t)
                # - nu propagam gradientul prin lantul y_prev (adica nu urmarim cum y_{t-1} depinde de parametri)
                for t in reversed(range(T)):
                    h_t = hs[t]
                    h_prev = hs[t - 1] if t > 0 else np.zeros((self.hidden_dim,), dtype=float)
                    x_t = X_seq[t]
                    y_prev_input = yprevs[t]  # y_{t-1} folosit ca input la pasul t

                    dt = dh_next * self._dtanh_from_h(h_t)  # (hidden_dim,)

                    dbh += dt
                    dWxh += np.outer(x_t, dt)
                    dWhh += np.outer(h_prev, dt)
                    dWyh += np.outer(np.array([y_prev_input], dtype=float), dt)

                    dh_next = dt @ self.Whh.T

                # update
                self.Wxh -= self.lr * dWxh
                self.Whh -= self.lr * dWhh
                self.Wyh -= self.lr * dWyh
                self.bh -= self.lr * dbh
                self.Why -= self.lr * dWhy
                self.by -= self.lr * dby

            train_loss = total_loss / max(1, len(X_train))
            hist["train_loss"].append(float(train_loss))

            if X_val is not None and len(X_val) > 0:
                pv = self.predict(X_val)
                val_loss = 0.5 * float(np.mean((pv - y_val) ** 2))
                hist["val_loss"].append(val_loss)

            if ep == 1 or ep % print_every == 0 or ep == epochs:
                if len(hist["val_loss"]) > 0:
                    print(f"Epoch {ep:4d} | train_loss={train_loss:.6f} | val_loss={hist['val_loss'][-1]:.6f}")
                else:
                    print(f"Epoch {ep:4d} | train_loss={train_loss:.6f}")

        return hist


def eval_metrics(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    return rmse, mae, r2


def main():
    t0 = time.time()
    print("=== MODEL: RNN JORDAN FROM SCRATCH (numpy) ===")

    df = pd.read_csv("dataset_features.csv")

    # FIX IMPORTANT: curata df inainte sa faci secventele (altfel ajung Inf/NaN in Xtr)
    fix_inf_nan_inplace(df)

    feat = select_features_anti_leakage(df)

    seq_len = 3
    Xtr, ytr, Xva, yva, Xte, yte = make_sequences_global(df, feat, seq_len=seq_len)
    print("Seq shapes:", Xtr.shape, ytr.shape, Xva.shape, yva.shape, Xte.shape, yte.shape)

    # Protectie extra: elimina orice non-finite ramas
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

    if len(Xtr) == 0:
        raise RuntimeError("Nu exista secvente in train. Verifica dataset_features.csv si seq_len.")

    # scale X using train only
    scaler = MinMaxScaler()
    n_feat = Xtr.shape[2]

    # debug rapid
    if not np.isfinite(Xtr).all():
        raise ValueError("Xtr contine inca valori non-finite dupa curatare. Cauta coloanele problematice.")

    scaler.fit(Xtr.reshape(-1, n_feat))

    Xtr_s = scaler.transform(Xtr.reshape(-1, n_feat)).reshape(Xtr.shape)
    Xva_s = scaler.transform(Xva.reshape(-1, n_feat)).reshape(Xva.shape) if len(Xva) else Xva
    Xte_s = scaler.transform(Xte.reshape(-1, n_feat)).reshape(Xte.shape) if len(Xte) else Xte

    model = JordanRNNRegressor(input_dim=len(feat), hidden_dim=16, lr=0.005, seed=42)
    hist = model.train(Xtr_s, ytr, Xva_s, yva, epochs=250, print_every=50)

    pred_val = model.predict(Xva_s) if len(Xva_s) else np.array([])
    pred_test = model.predict(Xte_s) if len(Xte_s) else np.array([])

    print("\n=== EVALUARE RNN JORDAN SCRATCH ===")
    rmse_v, mae_v, r2_v = eval_metrics(yva, pred_val, "VAL") if len(pred_val) else (np.nan, np.nan, np.nan)
    rmse_t, mae_t, r2_t = eval_metrics(yte, pred_test, "TEST") if len(pred_test) else (np.nan, np.nan, np.nan)

    out = pd.DataFrame([{
        "model": "RNN_Jordan_scratch",
        "seq_len": seq_len,
        "hidden_dim": 16,
        "lr": 0.005,
        "rmse_val": rmse_v, "mae_val": mae_v, "r2_val": r2_v,
        "rmse_test": rmse_t, "mae_test": mae_t, "r2_test": r2_t,
        "runtime_sec": round(time.time() - t0, 3)
    }])
    out.to_csv("metrics_rnn_jordan_scratch.csv", index=False)
    print("Saved: metrics_rnn_jordan_scratch.csv")

    # plots
    plt.figure()
    plt.plot(hist["train_loss"])
    if len(hist["val_loss"]) > 0:
        plt.plot(hist["val_loss"])
        plt.legend(["train_loss", "val_loss"])
    plt.title("RNN Jordan Scratch: Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (0.5*MSE)")
    plt.tight_layout()
    plt.savefig("rnn_jordan_loss_curve.png", dpi=150)
    plt.show()

    if len(pred_test):
        plt.figure()
        plt.scatter(yte, pred_test, s=18, alpha=0.7)
        plt.title("RNN Jordan Scratch: Actual vs Predicted (Test)")
        plt.xlabel("Actual total (TEST)")
        plt.ylabel("Predicted total (TEST)")
        plt.tight_layout()
        plt.savefig("rnn_jordan_actual_vs_pred_test.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
