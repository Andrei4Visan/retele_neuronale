# model_rnn_elman_scratch.py
# RNN Elman FROM SCRATCH (numpy) for regression (predict "total")
# Uses dataset_features.csv (already created).
# Builds sequences per country and predicts next-year total (y at end of sequence).
#
# Run: python model_rnn_elman_scratch.py

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
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_before = int(dset.isna().sum().sum())

    for col in dset.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(dset[col]):
            dset[col] = dset[col].fillna(dset[col].median())

    nan_after = int(dset.isna().sum().sum())
    return nan_before, nan_after


def clip_extremes_inplace(dset: pd.DataFrame, skip_cols=("country", "year"), q_low=0.001, q_high=0.999):
    # optional safety clip to reduce extreme values
    for col in dset.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(dset[col]):
            lo = dset[col].quantile(q_low)
            hi = dset[col].quantile(q_high)
            dset[col] = dset[col].clip(lo, hi)


def select_features_anti_leakage(df_feat: pd.DataFrame):
    # drop_cols includes target and components that sum to total (avoid leakage)
    drop_cols = [
        "country", "year", "total",
        "industry", "transport", "services", "households", "agriculture",
    ]
    feature_cols = [c for c in df_feat.columns if (c not in drop_cols) and (not c.startswith("total_"))]
    return feature_cols


def make_sequences_global(df, feature_cols, target_col="total", seq_len=3, train_end=2018, val_end=2021):
    """
    Build sequences on entire dataset, per country sorted by year.
    A sequence goes to Train/Val/Test depending on end_year (year of last timestep).
    """
    X_tr, y_tr = [], []
    X_va, y_va = [], []
    X_te, y_te = [], []

    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    for _, g in df.groupby("country"):
        g = g.sort_values("year").reset_index(drop=True)

        if len(g) < seq_len:
            continue

        Xg = g[feature_cols].to_numpy(dtype=float)
        yg = g[target_col].to_numpy(dtype=float)
        years = g["year"].to_numpy(dtype=int)

        for t in range(seq_len - 1, len(g)):
            x_seq = Xg[t - (seq_len - 1): t + 1]  # (seq_len, n_features)
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

    X_tr = np.array(X_tr, dtype=float)
    y_tr = np.array(y_tr, dtype=float)
    X_va = np.array(X_va, dtype=float)
    y_va = np.array(y_va, dtype=float)
    X_te = np.array(X_te, dtype=float)
    y_te = np.array(y_te, dtype=float)

    return X_tr, y_tr, X_va, y_va, X_te, y_te


class ElmanRNNRegressor:
    """
    Simple Elman RNN:
      h_t = tanh(x_t Wxh + h_{t-1} Whh + b_h)
      y   = h_last Why + b_y
    Trained with SGD on MSE (0.5 * error^2).
    """

    def __init__(self, input_dim, hidden_dim=16, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)

        self.Wxh = rng.normal(0.0, 1.0 / np.sqrt(self.input_dim), size=(self.input_dim, self.hidden_dim))
        self.Whh = rng.normal(0.0, 1.0 / np.sqrt(self.hidden_dim), size=(self.hidden_dim, self.hidden_dim))
        self.bh = np.zeros((self.hidden_dim,), dtype=float)

        self.Why = rng.normal(0.0, 1.0 / np.sqrt(self.hidden_dim), size=(self.hidden_dim, 1))
        self.by = np.zeros((1,), dtype=float)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _dtanh(tanh_x):
        # derivative wrt pre-activation when we already have tanh(pre)
        return 1.0 - tanh_x * tanh_x

    def forward_one(self, X_seq):
        """
        X_seq: (T, D)
        returns: y_hat (float), cache (X_seq, hs)
        """
        T = X_seq.shape[0]
        hs = np.zeros((T, self.hidden_dim), dtype=float)

        h_prev = np.zeros((self.hidden_dim,), dtype=float)
        for t in range(T):
            pre = X_seq[t] @ self.Wxh + h_prev @ self.Whh + self.bh
            h = self._tanh(pre)
            hs[t] = h
            h_prev = h

        y_hat = (hs[-1] @ self.Why + self.by).item()
        cache = (X_seq, hs)
        return y_hat, cache

    def predict(self, X):
        preds = np.zeros((X.shape[0],), dtype=np.float64)
        for i in range(X.shape[0]):
            y_hat, _ = self.forward_one(X[i])
            preds[i] = y_hat
        return preds

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=300, print_every=50):
        history = {"train_loss": [], "val_loss": []}

        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train is empty. Cannot train.")

        for ep in range(1, epochs + 1):
            idx = np.random.permutation(len(X_train))
            total_loss = 0.0

            for i in idx:
                y_hat, cache = self.forward_one(X_train[i])
                y_true = float(y_train[i])

                err = (y_hat - y_true)
                loss = 0.5 * err * err
                total_loss += loss

                X_seq, hs = cache
                T = X_seq.shape[0]

                d_yhat = err
                dWhy = hs[-1].reshape(-1, 1) * d_yhat
                dby = np.array([d_yhat], dtype=float)

                dh_next = (self.Why.flatten() * d_yhat)  # (H,)

                dWxh = np.zeros_like(self.Wxh)
                dWhh = np.zeros_like(self.Whh)
                dbh = np.zeros_like(self.bh)

                for t in reversed(range(T)):
                    h_t = hs[t]
                    h_prev = hs[t - 1] if t > 0 else np.zeros((self.hidden_dim,), dtype=float)
                    x_t = X_seq[t]

                    dh = dh_next
                    dt = dh * self._dtanh(h_t)

                    dbh += dt
                    dWxh += np.outer(x_t, dt)
                    dWhh += np.outer(h_prev, dt)

                    dh_next = dt @ self.Whh.T

                # SGD update
                self.Wxh -= self.lr * dWxh
                self.Whh -= self.lr * dWhh
                self.bh -= self.lr * dbh
                self.Why -= self.lr * dWhy
                self.by -= self.lr * dby

            avg_train = total_loss / max(1, len(X_train))
            history["train_loss"].append(float(avg_train))

            if X_val is not None and y_val is not None and len(X_val) > 0:
                preds_val = self.predict(X_val)
                val_loss = 0.5 * float(np.mean((preds_val - y_val) ** 2))
                history["val_loss"].append(val_loss)

            if ep % print_every == 0 or ep == 1 or ep == epochs:
                if X_val is not None and y_val is not None and len(history["val_loss"]) > 0:
                    print(f"Epoch {ep:4d} | train_loss={avg_train:.6f} | val_loss={history['val_loss'][-1]:.6f}")
                else:
                    print(f"Epoch {ep:4d} | train_loss={avg_train:.6f}")

        return history


def eval_metrics(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    return rmse, mae, r2


def main():
    t0 = time.time()
    print("=== MODEL: RNN ELMAN FROM SCRATCH (numpy) ===")

    # 1) Load
    path = "dataset_features.csv"
    df_feat = pd.read_csv(path)
    print("Read:", path, "Shape:", df_feat.shape)

    # 1.1) Global cleaning (important because sequences are built from df_feat)
    df_feat = df_feat.copy()
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN with median on numeric columns (exclude country/year)
    skip_cols = {"country", "year"}
    for col in df_feat.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(df_feat[col]):
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

    # Optional clip extremes for stability
    clip_extremes_inplace(df_feat, skip_cols=("country", "year"), q_low=0.001, q_high=0.999)

    # Sanity check: non-finite numeric values must be 0
    num_bad = np.sum(~np.isfinite(df_feat.select_dtypes(include=[np.number]).to_numpy()))
    print("Non-finite values in df_feat (must be 0):", int(num_bad))
    if int(num_bad) != 0:
        raise ValueError("There are still non-finite numeric values in df_feat. Fix data cleaning.")

    # 2) Split (audit only)
    train, val, test = temporal_split(df_feat, train_end=2018, val_end=2021)
    print("Train:", train.shape, "Val:", val.shape, "Test:", test.shape)

    # 3) Features (anti-leakage)
    feature_cols = select_features_anti_leakage(df_feat)
    print("Num features:", len(feature_cols))
    print("Features starting with total_ (must be 0):", sum(c.startswith("total_") for c in feature_cols))

    # 4) Fix NaN after split (audit; df_feat is already cleaned)
    for name, dset in [("train", train), ("val", val), ("test", test)]:
        nb, na = fix_inf_nan_inplace(dset, skip_cols=("country", "year"))
        print(f"{name}: NaN before={nb}, after={na}")

    # 5) Build sequences GLOBAL (correct for val/test)
    seq_len = 3
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = make_sequences_global(
        df_feat,
        feature_cols=feature_cols,
        target_col="total",
        seq_len=seq_len,
        train_end=2018,
        val_end=2021
    )
    print("Seq shapes:",
          X_train_seq.shape, y_train_seq.shape,
          X_val_seq.shape, y_val_seq.shape,
          X_test_seq.shape, y_test_seq.shape)

    if len(X_train_seq) == 0:
        raise ValueError("No training sequences were created. Check seq_len and your data coverage.")

    # 6) Scaling on train only (all timesteps)
    scaler = MinMaxScaler()
    n_feat = X_train_seq.shape[2]
    scaler.fit(X_train_seq.reshape(-1, n_feat))

    X_train_seq_s = scaler.transform(X_train_seq.reshape(-1, n_feat)).reshape(X_train_seq.shape)

    if len(X_val_seq) > 0:
        X_val_seq_s = scaler.transform(X_val_seq.reshape(-1, n_feat)).reshape(X_val_seq.shape)
    else:
        X_val_seq_s = X_val_seq

    if len(X_test_seq) > 0:
        X_test_seq_s = scaler.transform(X_test_seq.reshape(-1, n_feat)).reshape(X_test_seq.shape)
    else:
        X_test_seq_s = X_test_seq

    # 7) Train RNN
    model = ElmanRNNRegressor(input_dim=n_feat, hidden_dim=16, lr=0.001, seed=42)
    history = model.train(
        X_train_seq_s, y_train_seq,
        X_val_seq_s, y_val_seq,
        epochs=150, print_every=50
    )

    # 8) Evaluate
    pred_val = model.predict(X_val_seq_s) if len(X_val_seq_s) else np.array([])
    pred_test = model.predict(X_test_seq_s) if len(X_test_seq_s) else np.array([])

    print("\n=== EVALUATION: RNN ELMAN SCRATCH ===")
    if len(pred_val):
        rmse_v, mae_v, r2_v = eval_metrics(y_val_seq, pred_val, "VAL")
    else:
        rmse_v = mae_v = r2_v = np.nan
        print("VAL -> no sequences (0)")

    if len(pred_test):
        rmse_t, mae_t, r2_t = eval_metrics(y_test_seq, pred_test, "TEST")
    else:
        rmse_t = mae_t = r2_t = np.nan
        print("TEST -> no sequences (0)")

    # 9) Save metrics
    metrics = pd.DataFrame([{
        "model": "RNN_Elman_scratch",
        "seq_len": seq_len,
        "hidden_dim": 16,
        "lr": 0.001,
        "rmse_val": rmse_v, "mae_val": mae_v, "r2_val": r2_v,
        "rmse_test": rmse_t, "mae_test": mae_t, "r2_test": r2_t,
        "runtime_sec": round(time.time() - t0, 3)
    }])
    metrics.to_csv("metrics_rnn_elman_scratch.csv", index=False)
    print("Saved metrics to metrics_rnn_elman_scratch.csv")

    # 10) Plots
    plt.figure()
    plt.plot(history["train_loss"])
    if len(history["val_loss"]) > 0:
        plt.plot(history["val_loss"])
        plt.legend(["train_loss", "val_loss"])
    plt.title("RNN Elman Scratch: Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (0.5*MSE)")
    plt.tight_layout()
    plt.savefig("rnn_elman_loss_curve.png", dpi=150)
    plt.show()

    if len(pred_test):
        plt.figure()
        plt.scatter(y_test_seq, pred_test, s=18, alpha=0.7)
        plt.title("RNN Elman Scratch: Actual vs Predicted (Test)")
        plt.xlabel("Actual total (TEST)")
        plt.ylabel("Predicted total (TEST)")
        plt.tight_layout()
        plt.savefig("rnn_elman_actual_vs_pred_test.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
