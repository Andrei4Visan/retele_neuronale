import pandas as pd
import numpy as np

FILE_PATH = "data.xlsx"

# 1. Citim fisierul Excel
xls = pd.ExcelFile(FILE_PATH)

# 2. Citim fiecare sheet
df_industry = pd.read_excel(xls, sheet_name="Industry", index_col=0)
df_transport = pd.read_excel(xls, sheet_name="Transport", index_col=0)
df_services = pd.read_excel(xls, sheet_name="Services", index_col=0)
df_households = pd.read_excel(xls, sheet_name="Households", index_col=0)
df_agriculture = pd.read_excel(xls, sheet_name="Agriculture", index_col=0)
df_total = pd.read_excel(xls, sheet_name="Total", index_col=0)

# 3. Verificare rapida
print("Dimensiuni Industry:", df_industry.shape)
print("Dimensiuni Total:", df_total.shape)
print("Tari:", df_industry.index.tolist())
print("Ani:", df_industry.columns[:5], "...", df_industry.columns[-5:])

def to_long(df_wide, value_name):
    df_wide = df_wide.copy()
    df_wide.index.name = "country"

    df_long = (
        df_wide.reset_index()
        .melt(id_vars="country", var_name="year", value_name=value_name)
    )

    df_long["year"] = df_long["year"].astype(int)
    return df_long


# 1) Transformam fiecare sheet in format long
industry_long = to_long(df_industry, "industry")
transport_long = to_long(df_transport, "transport")
services_long = to_long(df_services, "services")
households_long = to_long(df_households, "households")
agriculture_long = to_long(df_agriculture, "agriculture")
total_long = to_long(df_total, "total")

# 2) Le combinam intr-un singur tabel
df = industry_long.merge(transport_long, on=["country", "year"])
df = df.merge(services_long, on=["country", "year"])
df = df.merge(households_long, on=["country", "year"])
df = df.merge(agriculture_long, on=["country", "year"])
df = df.merge(total_long, on=["country", "year"])

# 3) Sortam frumos
df = df.sort_values(["country", "year"]).reset_index(drop=True)

print("\nDataset final (long) - primele 10 randuri:")
print(df.head(10))

print("\nDimensiuni dataset final:", df.shape)
print("Coloane:", df.columns.tolist())

# Test: total trebuie sa fie suma sectoarelor
df["total_calc"] = (
    df["industry"] + df["transport"] + df["services"] + df["households"] + df["agriculture"]
)

max_diff = (df["total"] - df["total_calc"]).abs().max()
print("\nMax diferenta intre total si suma sectoarelor:", max_diff)

if max_diff < 1e-6:
    print("OK: Total = suma sectoarelor (perfect)")
else:
    print("ATENTIE: Total nu este egal cu suma sectoarelor, verifica datele")

# =========================
# PASUL 4: Audit initial
# =========================

print("\n=== AUDIT INITIAL ===")

# 1. Dimensiuni
print("Numar observatii:", df.shape[0])
print("Numar variabile:", df.shape[1])

# 2. Tipuri de date
print("\nTipuri de date:")
print(df.dtypes)

# 3. Valori lipsa
print("\nValori lipsa pe coloane:")
print(df.isnull().sum())

# 4. Statistici descriptive pentru variabilele numerice
print("\nStatistici descriptive:")
print(df.describe())

# 5. Verificare valori minime si maxime (potentiali outlieri)
print("\nMinime:")
print(df.min(numeric_only=True))

print("\nMaxime:")
print(df.max(numeric_only=True))

# =========================
# PASUL 5: Curatare finala
# =========================

df = df.drop(columns=["total_calc"])

# Salvam dataset-ul de baza (curat)
df.to_csv("dataset_baza.csv", index=False)

print("\nDataset de baza salvat ca dataset_baza.csv")
print("Dimensiuni dataset baza:", df.shape)

# =========================
# PASUL 6: Feature engineering
# =========================

print("\n=== PASUL 6: FEATURE ENGINEERING ===")

df_feat = df.copy()

sectors = ["industry", "transport", "services", "households", "agriculture"]
target = "total"

# 1) Feature de timp (normalizam anul intre 0 si 1)
df_feat["year_norm"] = (df_feat["year"] - df_feat["year"].min()) / (df_feat["year"].max() - df_feat["year"].min())

# Sortam ca sa putem face lag-uri corect
df_feat = df_feat.sort_values(["country", "year"]).reset_index(drop=True)

# 2) Lag-uri (t-1 si t-2)
for col in sectors + [target]:
    df_feat[f"{col}_lag1"] = df_feat.groupby("country")[col].shift(1)
    df_feat[f"{col}_lag2"] = df_feat.groupby("country")[col].shift(2)

# 3) Crestere anuala (YoY)
for col in sectors + [target]:
    df_feat[f"{col}_yoy"] = df_feat.groupby("country")[col].pct_change(1)

# 4) Ponderi in total (share)
for col in sectors:
    df_feat[f"{col}_share"] = df_feat[col] / df_feat[target]

# 5) Medii mobile pe 3 ani (rolling mean)
window = 3
for col in sectors + [target]:
    df_feat[f"{col}_rm{window}"] = (
        df_feat.groupby("country")[col]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

# Dupa feature engineering apar NaN la primii ani (din lag/rolling)
print("Dimensiuni inainte de dropna:", df_feat.shape)
print("Numar valori lipsa total (din lag/rolling):", int(df_feat.isna().sum().sum()))

# Scoatem randurile care au NaN generate de lag/rolling
df_feat = df_feat.dropna().reset_index(drop=True)

print("Dimensiuni dupa dropna:", df_feat.shape)
print("Numar coloane finale (features + target):", df_feat.shape[1])

# Salvam dataset-ul cu features
df_feat.to_csv("dataset_features.csv", index=False)
print("Dataset cu features salvat ca dataset_features.csv")

# =========================
# PASUL 7: Split temporal (date noi)
# =========================

print("\n=== PASUL 7: SPLIT TEMPORAL ===")

train_end = 2018
val_end = 2021

train = df_feat[df_feat["year"] <= train_end].copy()
val = df_feat[(df_feat["year"] > train_end) & (df_feat["year"] <= val_end)].copy()
test = df_feat[df_feat["year"] > val_end].copy()

print("Train:", train.shape)
print("Val:", val.shape)
print("Test:", test.shape)
# =========================
# PASUL 8.5: TRATARE INF / NAN (pe split-uri)
# =========================

print("\n=== PASUL 8.5: TRATARE INF / NAN (pe train/val/test) ===")

for dname, dset in [("train", train), ("val", val), ("test", test)]:
    dset.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_before = int(dset.isna().sum().sum())

    for col in dset.columns:
        if col not in ["country", "year"]:
            dset[col] = dset[col].fillna(dset[col].median())

    nan_after = int(dset.isna().sum().sum())
    print(f"{dname}: NaN inainte={nan_before}, dupa imputare={nan_after}")


# =========================
# PASUL 8: X, y + scalare (MinMax)
# =========================

from sklearn.preprocessing import MinMaxScaler

print("\n=== PASUL 8: SCALARE ===")

drop_cols = [
    "country",
    "year",
    "total",
    "industry",
    "transport",
    "services",
    "households",
    "agriculture"
]

feature_cols = [
    c for c in df_feat.columns
    if (c not in drop_cols) and (not c.startswith("total_"))
]

print("Features care incep cu total_ (trebuie sa fie 0):", sum([c.startswith("total_") for c in feature_cols]))


X_train = train[feature_cols].values
y_train = train["total"].values

X_val = val[feature_cols].values
y_val = val["total"].values

X_test = test[feature_cols].values
y_test = test["total"].values

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print("Nr features:", len(feature_cols))
print("X_train_s shape:", X_train_s.shape)
print("y_train shape:", y_train.shape)

# =========================
# PASUL 9: Baseline - regresie liniara
# =========================
print("\n=== TEST RECONSTRUCTIE TOTAL din LAG1 + YOY ===")

recon_total = np.zeros_like(y_train, dtype=float)

ok = True
for s in sectors:
    c_lag = f"{s}_lag1"
    c_yoy = f"{s}_yoy"
    if (c_lag not in feature_cols) or (c_yoy not in feature_cols):
        ok = False
        print("Lipseste:", c_lag, "sau", c_yoy)

if ok:
    for s in sectors:
        i_lag = feature_cols.index(f"{s}_lag1")
        i_yoy = feature_cols.index(f"{s}_yoy")
        sector_t = X_train[:, i_lag] * (1.0 + X_train[:, i_yoy])
        recon_total += sector_t

    diff = np.max(np.abs(recon_total - y_train))
    print("Max diff recon_total vs y_train:", diff)
else:
    print("Nu pot testa complet, nu sunt toate coloanele lag1+yoy in feature_cols.")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n=== DIAGNOSTIC LEAKAGE ===")

# 1) Exista vreo coloana identica cu target-ul?
same_as_y = []
for i, col in enumerate(feature_cols):
    if np.allclose(X_train[:, i], y_train, atol=1e-9):
        same_as_y.append(col)

print("Coloane identice cu y_train:", same_as_y)

# 2) Exista vreo coloana care e aproape identica (dif max foarte mic)?
almost_same = []
for i, col in enumerate(feature_cols):
    diff_max = np.max(np.abs(X_train[:, i] - y_train))
    if diff_max < 1e-6:
        almost_same.append((col, diff_max))

print("Coloane aproape identice cu y_train (diff_max < 1e-6):", almost_same)

# 3) Verificam daca exista valori extreme care fac modelul "perfect" din cauza numericii
print("Max abs y_train:", float(np.max(np.abs(y_train))))
print("Max abs X_train:", float(np.max(np.abs(X_train))))


lin = LinearRegression()
lin.fit(X_train_s, y_train)

pred_val = lin.predict(X_val_s)
pred_test = lin.predict(X_test_s)

rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
mae_val = mean_absolute_error(y_val, pred_val)
r2_val = r2_score(y_val, pred_val)

rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
mae_test = mean_absolute_error(y_test, pred_test)
r2_test = r2_score(y_test, pred_test)

print("VAL -> RMSE:", rmse_val, "MAE:", mae_val, "R2:", r2_val)
print("TEST -> RMSE:", rmse_test, "MAE:", mae_test, "R2:", r2_test)

metrics_df = pd.DataFrame([
    {"model": "LinearRegression", "split": "val", "RMSE": rmse_val, "MAE": mae_val, "R2": r2_val},
    {"model": "LinearRegression", "split": "test", "RMSE": rmse_test, "MAE": mae_test, "R2": r2_test},
])

metrics_df.to_csv("metrics.csv", index=False)
print("Metrici salvate in metrics.csv")
