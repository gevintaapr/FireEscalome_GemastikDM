"""
02_feature_engineering_modeling.py
Bangun grid spasial, label eskalasi, fitur graf spasial-temporal,
lalu latih LightGBM dengan cost-sensitive weighting untuk extreme imbalance.

Jalankan SETELAH 01_fetch_data.py selesai dan data_raw/firms_all.csv ada isinya.
Jalankan: python 02_feature_engineering_modeling.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# KONFIGURASI — kalibrasi ulang ambang ini begitu lihat data riil
# ============================================================
GRID_SIZE = 0.1          # ~11 km per sel; perkecil kalau data padat, perbesar kalau tipis
WINDOW_FORWARD = 3        # horizon "beberapa hari ke depan" untuk deteksi eskalasi
ESCALATION_MULTIPLIER = 3 # kelipatan lonjakan minimum dianggap "eskalasi"
MIN_ESCALATED_COUNT = 5   # ambang absolut minimum supaya "eskalasi" nggak trivial (mis. 1->3)
NEIGHBOR_RADIUS_CELLS = 1 # radius sel tetangga untuk fitur graf spasial


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(path: str = "data_raw/firms_all.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


# ============================================================
# 2. GRID SPASIAL + AGREGASI HARIAN
# ============================================================
def build_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grid_lat"] = (df["latitude"] // GRID_SIZE) * GRID_SIZE
    df["grid_lon"] = (df["longitude"] // GRID_SIZE) * GRID_SIZE
    df["cell_id"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    daily = (
        df.groupby(["cell_id", "grid_lat", "grid_lon", "acq_date"])
        .agg(hotspot_count=("latitude", "count"), frp_sum=("frp", "sum"))
        .reset_index()
    )

    # Lengkapi hari-hari kosong (0 titik panas) supaya time series per sel utuh
    all_cells = daily["cell_id"].unique()
    date_range = pd.date_range(daily["acq_date"].min(), daily["acq_date"].max())
    full_index = pd.MultiIndex.from_product([all_cells, date_range], names=["cell_id", "acq_date"])
    daily_full = daily.set_index(["cell_id", "acq_date"]).reindex(full_index, fill_value=0).reset_index()

    cell_coords = daily[["cell_id", "grid_lat", "grid_lon"]].drop_duplicates()
    daily_full = daily_full.drop(columns=["grid_lat", "grid_lon"]).merge(cell_coords, on="cell_id", how="left")
    return daily_full.sort_values(["cell_id", "acq_date"]).reset_index(drop=True)


# ============================================================
# 3. LABEL ESKALASI
#    PENTING: pakai kolom lead eksplisit, JANGAN shift+rolling (arahnya gampang
#    kebalik dan diam-diam salah tanpa error -- ini sudah divalidasi di sintetis).
# ============================================================
def add_escalation_label(daily_full: pd.DataFrame) -> pd.DataFrame:
    daily_full = daily_full.copy()
    lead_cols = []
    for lag in range(1, WINDOW_FORWARD + 1):
        col = f"lead_{lag}"
        daily_full[col] = daily_full.groupby("cell_id")["hotspot_count"].shift(-lag)
        lead_cols.append(col)
    daily_full["future_max_count"] = daily_full[lead_cols].max(axis=1)

    daily_full["is_small_cluster_today"] = (
        (daily_full["hotspot_count"] >= 1) & (daily_full["hotspot_count"] <= 3)
    )
    daily_full["label_escalation"] = (
        daily_full["is_small_cluster_today"]
        & (daily_full["future_max_count"] >= daily_full["hotspot_count"] * ESCALATION_MULTIPLIER)
        & (daily_full["future_max_count"] >= MIN_ESCALATED_COUNT)
    ).fillna(False).astype(int)

    return daily_full.drop(columns=lead_cols)


# ============================================================
# 4. FITUR GRAF SPASIAL (tetangga sel dalam radius tertentu)
#    Fitur graf sebagai kolom tabular ke LightGBM -- bukan GNN penuh.
#    Ini yang bikin timeline 6 hari realistis (tidak perlu setup PyTorch Geometric/GPU).
# ============================================================
def add_neighbor_features(sample_df: pd.DataFrame, daily_full: pd.DataFrame) -> pd.DataFrame:
    sample_df = sample_df.copy()
    coord_lookup = daily_full[["cell_id", "grid_lat", "grid_lon"]].drop_duplicates().set_index("cell_id")

    neighbor_map = {}
    for cid, row in coord_lookup.iterrows():
        lat, lon = row["grid_lat"], row["grid_lon"]
        mask = (
            (np.abs(coord_lookup["grid_lat"] - lat) <= NEIGHBOR_RADIUS_CELLS * GRID_SIZE)
            & (np.abs(coord_lookup["grid_lon"] - lon) <= NEIGHBOR_RADIUS_CELLS * GRID_SIZE)
            & (coord_lookup.index != cid)
        )
        neighbor_map[cid] = coord_lookup[mask].index.tolist()

    lookup = daily_full.set_index(["cell_id", "acq_date"])["hotspot_count"]

    def neighbor_density(row):
        neighbors = neighbor_map.get(row["cell_id"], [])
        if not neighbors:
            return 0.0
        vals = []
        for n in neighbors:
            vals.append(lookup.get((n, row["acq_date"]), 0))
        return float(np.mean(vals)) if vals else 0.0

    sample_df["neighbor_hotspot_density"] = sample_df.apply(neighbor_density, axis=1)
    return sample_df


# ============================================================
# 5. FITUR TEMPORAL
# ============================================================
def add_temporal_features(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample_df = sample_df.copy()
    sample_df["day_of_year"] = sample_df["acq_date"].dt.dayofyear
    sample_df["month"] = sample_df["acq_date"].dt.month
    sample_df["is_dry_season"] = sample_df["month"].isin([6, 7, 8, 9, 10]).astype(int)

    for lag in [1, 2, 3]:
        sample_df[f"hotspot_lag_{lag}"] = (
            sample_df.groupby("cell_id")["hotspot_count"].shift(lag).fillna(0)
        )
    sample_df["frp_trend"] = (
        sample_df.groupby("cell_id")["frp_sum"].transform(lambda s: s.diff().fillna(0))
    )
    return sample_df

    # TODO (opsional, kalau waktu sisa): join data_raw/weather_all.csv berdasarkan
    # region terdekat -> tambahkan precipitation_sum, windspeed_10m_max sebagai fitur.


# ============================================================
# 6. TRAIN / TEST SPLIT -- berbasis WAKTU, bukan random
#    (hindari bocor informasi masa depan, sama seperti diagnosis
#    distribution-shift yang sudah kamu lakukan di project traffic speed)
# ============================================================
FEATURE_COLS = [
    "hotspot_count", "frp_sum", "neighbor_hotspot_density",
    "day_of_year", "is_dry_season",
    "hotspot_lag_1", "hotspot_lag_2", "hotspot_lag_3", "frp_trend",
]


def time_based_split(sample_df: pd.DataFrame, train_frac: float = 0.8):
    sample_df = sample_df.sort_values("acq_date")
    split_date = sample_df["acq_date"].quantile(train_frac)
    train = sample_df[sample_df["acq_date"] <= split_date]
    test = sample_df[sample_df["acq_date"] > split_date]
    return train, test


# ============================================================
# 7. MODEL + EVALUASI
# ============================================================
def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame):
    X_train, y_train = train[FEATURE_COLS], train["label_escalation"]
    X_test, y_test = test[FEATURE_COLS], test["label_escalation"]

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    if n_pos == 0:
        raise ValueError(
            "Tidak ada label positif di training set. Longgarkan ESCALATION_MULTIPLIER "
            "atau MIN_ESCALATED_COUNT, atau perluas rentang data."
        )
    pos_weight = n_neg / n_pos
    print(f"Distribusi train -> positif: {n_pos}, negatif: {n_neg}, scale_pos_weight: {pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=pos_weight,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    print("MCC:", round(matthews_corrcoef(y_test, y_pred), 3))
    print("PR-AUC:", round(average_precision_score(y_test, y_proba), 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n=== Feature Importance ===")
    print(importance_df.to_string(index=False))

    importance_df.to_csv("hasil_feature_importance.csv", index=False)
    test.assign(pred=y_pred, proba=y_proba)[
        ["cell_id", "acq_date", "label_escalation", "pred", "proba"]
    ].to_csv("hasil_prediksi_test.csv", index=False)

    return model, importance_df


# ============================================================
# MAIN
# ============================================================
def main():
    print("1. Load data...")
    df = load_data()
    print(f"   {len(df)} baris titik panas dimuat.")

    print("2. Bangun grid spasial + agregasi harian...")
    daily_full = build_daily_grid(df)

    print("3. Label eskalasi...")
    daily_full = add_escalation_label(daily_full)
    sample_df = daily_full[daily_full["is_small_cluster_today"]].copy()
    print("   Distribusi label eskalasi (di antara cluster kecil):")
    print(sample_df["label_escalation"].value_counts())
    print(sample_df["label_escalation"].value_counts(normalize=True).round(4))

    if sample_df["label_escalation"].sum() < 20:
        print(
            "\n   PERINGATAN: kejadian positif < 20. Split train/test bisa tidak stabil.\n"
            "   Pertimbangkan longgarkan ESCALATION_MULTIPLIER/MIN_ESCALATED_COUNT\n"
            "   atau perbesar GRID_SIZE / perpanjang rentang tanggal di 01_fetch_data.py."
        )

    print("\n4. Fitur graf spasial (tetangga)...")
    sample_df = add_neighbor_features(sample_df, daily_full)

    print("5. Fitur temporal...")
    sample_df = add_temporal_features(sample_df)

    print("6. Split berbasis waktu...")
    train, test = time_based_split(sample_df)
    print(f"   Train: {len(train)} baris, Test: {len(test)} baris")

    print("\n7. Latih model + evaluasi...")
    train_and_evaluate(train, test)

    print("\n=== SELESAI ===")
    print("Hasil tersimpan: hasil_feature_importance.csv, hasil_prediksi_test.csv")


if __name__ == "__main__":
    main()
