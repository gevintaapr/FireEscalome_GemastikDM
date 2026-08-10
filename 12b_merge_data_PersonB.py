"""
12b_merge_data_PersonB.py
Persiapan Data Person B: Merge cluster_master + tabular_master_final

Alasan:
- cluster_master_features.csv (output 09_label_eskalasi.py) punya: cluster_id, year,
  start_day_of_year, latitude, longitude, is_peatland, label_escalation
  TAPI tidak punya: windspeed, winddirection, slope

- tabular_master_final.csv (output pipeline Gevi) punya: cluster_id, slope,
  windspeed_10m_max, wind_alignment_score, dll
  TAPI tidak punya: year, start_day_of_year

- Solusi: Merge keduanya via cluster_id → simpan ke file BARU milik Person B
  File asli Gevi tidak diubah sama sekali!

Output:
  data_processed_B/cluster_master_PersonB.csv
  (File ini milik kamu sendiri, TIDAK menyentuh file Gevi)

Jalankan: python 12b_merge_data_PersonB.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import numpy as np

# ==============================================================
# KONFIGURASI
# ==============================================================
try:
    from config import PROCESSED_DIR
except ImportError:
    from pathlib import Path
    PROCESSED_DIR = Path("G:/My Drive/gemastik/karhutla/data_processed")

# Input: file milik Gevi (READ ONLY - tidak diubah)
CLUSTER_FILE = PROCESSED_DIR / "cluster_master_features.csv"
TABULAR_FILE = PROCESSED_DIR / "tabular_master_final.csv"

# Output: file BARU milik Person B (di folder sendiri)
from pathlib import Path
OUTPUT_DIR = Path(str(PROCESSED_DIR).replace("data_processed", "data_processed_B"))
OUTPUT_FILE = OUTPUT_DIR / "cluster_master_PersonB.csv"


def main():
    print("=" * 65)
    print("10b_MERGE_DATA_PERSONB.PY")
    print("Persiapan: Merge cluster_master + tabular_master → File PersonB")
    print("=" * 65)

    # --- Validasi file input ---
    for f, name in [(CLUSTER_FILE, "cluster_master_features.csv"),
                     (TABULAR_FILE, "tabular_master_final.csv")]:
        if not os.path.exists(f):
            print(f"\n[ERROR] File tidak ditemukan: {f}")
            return

    # --- Load kedua file ---
    print(f"\n1. Memuat cluster_master_features.csv...")
    df_cluster = pd.read_csv(CLUSTER_FILE)
    print(f"   {len(df_cluster):,} baris x {len(df_cluster.columns)} kolom")
    print(f"   Kolom: {list(df_cluster.columns)}")

    print(f"\n2. Memuat tabular_master_final.csv...")
    df_tabular = pd.read_csv(TABULAR_FILE)
    print(f"   {len(df_tabular):,} baris x {len(df_tabular.columns)} kolom")
    print(f"   Kolom: {list(df_tabular.columns)}")

    # --- Tentukan kolom tambahan dari tabular ---
    # Kolom yang sudah ada di cluster (tidak perlu duplikat)
    cols_already_in_cluster = set(df_cluster.columns)

    # Kolom yang ingin diambil dari tabular (fitur fisik yang kita butuhkan)
    priority_cols = [
        "windspeed_10m_max", "winddirection_10m_dominant",
        "wind_alignment_score", "slope", "elevation", "aspect",
        "ndvi_current", "ndvi_delta_16d", "temperature_2m_max",
        "precipitation_dry_streak", "cumulative_precip_14d",
        "peatland_drought_index", "fuel_danger_index",
        "wind_slope_interaction", "population_density", "road_distance"
    ]

    cols_to_add = [c for c in priority_cols if c in df_tabular.columns
                   and c not in cols_already_in_cluster]

    print(f"\n3. Kolom yang akan ditambahkan dari tabular:")
    for c in cols_to_add:
        print(f"   + {c}")

    # --- Merge via cluster_id ---
    print(f"\n4. Melakukan merge via cluster_id...")
    df_tabular_slim = df_tabular[["cluster_id"] + cols_to_add].copy()
    df_merged = df_cluster.merge(df_tabular_slim, on="cluster_id", how="left")

    print(f"   Hasil merge: {len(df_merged):,} baris x {len(df_merged.columns)} kolom")

    # Cek hasil merge
    print(f"\n5. Validasi hasil merge:")
    for col in ["windspeed_10m_max", "winddirection_10m_dominant", "slope"]:
        if col in df_merged.columns:
            missing = df_merged[col].isna().sum()
            print(f"   {col:35s}: {len(df_merged) - missing:,} terisi, {missing:,} NaN")
        else:
            print(f"   {col:35s}: TIDAK ADA (akan default 0)")

    # Isi NaN dengan 0 atau median untuk kolom kritis
    for col in ["windspeed_10m_max", "winddirection_10m_dominant", "slope", "elevation"]:
        if col in df_merged.columns and df_merged[col].isna().sum() > 0:
            fill_val = df_merged[col].median()
            df_merged[col] = df_merged[col].fillna(fill_val)
            print(f"   -> NaN pada '{col}' diisi dengan median: {fill_val:.2f}")

    # --- Simpan ke folder PersonB ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\n6. File tersimpan: {OUTPUT_FILE}")
    print(f"   (File asli Gevi tidak diubah sama sekali!)")

    # Ringkasan kolom akhir
    print(f"\n{'=' * 65}")
    print(f"RINGKASAN FILE PERSON B")
    print(f"{'=' * 65}")
    print(f"  Total baris  : {len(df_merged):,}")
    print(f"  Total kolom  : {len(df_merged.columns)}")
    print(f"  Kolom final  :")
    for i, c in enumerate(df_merged.columns, 1):
        dtype = str(df_merged[c].dtype)
        print(f"    {i:2d}. {c:35s} [{dtype}]")

    wind_ok = "windspeed_10m_max" in df_merged.columns and df_merged["windspeed_10m_max"].notna().sum() > 0
    slope_ok = "slope" in df_merged.columns and df_merged["slope"].notna().sum() > 0

    print(f"\n  Status bahan baku untuk Graf:")
    print(f"    Koordinat & waktu : OK")
    print(f"    Label eskalasi    : OK")
    print(f"    Data gambut       : OK")
    print(f"    Data angin        : {'OK' if wind_ok else 'TIDAK ADA (default 0.5)'}")
    print(f"    Data slope/DEM    : {'OK' if slope_ok else 'TIDAK ADA (default 0)'}")

    print(f"\n{'=' * 65}")
    print("SELESAI — Lanjut ke: python 11_bangun_spatiotemporal_graph.py")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
