"""
10_eksplorasi_cluster_data.py
Fase 1: Eksplorasi & Validasi Input Data untuk Step 7 (Spatio-Temporal Graph).

Cek kelengkapan kolom yang wajib ada sebelum membangun graf:
  - latitude, longitude          -> koordinat node
  - start_day_of_year, year      -> waktu node (untuk edge temporal)
  - windspeed & winddirection    -> Wind Alignment (cos theta)
  - is_peatland                  -> Peatland Continuity
  - slope / elevation            -> Slope Direction
  - label_escalation             -> untuk analisis explainability nanti

Jalankan: python 10_eksplorasi_cluster_data.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import os

# ==============================================================
# KONFIGURASI
# ==============================================================
try:
    from config import PROCESSED_DIR
    INPUT_FILE = PROCESSED_DIR / "cluster_master_features.csv"
except ImportError:
    # Fallback jika tidak ada config.py
    INPUT_FILE = "data_processed/cluster_master_features.csv"

# Kolom yang WAJIB ada untuk membangun graf
REQUIRED_COLS = {
    "latitude":             "Koordinat pusat cluster (node position)",
    "longitude":            "Koordinat pusat cluster (node position)",
    "start_day_of_year":    "Hari pertama cluster muncul (edge temporal)",
    "year":                 "Tahun cluster (edge temporal, hindari linking lintas tahun)",
    "label_escalation":     "Label 0/1 eskalasi (target analisis)",
    "growth_ratio":         "Rasio pertumbuhan cluster",
    "is_peatland":          "Indikator gambut (Peatland Continuity weight)",
}

# Kolom cuaca angin — coba beberapa nama alternatif
WIND_SPEED_CANDIDATES = [
    "windspeed_10m_max", "windspeed_10m_max_weather",
    "wind_speed_10m_max", "windspeed"
]
WIND_DIR_CANDIDATES = [
    "winddirection_10m_dominant", "winddirection_10m_dominant_weather",
    "wind_direction_10m_dominant", "winddirection"
]

# Kolom topografi — dari DEM SRTM
SLOPE_CANDIDATES = [
    "slope", "slope_deg", "slope_mean", "slope_avg",
    "elevation", "elevation_mean", "dem_slope"
]


def check_column_availability(df):
    """Cek semua kolom kritis dan laporkan status."""
    print("=" * 65)
    print("LAPORAN KETERSEDIAAN KOLOM KRITIS")
    print("=" * 65)

    all_cols = set(df.columns.tolist())
    missing_required = []

    # 1. Cek kolom wajib
    print("\n[A] KOLOM WAJIB:")
    for col, desc in REQUIRED_COLS.items():
        status = "✅" if col in all_cols else "❌ MISSING"
        print(f"  {status:10s} {col:30s} ({desc})")
        if col not in all_cols:
            missing_required.append(col)

    # 2. Cek kolom angin
    print("\n[B] KOLOM ANGIN (Wind Alignment):")
    found_speed = None
    for c in WIND_SPEED_CANDIDATES:
        if c in all_cols:
            found_speed = c
            print(f"  ✅          {c:30s} <- akan dipakai sebagai windspeed")
            break
    if not found_speed:
        print(f"  ❌ MISSING  Tidak ada kolom windspeed dari kandidat: {WIND_SPEED_CANDIDATES}")

    found_dir = None
    for c in WIND_DIR_CANDIDATES:
        if c in all_cols:
            found_dir = c
            print(f"  ✅          {c:30s} <- akan dipakai sebagai winddirection (derajat)")
            break
    if not found_dir:
        print(f"  ❌ MISSING  Tidak ada kolom winddirection dari kandidat: {WIND_DIR_CANDIDATES}")

    # 3. Cek kolom topografi
    print("\n[C] KOLOM TOPOGRAFI / SLOPE (dari DEM SRTM):")
    found_slope = None
    for c in SLOPE_CANDIDATES:
        if c in all_cols:
            found_slope = c
            print(f"  ✅          {c:30s} <- akan dipakai sebagai slope")
            break
    if not found_slope:
        print(f"  ⚠️  MISSING  Tidak ada kolom slope/elevation dari kandidat: {SLOPE_CANDIDATES}")
        print(f"             -> Perlu jalankan 10b_join_slope_ke_cluster.py")

    return missing_required, found_speed, found_dir, found_slope


def print_basic_stats(df, found_speed, found_dir, found_slope):
    """Tampilkan statistik dasar kolom kunci."""
    print("\n" + "=" * 65)
    print("STATISTIK DASAR KOLOM KUNCI")
    print("=" * 65)

    print(f"\n  Total cluster (node): {len(df):,}")
    print(f"  Rentang tahun       : {df['year'].min()} – {df['year'].max()}"
          if "year" in df.columns else "  year: MISSING")

    if "label_escalation" in df.columns:
        vc = df["label_escalation"].value_counts()
        pct = df["label_escalation"].value_counts(normalize=True) * 100
        print(f"\n  Distribusi Label:")
        print(f"    Label 1 (Eskalasi)    : {vc.get(1, 0):,} ({pct.get(1, 0):.1f}%)")
        print(f"    Label 0 (Non-Eskalasi): {vc.get(0, 0):,} ({pct.get(0, 0):.1f}%)")

    if "is_peatland" in df.columns:
        peat = df["is_peatland"].sum()
        print(f"\n  Cluster di lahan gambut: {peat:,} ({peat/len(df)*100:.1f}%)")

    if "latitude" in df.columns and "longitude" in df.columns:
        print(f"\n  Rentang koordinat:")
        print(f"    Latitude : {df['latitude'].min():.3f} – {df['latitude'].max():.3f}")
        print(f"    Longitude: {df['longitude'].min():.3f} – {df['longitude'].max():.3f}")

    if found_speed and found_speed in df.columns:
        ws = df[found_speed].dropna()
        print(f"\n  Windspeed ({found_speed}):")
        print(f"    Mean={ws.mean():.2f}, Min={ws.min():.2f}, Max={ws.max():.2f}, NaN={df[found_speed].isna().sum()}")

    if found_dir and found_dir in df.columns:
        wd = df[found_dir].dropna()
        print(f"  Winddirection ({found_dir}):")
        print(f"    Mean={wd.mean():.1f}°, Min={wd.min():.1f}°, Max={wd.max():.1f}°, NaN={df[found_dir].isna().sum()}")

    if found_slope and found_slope in df.columns:
        sl = df[found_slope].dropna()
        print(f"  Slope/Elevation ({found_slope}):")
        print(f"    Mean={sl.mean():.3f}, Min={sl.min():.3f}, Max={sl.max():.3f}, NaN={df[found_slope].isna().sum()}")

    print(f"\n  Total missing values per kolom kunci:")
    key_cols = (
        ["latitude", "longitude", "start_day_of_year", "year", "is_peatland", "label_escalation"]
        + ([found_speed] if found_speed else [])
        + ([found_dir] if found_dir else [])
        + ([found_slope] if found_slope else [])
    )
    for c in key_cols:
        if c in df.columns:
            nan_count = df[c].isna().sum()
            flag = " ⚠️" if nan_count > 0 else ""
            print(f"    {c:40s}: {nan_count:,}{flag}")


def print_all_columns(df):
    """Tampilkan semua kolom yang ada di file."""
    print("\n" + "=" * 65)
    print(f"SEMUA KOLOM YANG ADA ({len(df.columns)} kolom total):")
    print("=" * 65)
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        print(f"  {i:3d}. {col:40s} [{dtype}]")


def print_verdict(missing_required, found_speed, found_dir, found_slope):
    """Cetak verdict akhir dan instruksi langkah selanjutnya."""
    print("\n" + "=" * 65)
    print("VERDICT & LANGKAH SELANJUTNYA")
    print("=" * 65)

    issues = []
    if missing_required:
        issues.append(f"Kolom wajib missing: {missing_required}")
    if not found_speed:
        issues.append("Kolom windspeed tidak ditemukan")
    if not found_dir:
        issues.append("Kolom winddirection tidak ditemukan")
    if not found_slope:
        issues.append("Kolom slope/elevation tidak ditemukan -> perlu 10b_join_slope_ke_cluster.py")

    if not issues:
        print("\n  ✅ SEMUA KOLOM SIAP! Bisa langsung ke:")
        print("     python 11_bangun_spatiotemporal_graph.py")
    else:
        print("\n  ⚠️  Ada yang perlu dibenahi dulu:")
        for issue in issues:
            print(f"     - {issue}")

        if not found_slope:
            print("\n  📌 Untuk kolom slope: cek apakah ada file DEM CSV di data_raw/")
            print("     Kalau ada -> jalankan python 10b_join_slope_ke_cluster.py")
            print("     Kalau tidak ada -> slope akan di-set default 0 (lereng datar)")


def main():
    print("=" * 65)
    print("12_eksplorasi_data_PersonB.py")
    print("Fase 1: Validasi Input Sebelum Membangun Spatio-Temporal Graph")
    print("=" * 65)

    if not os.path.exists(INPUT_FILE):
        print(f"\n[ERROR] File tidak ditemukan: {INPUT_FILE}")
        print("Pastikan 09_label_eskalasi.py sudah dijalankan terlebih dahulu.")
        return

    print(f"\nMemuat: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Berhasil dimuat: {len(df):,} baris x {len(df.columns)} kolom\n")

    # Cek kolom
    missing_required, found_speed, found_dir, found_slope = check_column_availability(df)

    # Statistik dasar
    print_basic_stats(df, found_speed, found_dir, found_slope)

    # Tampilkan semua kolom
    print_all_columns(df)

    # Verdict
    print_verdict(missing_required, found_speed, found_dir, found_slope)

    print("\n" + "=" * 65)
    print("EKSPLORASI SELESAI")
    print("=" * 65)


if __name__ == "__main__":
    main()
