import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from config import RAW_DIR, PROCESSED_DIR

def calculate_growth_ratio(df_cluster):
    """
    Menghitung growth ratio dari sebuah cluster kebakaran.
    df_cluster: dataframe berisi hotspot yang sudah dikelompokkan dalam 1 cluster_id.
    """
    days = df_cluster['day_of_year'].values
    min_day = days.min()
    
    # Hitung jumlah hotspot pada hari pertama (H0)
    day_0_count = np.sum(days == min_day)
    
    # Cari jumlah hotspot maksimum harian dalam 3 hari berikutnya (H+1 sampai H+3)
    counts_next_3_days = [np.sum(days == d) for d in range(min_day + 1, min_day + 4)]
    max_daily_next_3_days = max(counts_next_3_days)
    
    return max_daily_next_3_days / day_0_count

def main():
    print("============================================================")
    print("09_LABEL_ESKALASI.PY")
    print("Tahap 2: Spatio-Temporal DBSCAN & Pembentukan Label P-95")
    print("============================================================\n")

    input_file = PROCESSED_DIR / "firms_master_features.csv"
    if not os.path.exists(input_file):
        print(f"[ERROR] File input tidak ditemukan: {input_file}")
        print("Pastikan kamu sudah sukses menjalankan 08_sanitasi_validasi_master.py")
        return

    print("1. Memuat data FIRMS Master Features (100% tervalidasi MCD64A1)...")
    df = pd.read_csv(input_file)
    
    if len(df) == 0:
        print("[BAHAYA] Dataset kosong!")
        return

    # Rekonstruksi kolom waktu dan spasial
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["year"] = df["acq_date"].dt.year
    df["day_of_year"] = df["acq_date"].dt.dayofyear
    df["island"] = np.where(df["longitude"] < 108.5, "Sumatra", "Kalimantan")
    
    print("\n2. Menjalankan Spatio-Temporal DBSCAN per Tahun & Region...")
    # Parameter ST-DBSCAN
    SPATIAL_EPS_DEG = 0.05  # ~5.5 km
    TEMPORAL_WINDOW_DAYS = 3
    # Agar 3 hari setara dengan jarak 5.5 km dalam hitungan Euclidean:
    TIME_WEIGHT = SPATIAL_EPS_DEG / TEMPORAL_WINDOW_DAYS 
    
    df["cluster_id"] = -1
    cluster_counter = 0
    
    for island in df["island"].unique():
        for yr in df["year"].unique():
            mask = (df["island"] == island) & (df["year"] == yr)
            if mask.sum() == 0: continue
            
            subset = df[mask].copy()
            # Buat array 3D: [lat, lon, day_of_year * TIME_WEIGHT]
            coords = subset[["latitude", "longitude"]].values
            time_scaled = (subset["day_of_year"] * TIME_WEIGHT).values.reshape(-1, 1)
            X = np.hstack((coords, time_scaled))
            
            # min_samples=1 agar 1 titik api pun dihitung sebagai kejadian kebakaran (tapi tidak bereskalasi)
            db = DBSCAN(eps=SPATIAL_EPS_DEG, min_samples=1).fit(X)
            
            # Offset label DBSCAN agar id unik secara global
            labels = db.labels_ + cluster_counter
            cluster_counter += len(np.unique(db.labels_))
            
            df.loc[mask, "cluster_id"] = labels
            
    print(f"   Terbentuk {cluster_counter} kejadian kebakaran (cluster) unik.")

    print("\n3. Menghitung Growth Ratio dan Label Eskalasi (P-95)...")
    cluster_stats = []
    
    grouped = df.groupby("cluster_id")
    for cid, group in grouped:
        growth = calculate_growth_ratio(group)
        
        min_day = group["day_of_year"].min()
        day0_data = group[group["day_of_year"] == min_day]
        
        # Fitur agregasi dasar
        stats = {
            "cluster_id": cid,
            "island": group["island"].iloc[0],
            "year": group["year"].iloc[0],
            "start_day_of_year": min_day,
            "latitude": group["latitude"].mean(),
            "longitude": group["longitude"].mean(),
            "is_peatland": group["is_peatland"].max(),
            "landcover_class": group["landcover_class"].mode()[0],
            "confidence": group["confidence"].mode()[0] if "confidence" in group.columns else "n",
            "total_hotspots": len(group),
            "growth_ratio": growth
        }
        
        # Ambil fitur cuaca dari hari pertama
        weather_cols = [c for c in df.columns if c.endswith("_weather") or c in ["temperature_2m", "relative_humidity_2m", "precipitation"]]
        for wc in weather_cols:
            if wc in day0_data.columns:
                stats[wc] = day0_data[wc].mean()
                
        cluster_stats.append(stats)
        
    df_clusters = pd.DataFrame(cluster_stats)
    
    # Hitung Persentil 95
    p95_threshold = np.percentile(df_clusters["growth_ratio"], 95)
    print(f"   Batas Potong (Cutoff) Eskalasi P-95: {p95_threshold:.2f}")
    
    # Labeling
    df_clusters["label_escalation"] = (df_clusters["growth_ratio"] >= p95_threshold).astype(int)
    
    count_1 = df_clusters["label_escalation"].sum()
    count_0 = len(df_clusters) - count_1
    print(f"   Distribusi Label: {count_1} Eskalasi (Label 1) | {count_0} Non-Eskalasi (Label 0)")

    print("\n4. Menyimpan dataset master level cluster...")
    output_path = PROCESSED_DIR / "cluster_master_features.csv"
    df_clusters.to_csv(output_path, index=False)
    print(f"   Sukses! File tersimpan di: {output_path}")
    print("   Dataset ini (berbasis per kejadian kebakaran) sudah siap masuk model Machine Learning!")

if __name__ == "__main__":
    main()
