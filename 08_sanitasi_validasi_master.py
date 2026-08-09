import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from config import RAW_DIR, PROCESSED_DIR

print("============================================================")
print("08_SANITASI_VALIDASI_MASTER.PY")
print("Tahap 4 & 5: Memvalidasi kejadian karhutla, filter Land Cover,")
print("dan menyatukan semua fitur menjadi firms_master_features.csv")
print("============================================================\n")

# Konfigurasi Jarak
SPATIAL_TOLERANCE_DEG = 0.01  # ~1.1 km tolerance untuk KDTree (sesuai resolusi satelit)
TEMPORAL_TOLERANCE_DAYS = 15  # +/- 15 hari jarak antara FIRMS dan BurnDate MCD64A1

# Titik cuaca untuk mapping hotspot ke Region cuaca terdekat
WEATHER_POINTS = {
    "Riau": (0.5, 101.4),
    "SumSel": (-3.0, 104.7),
    "Jambi": (-1.6, 103.6),
    "KalBar": (0.0, 109.3),
    "KalTeng": (-1.7, 113.3),
}

def safe_read_csv(filepath):
    # Workaround untuk file di Google Drive Stream yang kadang bikin pandas OSError 22
    try:
        return pd.read_csv(filepath)
    except OSError:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return pd.read_csv(f)

def get_nearest_region(lat, lon):
    min_dist = float('inf')
    best_region = "Riau"
    for region, (w_lat, w_lon) in WEATHER_POINTS.items():
        dist = np.sqrt((lat - w_lat)**2 + (lon - w_lon)**2)
        if dist < min_dist:
            min_dist = dist
            best_region = region
    return best_region

def step_1_to_3_validasi_dan_filter():
    firms_path = RAW_DIR / "firms_all.csv"
    if not os.path.exists(firms_path):
        print(f"[ERROR] File FIRMS tidak ditemukan di: {firms_path}")
        return None

    print("1. Memuat data FIRMS (Hotspot)...")
    df_firms = safe_read_csv(firms_path)
    df_firms["acq_date"] = pd.to_datetime(df_firms["acq_date"])
    df_firms["year"] = df_firms["acq_date"].dt.year
    df_firms["day_of_year"] = df_firms["acq_date"].dt.dayofyear
    
    df_firms["island"] = np.where(df_firms["longitude"] < 108.5, "Sumatra", "Kalimantan")
    df_firms["is_valid_burned_area"] = False
    df_firms["landcover_class"] = -1

    years = df_firms["year"].unique()
    islands = df_firms["island"].unique()

    print("\n2. Memulai proses Validasi MCD64A1 & Land Cover per Region & Tahun...")
    for island in islands:
        for yr in years:
            print(f"   -> Memproses {island} Tahun {yr}...")
            mask = (df_firms["island"] == island) & (df_firms["year"] == yr)
            if mask.sum() == 0: continue
            
            # --- VALIDASI BURNED AREA ---
            burned_file = RAW_DIR / f"BurnedArea_{island}_{yr}.csv"
            if os.path.exists(burned_file):
                df_ba = safe_read_csv(burned_file)
                if "BurnDate" in df_ba.columns and not df_ba.empty:
                    tree_ba = cKDTree(df_ba[["latitude", "longitude"]].values)
                    firms_coords = df_firms.loc[mask, ["latitude", "longitude"]].values
                    dists, idxs = tree_ba.query(firms_coords, distance_upper_bound=SPATIAL_TOLERANCE_DEG)
                    
                    for i, (dist, idx) in enumerate(zip(dists, idxs)):
                        if dist != float('inf'): 
                            burn_date_val = df_ba.iloc[idx]["BurnDate"]
                            hotspot_doy = df_firms.loc[mask].iloc[i]["day_of_year"]
                            if burn_date_val > 0 and abs(burn_date_val - hotspot_doy) <= TEMPORAL_TOLERANCE_DAYS:
                                df_firms.loc[df_firms.index[mask][i], "is_valid_burned_area"] = True
                else:
                    print(f"      [WARNING] File {burned_file.name} kosong/tidak punya BurnDate.")
            else:
                print(f"      [WARNING] File MCD64A1 tidak ditemukan: {burned_file.name}")

            # --- FILTER LAND COVER ---
            lc_file = RAW_DIR / f"LandCover_DynamicWorld_{island}_{yr}_500m.csv"
            if os.path.exists(lc_file):
                df_lc = safe_read_csv(lc_file)
                candidates = [c for c in df_lc.columns if c not in ["latitude", "longitude", "system:index", ".geo"]]
                lc_col = "LandCover_Class" if "LandCover_Class" in df_lc.columns else (candidates[0] if candidates else df_lc.columns[-1])
                if not df_lc.empty:
                    tree_lc = cKDTree(df_lc[["latitude", "longitude"]].values)
                    firms_coords = df_firms.loc[mask, ["latitude", "longitude"]].values
                    dists, idxs = tree_lc.query(firms_coords, distance_upper_bound=SPATIAL_TOLERANCE_DEG)
                    
                    for i, (dist, idx) in enumerate(zip(dists, idxs)):
                        if dist != float('inf'):
                            lc_val = df_lc.iloc[idx][lc_col]
                            df_firms.loc[df_firms.index[mask][i], "landcover_class"] = lc_val
            else:
                print(f"      [WARNING] File Land Cover tidak ditemukan: {lc_file.name}")

    print("\n3. Membuang (Filtering) area Urban/Built-up dan Data Tidak Valid...")
    total_before = len(df_firms)
    df_valid = df_firms[df_firms["is_valid_burned_area"] == True].copy()
    print(f"   Lolos Validasi Burned Area: {len(df_valid)} dari {total_before} titik.")
    
    df_filtered = df_valid[df_valid["landcover_class"] != 6].copy()
    print(f"   Lolos Filter Land Cover (Bukan Urban): {len(df_filtered)} titik.")
    
    return df_filtered


def main():
    checkpoint_1 = PROCESSED_DIR / "checkpoint_1_validasi_lc.csv"
    
    if os.path.exists(checkpoint_1):
        print(f"\n[CHECKPOINT] Ditemukan file {checkpoint_1.name}!")
        print("-> Memuat data dari checkpoint (melewati proses Spasial Join yang memakan waktu)...")
        df_filtered = safe_read_csv(checkpoint_1)
        # Pastikan kolom acq_date terbaca sebagai datetime lagi
        df_filtered["acq_date"] = pd.to_datetime(df_filtered["acq_date"])
    else:
        df_filtered = step_1_to_3_validasi_dan_filter()
        if df_filtered is None or len(df_filtered) == 0:
            print("[BAHAYA] Tidak ada data yang lolos filter!")
            return
            
        print(f"   -> Menyimpan Checkpoint 1 ke: {checkpoint_1.name}")
        df_filtered.to_csv(checkpoint_1, index=False)

    print("\n4. Mengintegrasikan Data Gambut (Indonesia_peat_lands.zip)...")
    peat_file = RAW_DIR / "Indonesia_peat_lands.zip"
    df_filtered["is_peatland"] = 0
    if os.path.exists(peat_file):
        import geopandas as gpd
        print("   Membaca Shapefile Gambut (membutuhkan waktu sejenak)...")
        gdf_peat = gpd.read_file(f"zip://{peat_file}")
        
        gdf_firms = gpd.GeoDataFrame(
            df_filtered, 
            geometry=gpd.points_from_xy(df_filtered.longitude, df_filtered.latitude),
            crs="EPSG:4326"
        )
        
        if gdf_peat.crs != gdf_firms.crs:
            gdf_peat = gdf_peat.to_crs(gdf_firms.crs)
            
        joined = gpd.sjoin(gdf_firms, gdf_peat, how="left", predicate="intersects")
        joined = joined[~joined.index.duplicated(keep='first')]
        
        df_filtered["is_peatland"] = (~joined["index_right"].isna()).astype(int)
        print(f"   Terdapat {df_filtered['is_peatland'].sum()} titik jatuh di lahan gambut.")
    else:
        print("   [WARNING] File Peta Gambut (.shp) tidak ditemukan.")

    print("\n5. Mengintegrasikan Data Cuaca (Open-Meteo)...")
    weather_file = RAW_DIR / "weather_all.csv"
    if os.path.exists(weather_file):
        df_weather = safe_read_csv(weather_file)
        df_weather["time"] = pd.to_datetime(df_weather["time"])
        
        print("   Memetakan hotspot ke stasiun cuaca terdekat...")
        df_filtered["region"] = df_filtered.apply(lambda row: get_nearest_region(row["latitude"], row["longitude"]), axis=1)
        
        df_weather = df_weather.rename(columns={"time": "acq_date"})
        df_master = df_filtered.merge(df_weather, on=["acq_date", "region"], how="left", suffixes=("", "_weather"))
    else:
        print("   [WARNING] Data cuaca tidak ditemukan.")
        df_master = df_filtered

    drop_cols = ["year", "day_of_year", "island", "is_valid_burned_area"]
    df_master = df_master.drop(columns=[c for c in drop_cols if c in df_master.columns])

    print("\n6. Menyimpan hasil akhir ke firms_master_features.csv...")
    output_path = PROCESSED_DIR / "firms_master_features.csv"
    df_master.to_csv(output_path, index=False)
    print(f"   Sukses! File tersimpan di: {output_path}")
    print(f"   Total baris data siap pakai: {len(df_master)}")

if __name__ == "__main__":
    main()
