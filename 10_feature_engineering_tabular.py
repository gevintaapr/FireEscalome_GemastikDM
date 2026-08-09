import os
import math
import pandas as pd
import numpy as np
import ee
import geopandas as gpd
from scipy.spatial import cKDTree
from datetime import timedelta
from config import RAW_DIR, PROCESSED_DIR

def chunked_iterable(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def fetch_gee_features(df):
    print("   [GEE] Menginisialisasi Earth Engine...")
    try:
        ee.Initialize(project='karhutlagemastik')
    except Exception as e:
        print("   [ERROR] Gagal inisialisasi GEE.")
        return df

    features = []
    for idx, row in df.iterrows():
        geom = ee.Geometry.Point([row['longitude'], row['latitude']])
        feat = ee.Feature(geom, {
            'cluster_id': int(row['cluster_id']),
            'year': int(row['year']),
            'start_day_of_year': int(row['start_day_of_year'])
        })
        features.append(feat)
    
    batch_size = 500
    all_results = []
    
    for i, batch in enumerate(chunked_iterable(features, batch_size)):
        print(f"   [GEE] Memproses batch {i+1} (ukuran {len(batch)})...")
        fc = ee.FeatureCollection(batch)
        
        def process_feature(feat):
            year = ee.Number(feat.get('year'))
            doy = ee.Number(feat.get('start_day_of_year'))
            start_date = ee.Date.fromYMD(year, 1, 1).advance(doy.subtract(1), 'day')
            
            srtm = ee.Image('USGS/SRTMGL1_003')
            elevation = srtm.select('elevation')
            slope = ee.Terrain.slope(elevation).rename('slope')
            aspect = ee.Terrain.aspect(elevation).rename('aspect')
            
            modis = ee.ImageCollection("MODIS/061/MOD13Q1")
            img_current = modis.filterDate(start_date.advance(-16, 'day'), start_date.advance(16, 'day')).first()
            ndvi_current = ee.Algorithms.If(img_current, ee.Image(img_current).select('NDVI').multiply(0.0001), ee.Image.constant(0).rename('NDVI'))
            
            img_prev = modis.filterDate(start_date.advance(-32, 'day'), start_date.advance(0, 'day')).first()
            ndvi_prev = ee.Algorithms.If(img_prev, ee.Image(img_prev).select('NDVI').multiply(0.0001), ee.Image.constant(0).rename('NDVI'))
            
            combined = ee.Image([elevation, slope, aspect, ndvi_current, ndvi_prev]).rename(['elevation', 'slope', 'aspect', 'ndvi_current', 'ndvi_prev'])
            values = combined.reduceRegion(reducer=ee.Reducer.first(), geometry=feat.geometry(), scale=500)
            return feat.setMulti(values)
        
        processed_fc = fc.map(process_feature)
        try:
            data = processed_fc.getInfo()['features']
            for item in data:
                props = item['properties']
                all_results.append({
                    'cluster_id': props.get('cluster_id'),
                    'elevation': props.get('elevation', 0),
                    'slope': props.get('slope', 0),
                    'aspect': props.get('aspect', 0),
                    'ndvi_current': props.get('ndvi_current', 0),
                    'ndvi_prev': props.get('ndvi_prev', 0)
                })
        except Exception as e:
            print(f"   [ERROR GEE] Batch gagal: {e}")

    df_gee = pd.DataFrame(all_results)
    if not df_gee.empty:
        df_gee['ndvi_delta_16d'] = df_gee['ndvi_current'] - df_gee['ndvi_prev']
        df_gee = df_gee.drop(columns=['ndvi_prev'])
        return pd.merge(df, df_gee, on='cluster_id', how='left')
    return df

def process_anthropogenic(df):
    print("   [FITUR] Menghitung Kepadatan Penduduk (WorldPop)...")
    worldpop_dir = RAW_DIR / "Gemastik_Karhutla_WorldPop"
    wp_files = {
        "Kalimantan": worldpop_dir / "WorldPop_Density_kalimantan_500m.csv",
        "Sumatra": worldpop_dir / "WorldPop_Density_sumatra_500m.csv"
    }
    
    df['population_density'] = 0.0
    for island, path in wp_files.items():
        # Case insensitive file check
        actual_path = None
        if os.path.exists(worldpop_dir):
            for f in os.listdir(worldpop_dir):
                if island.lower() in f.lower() and f.endswith('.csv'):
                    actual_path = worldpop_dir / f
                    break
        
        if actual_path:
            df_wp = pd.read_csv(actual_path)
            if 'latitude' in df_wp.columns and 'longitude' in df_wp.columns:
                tree = cKDTree(df_wp[['latitude', 'longitude']].values)
                mask = df['island'] == island
                if mask.sum() > 0:
                    coords = df.loc[mask, ['latitude', 'longitude']].values
                    dist, idx = tree.query(coords, k=1)
                    candidates = [c for c in df_wp.columns if c not in ["latitude", "longitude", "system:index", ".geo"]]
                    pop_col = candidates[0] if candidates else 'b1'
                    df.loc[mask, 'population_density'] = df_wp.iloc[idx][pop_col].values
        else:
            print(f"      [WARNING] WorldPop {island} tidak ditemukan.")

    print("   [FITUR] Menghitung Jarak Jalan (OSM GPKG)... (Mungkin lambat)")
    df['road_distance'] = 99999.0
    
    # Hanya lakukan jika library pyogrio tersedia, geopandas butuh ini untuk gpkg besar
    try:
        import pyogrio
        has_pyogrio = True
    except:
        has_pyogrio = False
        print("      [WARNING] Modul pyogrio tidak ada. Pembacaan jalan OSM diskip.")
        
    if has_pyogrio:
        road_files = {
            "Kalimantan": RAW_DIR / "kalimantan-260806-free.gpkg" / "kalimantan.gpkg",
            "Sumatra": RAW_DIR / "sumatra-260806-free.gpkg" / "sumatra.gpkg"
        }
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        for island, path in road_files.items():
            mask = df['island'] == island
            if mask.sum() == 0 or not os.path.exists(path): continue
            
            print(f"      Membaca dan mencari di peta jalan {island}...")
            try:
                gdf_roads = gpd.read_file(path, layer='gis_osm_roads_free', engine='pyogrio', columns=['geometry'])
                gdf_roads_m = gdf_roads.to_crs("EPSG:3857")
                gdf_pts_m = gdf[mask].to_crs("EPSG:3857")
                
                joined = gpd.sjoin_nearest(gdf_pts_m, gdf_roads_m, how='left', distance_col='dist_m')
                joined = joined[~joined.index.duplicated(keep='first')]
                df.loc[mask, 'road_distance'] = joined['dist_m']
            except Exception as e:
                print(f"      [ERROR] Jalan OSM gagal: {e}")
                
    return df

def get_azimuth(lat1, lon1, lat2, lon2):
    # Menghitung sudut arah dari titik 1 ke titik 2
    dlon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360

def calculate_wind_alignment_and_weather(df):
    print("   [FITUR] Menghitung Cuaca Jangka Panjang (Time-Series)...")
    weather_file = RAW_DIR / "weather_all.csv"
    if not os.path.exists(weather_file):
        return df

    df_weather = pd.read_csv(weather_file)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    # KDTree untuk stasiun cuaca terdekat
    weather_stations = df_weather[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    tree_weather = cKDTree(weather_stations[['latitude', 'longitude']].values)

    df['precipitation_dry_streak'] = 0
    df['cumulative_precip_14d'] = 0.0
    df['cumulative_precip_30d'] = 0.0
    df['wind_alignment_score'] = 0.0
    df['windspeed_10m_max'] = 0.0
    df['temperature_2m_max'] = 0.0
    
    # Load raw firms data for wind alignment
    raw_firms_file = PROCESSED_DIR / "firms_master_features.csv"
    if os.path.exists(raw_firms_file):
        df_firms = pd.read_csv(raw_firms_file)
        df_firms['acq_date'] = pd.to_datetime(df_firms['acq_date']).dt.date
    else:
        df_firms = pd.DataFrame()

    for idx, row in df.iterrows():
        # 1. Cari cuaca
        dist, widx = tree_weather.query([row['latitude'], row['longitude']])
        closest_station = weather_stations.iloc[widx]
        w_lat, w_lon = closest_station['latitude'], closest_station['longitude']
        
        # Ambil seluruh seri waktu cuaca di stasiun ini
        station_weather = df_weather[(df_weather['latitude'] == w_lat) & (df_weather['longitude'] == w_lon)].sort_values('time')
        
        year = int(row['year'])
        doy = int(row['start_day_of_year'])
        start_date = pd.to_datetime(f"{year} {doy}", format="%Y %j")
        
        # H-30 hingga H-1
        past_30d = station_weather[(station_weather['time'] >= start_date - timedelta(days=30)) & (station_weather['time'] < start_date)]
        past_14d = station_weather[(station_weather['time'] >= start_date - timedelta(days=14)) & (station_weather['time'] < start_date)]
        
        if not past_14d.empty:
            df.loc[idx, 'cumulative_precip_14d'] = past_14d['precipitation_sum'].sum()
        if not past_30d.empty:
            df.loc[idx, 'cumulative_precip_30d'] = past_30d['precipitation_sum'].sum()
            
            # Hitung dry streak (hari berturut-turut tanpa hujan dari start_date ke belakang)
            streak = 0
            for precip in reversed(past_30d['precipitation_sum'].values):
                if precip < 0.1: streak += 1
                else: break
            df.loc[idx, 'precipitation_dry_streak'] = streak

        # Cuaca pada hari H0
        day0_weather = station_weather[station_weather['time'] == start_date]
        wind_dir = 0
        if not day0_weather.empty:
            df.loc[idx, 'windspeed_10m_max'] = day0_weather['windspeed_10m_max'].values[0]
            df.loc[idx, 'temperature_2m_max'] = day0_weather['temperature_2m_max'].values[0]
            wind_dir = day0_weather['winddirection_10m_dominant'].values[0]
            
            # --- TAMBAHAN FITUR INTERAKSI BINTANG LIMA ---
            # 1. Wind-Slope Interaction (Updraft Cerobong Asap)
            # Aspect (arah lereng turun) vs Arah Angin Datang. Cos = 1 berarti angin meniup naik lereng.
            angle_diff_slope = math.radians(wind_dir - row['aspect'])
            df.loc[idx, 'wind_slope_interaction'] = math.cos(angle_diff_slope) * row['slope'] * df.loc[idx, 'windspeed_10m_max']
            
        # 2. Hitung Wind Alignment Score
        if not df_firms.empty:
            # Tetangga: titik api pada H0, kecuali dirinya sendiri
            neighbors = df_firms[df_firms['acq_date'] == start_date.date()]
            
            # Cari jarak spasial ke semua tetangga H0
            if len(neighbors) > 1:
                coords = neighbors[['latitude', 'longitude']].values
                target = np.array([row['latitude'], row['longitude']])
                # Cari tetangga terdekat (selain dirinya sendiri jika ia ada di sana)
                dists = np.linalg.norm(coords - target, axis=1)
                valid_dists = np.where(dists > 0.001)[0] # bukan dirinya sendiri
                
                if len(valid_dists) > 0:
                    nearest_idx = valid_dists[np.argmin(dists[valid_dists])]
                    n_lat = coords[nearest_idx][0]
                    n_lon = coords[nearest_idx][1]
                    
                    neighbor_dir = get_azimuth(row['latitude'], row['longitude'], n_lat, n_lon)
                    # cos(theta) dari sudut selisih
                    angle_diff = math.radians(wind_dir - neighbor_dir)
                    df.loc[idx, 'wind_alignment_score'] = math.cos(angle_diff)

    # 2. Peatland Drought Index
    if 'is_peatland' in df.columns:
        df['peatland_drought_index'] = df['is_peatland'] * df['precipitation_dry_streak']
        
    # 3. Fuel Danger Index (Keringnya daun dikali Suhu terik)
    if 'ndvi_delta_16d' in df.columns and 'temperature_2m_max' in df.columns:
        # Defisit air daun absolut dikali suhu tinggi = bahan bakar ekstrem
        df['fuel_danger_index'] = df['ndvi_delta_16d'].abs() * df['temperature_2m_max']

    return df

def main():
    print("============================================================")
    print("10_FEATURE_ENGINEERING_TABULAR.PY")
    print("Tahap 3: Master Table & Fitur Lanjutan (Dengan Checkpoint)")
    print("============================================================\n")

    input_file = PROCESSED_DIR / "cluster_master_features.csv"
    if not os.path.exists(input_file):
        print(f"[ERROR] File input tidak ditemukan: {input_file}")
        return

    cp_gee = PROCESSED_DIR / "checkpoint_2_gee_features.csv"
    cp_antro = PROCESSED_DIR / "checkpoint_3_antro.csv"
    cp_final = PROCESSED_DIR / "tabular_master_final.csv"

    # --- STEP A & B: GEE (Topografi & NDVI) ---
    print("\n[STEP A & B] Integrasi SRTM & NDVI via Google Earth Engine")
    if os.path.exists(cp_gee):
        print(f"   -> Ditemukan checkpoint GEE: {cp_gee.name}")
        df_master = pd.read_csv(cp_gee)
    else:
        df = pd.read_csv(input_file)
        df_master = fetch_gee_features(df)
        df_master.to_csv(cp_gee, index=False)

    # --- STEP C: Antropogenik (Roads, Population) ---
    print("\n[STEP C] Integrasi Antropogenik (OpenStreetMap & WorldPop)")
    if os.path.exists(cp_antro):
        print(f"   -> Ditemukan checkpoint Antropogenik: {cp_antro.name}")
        df_master = pd.read_csv(cp_antro)
    else:
        df_master = process_anthropogenic(df_master)
        df_master.to_csv(cp_antro, index=False)

    # --- STEP D: Weather & Fitur Interaksi ---
    print("\n[STEP D] Integrasi Cuaca Lanjutan & Fitur Interaksi")
    df_master = calculate_wind_alignment_and_weather(df_master)
    
    if 'year' in df_master.columns: df_master = df_master.drop(columns=['year'])
    if 'start_day_of_year' in df_master.columns: df_master = df_master.drop(columns=['start_day_of_year'])

    print(f"\n[FINAL] Menyimpan Master Dataset Final ke: {cp_final.name}")
    df_master.to_csv(cp_final, index=False)
    print("SUKSES! Master Dataset Siap!")

if __name__ == "__main__":
    main()
