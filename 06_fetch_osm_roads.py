"""
06_fetch_osm_roads.py
Mengambil data jaringan jalan (highway) dari OpenStreetMap (OSM) 
untuk digunakan sebagai indikator Human-induced Escalation 
(misal: jarak titik api ke jalan terdekat).

Wilayah yang diambil adalah 5 Provinsi Prioritas:
- Riau
- Sumatera Selatan
- Jambi
- Kalimantan Barat
- Kalimantan Tengah
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import os

def main():
    # Konfigurasi OSMnx
    ox.settings.timeout = 3000  # Waktu tunggu diperpanjang karena data 1 provinsi sangat besar
    ox.settings.use_cache = True # Gunakan cache agar jika putus tidak mengulang dari nol

    # 5 Provinsi Prioritas Karhutla
    provinces = [
        "Riau, Indonesia",
        "Sumatera Selatan, Indonesia",
        "Jambi, Indonesia",
        "Kalimantan Barat, Indonesia",
        "Kalimantan Tengah, Indonesia"
    ]

    # Kita memfilter data spasial agar HANYA mengambil jaringan jalan ('highway')
    tags = {'highway': True}
    
    output_dir = "data_raw/osm_roads"
    os.makedirs(output_dir, exist_ok=True)

    all_roads_gdf = []

    print("==================================================================")
    print("Mulai mengunduh data jalan dari OpenStreetMap (Overpass API)...")
    print("PRO TIP: Proses ini bisa memakan waktu yang cukup lama dan memori.")
    print("==================================================================\n")

    for place in provinces:
        print(f"[+] Mendownload jaringan jalan untuk: {place}...")
        try:
            # Menggunakan features_from_place untuk langsung mengunduh vektor jalan
            roads = ox.features_from_place(place, tags=tags)
            
            # Filter hanya bentuk garis (LineString) untuk menghindari poligon/titik nyasar
            roads_lines = roads[roads.geometry.type == 'LineString'].copy()
            
            # Kita hanya menyimpan 2 kolom: koordinat geometri dan tipe jalannya
            cols_to_keep = ['geometry', 'highway']
            roads_lines = roads_lines[cols_to_keep].reset_index(drop=True)
            
            # Simpan ke memori untuk nanti digabungkan jadi satu
            all_roads_gdf.append(roads_lines)
            
            # Buat file backup GeoJSON per provinsi agar aman jika error di tengah jalan
            clean_name = place.split(',')[0].replace(" ", "_").lower()
            backup_path = f"{output_dir}/roads_{clean_name}.geojson"
            
            roads_lines.to_file(backup_path, driver='GeoJSON')
            print(f"    Berhasil menyimpan {len(roads_lines)} ruas jalan ke {backup_path}")
            
        except Exception as e:
            print(f"    GAGAL mendownload wilayah {place}. Error: {e}")

    # Menggabungkan semua data per provinsi menjadi 1 file GeoJSON Master
    if all_roads_gdf:
        print("\n[+] Menggabungkan seluruh data jalan 5 provinsi menjadi satu file master...")
        master_roads = gpd.GeoDataFrame(pd.concat(all_roads_gdf, ignore_index=True))
        
        master_path = f"{output_dir}/master_roads_priority.geojson"
        master_roads.to_file(master_path, driver='GeoJSON')
        print(f"SELESAI! Master file jalan tersimpan di: {master_path}")
    else:
        print("\n[-] Yaaah... Tidak ada data jalan satupun yang berhasil diunduh.")

if __name__ == "__main__":
    main()
