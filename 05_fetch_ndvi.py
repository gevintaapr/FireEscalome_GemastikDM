"""
05_fetch_ndvi.py
Mengambil data NDVI (kesehatan/kerapatan vegetasi) dari satelit MODIS (MOD13Q1)
menggunakan Google Earth Engine (GEE).

Fitur:
- Mengambil nilai NDVI minimum tahunan (sebagai proksi kekeringan ekstrem/fuel drying).
- Scaling factor: 0.0001 (mengonversi raw value menjadi range -0.2 s.d 1.0).
- Skala: 500m (disesuaikan dengan Burned Area & Land Cover agar konsisten saat di-join).
- Output: Diekspor sebagai CSV ke Google Drive (2020-2026).
"""

import ee

def main():
    print("Mencoba inisialisasi Earth Engine...")
    try:
        ee.Initialize(project='karhutlagemastik')
        print("Inisialisasi Earth Engine berhasil!")
    except Exception as e:
        print("Gagal inisialisasi Earth Engine.")
        print("Pastikan Anda sudah mengautentikasi akun dengan perintah 'earthengine authenticate'.")
        print(f"Detail error: {e}")
        return

    regions = {
        'Sumatra': ee.Geometry.BBox(95, -6, 109, 3),
        'Kalimantan': ee.Geometry.BBox(108, -4, 118, 4)
    }

    start_year = 2020
    end_year = 2026
    folder_drive = "Gemastik_Karhutla_NDVI"
    
    # Skala asli MOD13Q1 adalah 250m. Kita set ke 500m agar seragam dengan 
    # file CSV lainnya (MCD64A1, SRTM, LandCover) dan lebih ringan saat di-join di Pandas.
    export_scale = 500  

    for year in range(start_year, end_year + 1):
        for region_name, geometry in regions.items():
            task_name = f'Export_MODIS_NDVI_{region_name}_{year}'
            
            # Panggil Collection MODIS NDVI (MOD13Q1)
            ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
                                .filterDate(f'{year}-01-01', f'{year}-12-31') \
                                .select("NDVI")

            # Ambil nilai minimum NDVI tahunan (titik kekeringan terparah / fuel drying)
            # Kalibrasi dengan Scaling Factor 0.0001 agar mendapat nilai -0.2 s/d 1.0
            min_ndvi = ndvi_collection.min().multiply(0.0001).rename('NDVI_min')
            
            # Tambahkan band latitude dan longitude bawaan GEE
            latlon = ee.Image.pixelLonLat()
            combined = min_ndvi.addBands(latlon)

            print(f"Menyiapkan task ekspor: {task_name}")

            # Konversi piksel image menjadi titik (CSV)
            points = combined.sample(
                region=geometry,
                scale=export_scale,
                geometries=False
            )

            # Ekspor FeatureCollection sebagai CSV ke Google Drive
            task = ee.batch.Export.table.toDrive(
                collection=points,
                description=task_name,
                folder=folder_drive,
                fileNamePrefix=f'NDVI_Min_{region_name}_{year}_{export_scale}m',
                fileFormat='CSV'
            )

            # Memulai eksekusi task
            task.start()
            print(f"Task {task_name} dimulai.")

    print("\n=======================================================")
    print("Semua task NDVI telah berhasil dikirim ke server Earth Engine.")
    print("Proses ekspor data CSV ke Google Drive sedang berjalan di background.")
    print("Kamu bisa memantau progresnya di: https://code.earthengine.google.com/tasks")
    print("=======================================================")

if __name__ == "__main__":
    main()
