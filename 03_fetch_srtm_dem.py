"""
03_fetch_srtm_dem.py
Mengambil data Elevasi dan Slope dari SRTM DEM (USGS/SRTMGL1_003) 
menggunakan Google Earth Engine. Data ini sifatnya statis (tidak berubah tiap hari).
Ini adalah "Tier 1" prioritas tinggi dalam penelitian karhutla, di mana kemiringan lereng
sangat mempengaruhi kecepatan rambat api.
"""

import ee

def main():
    print("Mencoba inisialisasi Earth Engine...")
    try:
        # Menggunakan project ID yang sama seperti skrip sebelumnya
        ee.Initialize(project='karhutlagemastik')
        print("Inisialisasi Earth Engine berhasil!")
    except Exception as e:
        print("Gagal inisialisasi Earth Engine.")
        print("Pastikan Anda sudah mengautentikasi akun dengan perintah 'earthengine authenticate'.")
        print(f"Detail error: {e}")
        return

    # Bounding box berdasarkan wilayah prioritas (Sumatra & Kalimantan)
    regions = {
        'Sumatra': ee.Geometry.BBox(95, -6, 109, 3),
        'Kalimantan': ee.Geometry.BBox(108, -4, 118, 4)
    }

    folder_drive = "Gemastik_Karhutla_SRTM"
    
    # Resolusi spasial untuk diekspor (dalam meter).
    # Disetel 500m agar sepadan/selaras dengan resolusi Burned Area MCD64A1.
    # Jika file CSV dirasa terlalu besar, bisa diperbesar menjadi 1000m (1km) atau 11000m (11km).
    export_scale = 500

    # 1. Panggil dataset SRTM DEM di GEE
    srtm = ee.Image('USGS/SRTMGL1_003')

    # 2. Ambil band elevasi (ketinggian dalam meter)
    elevation = srtm.select('elevation')

    # 3. Hitung otomatis slope (kemiringan lereng dalam derajat)
    slope = ee.Terrain.slope(elevation).rename('slope')

    # Gabungkan elevation dan slope menjadi satu image
    srtm_combined = elevation.addBands(slope)

    # Tambahkan band latitude dan longitude bawaan GEE
    latlon = ee.Image.pixelLonLat()
    srtm_final = srtm_combined.addBands(latlon)

    for region_name, geometry in regions.items():
        task_name = f'Export_SRTM_DEM_{region_name}'
        print(f"Menyiapkan task ekspor: {task_name} dengan scale {export_scale}m")

        # Konversi image menjadi FeatureCollection (titik/vektor) 
        # Hanya sampling pada region prioritas
        points = srtm_final.sample(
            region=geometry,
            scale=export_scale,
            geometries=False
        )

        # Ekspor FeatureCollection sebagai CSV ke Google Drive
        task = ee.batch.Export.table.toDrive(
            collection=points,
            description=task_name,
            folder=folder_drive,
            fileNamePrefix=f'SRTM_DEM_Slope_{region_name}_{export_scale}m',
            fileFormat='CSV'
        )

        # Memulai eksekusi task di server GEE
        task.start()
        print(f"Task {task_name} dimulai.")

    print("\n=======================================================")
    print("Semua task SRTM DEM telah berhasil dikirim ke server Earth Engine.")
    print("Proses ekspor data CSV ke Google Drive sedang berjalan di background.")
    print("Kamu bisa memantau progresnya di: https://code.earthengine.google.com/tasks")
    print("=======================================================")

if __name__ == "__main__":
    main()
