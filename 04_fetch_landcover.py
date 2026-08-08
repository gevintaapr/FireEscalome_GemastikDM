"""
04_fetch_landcover.py
Mengambil data Land Cover (Tutupan Lahan) untuk memfilter false-positive 
(seperti kilang minyak, area urban/industri) dan mendeteksi Escalation Archetypes.

Menggunakan GOOGLE/DYNAMICWORLD/V1 (Near Real-Time hingga 2026).
Logika Pipeline (Pendekatan Eliminasi Aman):
- WAJIB DISIMPAN (Kandidat Karhutla): 1 (Trees), 2 (Grass), 3 (Flooded Veg - Peat driven), 
  4 (Crops - Human induced), 5 (Shrub & Scrub).
- WAJIB DIBUANG (Anti False-Positive): 0 (Water), 6 (Built/Urban), 7 (Bare), 8 (Snow/Ice).
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

    # Bounding box berdasarkan wilayah prioritas (Sumatra & Kalimantan)
    regions = {
        'Sumatra': ee.Geometry.BBox(95, -6, 109, 3),
        'Kalimantan': ee.Geometry.BBox(108, -4, 118, 4)
    }

    start_year = 2020
    end_year = 2026
    folder_drive = "Gemastik_Karhutla_LandCover"
    
    # Samakan dengan Burned Area & DEM (500m)
    export_scale = 500 

    for year in range(start_year, end_year + 1):
        for region_name, geometry in regions.items():
            task_name = f'Export_DynamicWorld_{region_name}_{year}'
            
            # Dynamic World adalah ImageCollection yang terus di-update.
            # Kita ambil nilai modus (nilai kemunculan terbanyak / mode) dalam 1 tahun
            # agar kita mendapatkan label tutupan lahan utama di lokasi tersebut.
            dataset = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
                        .filter(ee.Filter.date(f'{year}-01-01', f'{year}-12-31')) \
                        .select('label')
            
            # Mode() digunakan untuk mengagregasi data harian menjadi representasi tahunan
            landcover = dataset.mode().clip(geometry).rename('LandCover_Class')
            
            # Tambahkan band latitude dan longitude
            latlon = ee.Image.pixelLonLat()
            combined = landcover.addBands(latlon)

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
                fileNamePrefix=f'LandCover_DynamicWorld_{region_name}_{year}_{export_scale}m',
                fileFormat='CSV'
            )

            # Memulai eksekusi task
            task.start()
            print(f"Task {task_name} dimulai.")

    print("\n=======================================================")
    print("Semua task Land Cover (Dynamic World) telah berhasil dikirim ke server.")
    print("Proses ekspor data CSV ke Google Drive sedang berjalan di background.")
    print("Kamu bisa memantau progresnya di: https://code.earthengine.google.com/tasks")
    print("=======================================================")

if __name__ == "__main__":
    main()
