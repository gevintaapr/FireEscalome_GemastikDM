"""
02_fetch_burned_area.py
Mengambil data area terbakar (Burned Area) menggunakan produk satelit MODIS MCD64A1 
dari Google Earth Engine (GEE). Data ini merupakan "Tier 1 - Prioritas Tertinggi" 
sebagai ground truth untuk validasi eskalasi karhutla.

Tips Penting:
1. Atribut BurnDate: 0 berarti area tidak terbakar. Angka 1-366 menandakan tanggal persis.
2. Skala Resolusi: 500 m (sesuai resolusi MODIS MCD64A1).
3. Penyimpanan: Ekspor tahunan (2020-2026) langsung ke Google Drive agar tidak membebani RAM lokal.
"""

import ee

def main():
    print("Mencoba inisialisasi Earth Engine...")
    try:
        ee.Initialize(project='karhutlagemastik')
        print("Inisialisasi Earth Engine berhasil!")
    except Exception as e:
        print("Gagal inisialisasi Earth Engine.")
        print("Silakan lakukan autentikasi terlebih dahulu dengan menjalankan perintah berikut di terminal:")
        print("earthengine authenticate")
        print(f"Detail error: {e}")
        return

    # Bounding box berdasarkan wilayah prioritas (Sumatra & Kalimantan)
    regions = {
        'Sumatra': ee.Geometry.BBox(95, -6, 109, 3),
        'Kalimantan': ee.Geometry.BBox(108, -4, 118, 4)
    }

    start_year = 2020
    end_year = 2026
    folder_drive = "Gemastik_Karhutla_BurnedArea"

    for year in range(start_year, end_year + 1):
        for region_name, geometry in regions.items():
            task_name = f'Export_MCD64A1_BurnDate_{region_name}_{year}'
            
            # Load MODIS MCD64A1 Collection (Burned Area Monthly)
            dataset = ee.ImageCollection('MODIS/061/MCD64A1') \
                        .filter(ee.Filter.date(f'{year}-01-01', f'{year}-12-31')) \
                        .select('BurnDate')

            # Ambil nilai max untuk mendapatkan hari terbakar dalam tahun berjalan
            # Beri nama band 'BurnDate' agar jelas di CSV nanti
            burned_area = dataset.max().clip(geometry).rename('BurnDate')

            # 1. Masking: hanya simpan piksel yang terbakar (BurnDate > 0)
            burned_area_masked = burned_area.updateMask(burned_area.gt(0))

            # 2. Tambahkan band latitude dan longitude bawaan GEE
            latlon = ee.Image.pixelLonLat()
            combined = burned_area_masked.addBands(latlon)

            print(f"Menyiapkan task ekspor: {task_name}")

            # 3. Konversi piksel image menjadi FeatureCollection (titik/vektor)
            points = combined.sample(
                region=geometry,
                scale=500,
                geometries=False  # Band 'longitude' dan 'latitude' otomatis menjadi kolom CSV
            )

            # 4. Ekspor FeatureCollection sebagai CSV ke Google Drive
            task = ee.batch.Export.table.toDrive(
                collection=points,
                description=task_name,
                folder=folder_drive,
                fileNamePrefix=f'BurnedArea_{region_name}_{year}',
                fileFormat='CSV'
            )

            # Memulai eksekusi task di server GEE
            task.start()
            print(f"Task {task_name} dimulai.")

    print("\n=======================================================")
    print("Semua task telah berhasil dikirim ke server Earth Engine.")
    print("Proses ekspor data ke Google Drive sedang berjalan.")
    print("Kamu bisa memantau progresnya di: https://code.earthengine.google.com/tasks")
    print("=======================================================")

if __name__ == "__main__":
    main()
