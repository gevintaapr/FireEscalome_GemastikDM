"""
07_fetch_worldpop.py
Mengambil data Kepadatan Penduduk (Population Density) dari WorldPop.
Digunakan sebagai indikator kuat untuk 'Human-induced Escalation' (kebakaran 
yang dipicu oleh aktivitas/kedekatan dengan manusia).

Meskipun di dokumen disarankan 'direct download' dari worldpop.org, 
mengambilnya melalui Google Earth Engine (GEE) jauh lebih praktis karena 
kita bisa langsung menyelaraskan resolusinya menjadi 500m (sama seperti data 
sebelumnya) dan mengekstraknya otomatis ke bentuk CSV (Titik Koordinat).
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

    folder_drive = "Gemastik_Karhutla_WorldPop"
    
    # Skala disesuaikan 500m agar konsisten dengan Burned Area, Land Cover, dan NDVI
    export_scale = 500  

    # Dataset WorldPop Global Project Population Data
    # Filter khusus untuk negara Indonesia ('IDN')
    dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                .filter(ee.Filter.equals('country', 'IDN')) \
                .select('population')

    # Karena dataset ini berupa ImageCollection (biasanya 1 gambar per tahun/negara), 
    # kita konversi menjadi Image tunggal (mengambil nilai estimasi populasinya)
    pop_image = dataset.max().rename('population')

    # Tambahkan band latitude dan longitude bawaan GEE
    latlon = ee.Image.pixelLonLat()
    combined = pop_image.addBands(latlon)

    for region_name, geometry in regions.items():
        task_name = f'Export_WorldPop_{region_name}'
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
            fileNamePrefix=f'WorldPop_Density_{region_name}_{export_scale}m',
            fileFormat='CSV'
        )

        # Memulai eksekusi task
        task.start()
        print(f"Task {task_name} dimulai.")

    print("\n=======================================================")
    print("Semua task WorldPop telah berhasil dikirim ke server Earth Engine.")
    print("Proses ekspor data CSV ke Google Drive sedang berjalan di background.")
    print("Kamu bisa memantau progresnya di: https://code.earthengine.google.com/tasks")
    print("=======================================================")

if __name__ == "__main__":
    main()
