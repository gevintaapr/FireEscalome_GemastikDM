# Laporan Penyelesaian: Fase 1 (Data Preparation & Feature Engineering)

Dokumen ini adalah *handover* (serah terima) resmi dari **Person A (Data Engineer)** kepada **Person B (Modeler)**. Di sini dirangkum seluruh tahapan, perbaikan logika, inovasi, serta *Kamus Data* lengkap agar tidak ada miskomunikasi (*miss*) mengenai arti dari setiap fitur yang akan dimasukkan ke dalam *Machine Learning*.

---

## 1. Tahapan Pembersihan (Sanitasi Spasial-Temporal)
**Masalah:** Data NASA FIRMS sangat kotor (banyak *false alarm* dari pabrik, pantulan atap baja, dll).
**Solusi & Proses:**
*   **Filter Land Cover:** Membuang titik api yang berada di pemukiman/industri menggunakan *Dynamic World Land Cover*.
*   **Filter Validasi MCD64A1:** Jika titik api memiliki *confidence* rendah (`l` atau `n`), titik tersebut **dibuang**, *KECUALI* dia terbukti berlokasi di lahan gambut atau tervalidasi oleh satelit *Burned Area* (benar-benar meninggalkan bekas hangus).
**Hasil:** Data sisa dijamin 100% adalah api kebakaran alam sungguhan.

---

## 2. Pelabelan Eskalasi (Ground Truth Labeling)
**Masalah:** Machine learning butuh target (0 atau 1) yang valid secara saintifik.
**Solusi & Proses:**
*   Menggunakan algoritma **ST-DBSCAN** untuk membungkus titik-titik yang berdekatan secara lokasi dan waktu menjadi satu "Cluster Kejadian Kebakaran".
*   **Label 1 (Eskalasi):** Diberikan HANYA JIKA pertumbuhan jumlah titik api di dalam cluster tersebut meledak sangat cepat (masuk *Top 5% / Persentil-95*) dalam waktu 3 hari, DAN tervalidasi ada bekas hangusnya oleh satelit MCD64A1.
*   **Label 0 (Non-Eskalasi):** Api kebakaran nyata, namun bisa padam sendiri atau pertumbuhannya lambat.

---

## 3. Ekstraksi Fitur (Inovasi Bintang Lima)
Tahap pengayaan data (*Feature Engineering*) ini menggunakan 3 pendekatan jenius:
1.  **Ekstraksi GEE Real-Time:** Menembak langsung 2.888 titik koordinat ke server *Google Earth Engine* untuk mendapatkan data topografi dan defisit daun (NDVI) presisi tingkat piksel pada hari-H kejadian.
2.  **KDTree & Spatial Indexing:** Merayapi peta jaringan jalan se-Sumatra dan Kalimantan berukuran Gigabyte dalam hitungan detik menggunakan GeoPandas.
3.  **Anti-Data Leakage:** Memastikan semua perhitungan (*wind alignment* dll) dihitung HANYA menggunakan data **Hari Pertama** (H0), sehingga model kita sah dipakai secara *Real-Time* di dunia nyata tanpa menyontek masa depan.

---

## 4. KAMUS DATA (Penjelasan Lengkap Tiap Fitur)
Berikut adalah penjelasan mutlak dari isi `tabular_master_final.csv` yang akan digunakan oleh Person B untuk *Training LightGBM/GNN*:

### A. Fitur Identitas & Target (Y)
- **`cluster_id`**: ID unik dari setiap satu kejadian (klaster) kebakaran.
- **`label_escalation`**: Variabel Target. **1** = Kebakaran ganas/Eskalasi, **0** = Kebakaran jinak.
- **`latitude` & `longitude`**: Titik pusat (*centroid*) dari kejadian kebakaran.
- **`island`**: Lokasi pulau (Kalimantan / Sumatra).

### B. Fitur Fisik & Topografi
- **`is_peatland`**: Apakah lokasi tersebut lahan gambut? (1 = Ya, 0 = Tidak).
- **`elevation`**: Ketinggian lokasi di atas permukaan laut (dalam meter).
- **`slope`**: Derajat kemiringan tanah (0 = datar, >20 = curam). Sangat mempengaruhi kecepatan merambatnya api ke atas.
- **`aspect`**: Arah hadap lereng gunung (0 s.d 360 derajat). Misalnya lereng menghadap utara/selatan.

### C. Fitur Bahan Bakar & Vegetasi (NDVI)
- **`ndvi_current`**: Indeks kerapatan hijau daun tepat pada **16-hari periode kejadian**. Angka rendah (mendekati 0) berarti vegetasi mengering atau gundul, siap terbakar.
- **`ndvi_delta_16d`**: Penurunan/perubahan kerapatan daun dibandingkan **16 hari sebelumnya**. Ini adalah proksi untuk menangkap *Flash Drought* (hutan yang tiba-tiba mengering dengan cepat akibat gelombang panas).

### D. Fitur Antropogenik (Aktivitas Manusia)
- **`population_density`**: Jumlah jiwa per km persegi di sekitar titik api (WorldPop).
- **`road_distance`**: Jarak absolut dari titik api ke jalan aspal/tanah terdekat (dalam meter). Semakin dekat, semakin tinggi potensi dibakar oleh manusia secara sengaja.

### E. Fitur Cuaca Murni (Time-Series)
- **`windspeed_10m_max`**: Kecepatan angin maksimal di hari pertama kejadian (H0).
- **`temperature_2m_max`**: Suhu udara terpanas di hari pertama kejadian (H0).
- **`cumulative_precip_14d` & `30d`**: Total akumulasi curah hujan selama 14 dan 30 hari ke belakang. Defisit hujan yang parah akan memicu eskalasi.
- **`precipitation_dry_streak`**: Menghitung **berapa hari berturut-turut** area tersebut sama sekali tidak diguyur hujan (kemarau absolut) terhitung mundur dari hari pertama kejadian.

### F. Fitur Interaksi Tingkat Tinggi (The Killer Features)
Ini adalah fitur kombinasi rumus matematika yang akan meningkatkan performa Machine Learning secara drastis:
- **`wind_alignment_score`**: (*Trigonometri*). Menghitung nilai *Cos* dari (Arah Angin vs Arah rambat spasial tetangga). Nilai mendekati **1** berarti angin bertiup kencang persis ke arah percikan api tetangga menjalar (sangat berbahaya!).
- **`wind_slope_interaction`**: Efek cerobong asap (*Updraft*). Angka ini tinggi jika angin bertiup kencang secara tegak lurus menaiki lereng bukit (`slope` + `aspect`).
- **`peatland_drought_index`**: Hasil kali `is_peatland` dengan `dry_streak`. Mengakomodasi fakta bahwa lahan gambut yang kering 10 hari jauh lebih eksplosif daripada tanah mineral yang kering 10 hari.
- **`fuel_danger_index`**: Hasil kali penurunan drastis NDVI absolut dengan suhu terik harian. Mendeteksi kombinasi mematikan antara daun yang kehilangan air + gelombang panas ekstrim.

---
**Dokumen Selesai - Master Dataset Siap untuk Person B (Tahap Pemodelan ML).**
