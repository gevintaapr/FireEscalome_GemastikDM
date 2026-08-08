# 📄 Proposal & Penjelasan Teknis
**Judul:** Decoding the Fire Escalome: Knowledge Discovery of Hidden Escalation Mechanisms in Indonesian Wildland Fires
**Target:** Dosen Pembimbing / Reviewer Gemastik 2026 Divisi Penambangan Data

---

## 1. Latar Belakang Masalah (Kenapa Ide Ini Penting?)
*(Catatan untuk Dosen: Berisi argumentasi mengapa kita tidak membuat sistem deteksi hotspot biasa)*
- **Masalah Operasional:** Saat puncak kemarau, satelit mendeteksi ribuan titik panas (hotspot) per hari. Sistem yang ada saat ini (seperti SiPongi milik pemerintah) masih bersifat *occurrence-based* (hanya memetakan "ada api di mana"). Mengirimkan regu pemadam ke ribuan titik secara merata adalah hal yang mustahil dan tidak efisien.
- **Kesenjangan (Gap):** Kenyataannya, tidak semua hotspot berbahaya. Mayoritas akan padam sendiri, namun sebagian kecil (<10%) berpotensi meledak menjadi kebakaran masif (*eskalasi*). Selain itu, strategi penanganan karhutla sangat bervariasi; misalnya, pemadaman lahan gambut (*Peat-driven*) membutuhkan pembasahan tanah total yang jauh berbeda dengan kebakaran padang semak karena angin (*Wind-driven*).
- **Solusi Kami:** Alih-alih sekadar memprediksi "apakah ini karhutla atau bukan", kami membangun arsitektur **Knowledge Discovery in Databases (KDD)** untuk membedah DNA kebakaran. Kami menyebutnya **Escalation Archetypes** (Taksonomi Pola Eskalasi). Output sistem kami adalah *Early-Warning Pattern Library* yang merekomendasikan strategi prioritas pemadaman secara spesifik.

---

## 2. Arsitektur Sistem & Pemrosesan Data
*(Catatan untuk Dosen: Kami menggunakan fusi data dari satelit NASA, Google Earth Engine (GEE), dan portal OneMap BIG untuk menghasilkan Dataset Multi-layer).*

### Tabel Sumber Data & Logika Pemrosesannya
| Tier | Nama Data | Sumber & Resolusi Asli | Kegunaan dalam Feature Engineering | Status & Logika Pemrosesan Saat Ini |
|---|---|---|---|---|
| **0** | **Hotspot & Cuaca** | NASA FIRMS (375m) & Open-Meteo | Data mentah awal (label) & indikator cuaca harian (suhu, curah hujan, arah/kecepatan angin). | ✅ **Selesai.** Menjadi **Base Dataset (Jangkar)** tempat semua data lain akan di-join berdasarkan koordinat `lat`/`lon`. |
| **1** | **Burned Area (Ground Truth)** | MODIS MCD64A1 via GEE (500m) | Validasi absolut bahwa titik tersebut benar-benar membesar menjadi area terbakar (bukan sekadar anomali panas sesaat). | ✅ **Selesai.** Diekspor dari GEE via script Python. Kami menggunakan logika `.max()` untuk mendeteksi 'BurnDate' pasti. |
| **1** | **Elevasi & Kemiringan (DEM)** | USGS SRTMGL1_003 via GEE (30m $\rightarrow$ 500m) | Indikator *Topography-driven*. Api menjalar jauh lebih cepat dan membesar jika mendaki lereng yang curam (Slope). | ✅ **Selesai.** Ditarik via GEE. Sengaja kami resample resolusinya ke 500m agar sejajar sempurna dengan data lainnya. |
| **1** | **Land Cover (Tutupan Lahan)** | Google Dynamic World V1 via GEE (10m $\rightarrow$ 500m) | Memfilter *False Positive* (Kilang minyak, Pabrik) & Mendeteksi tipe vegetasi mentah (Semak, Kebun, Hutan). | ✅ **Selesai.** Ditarik dari GEE. Kami menerapkan **Pendekatan Eliminasi Aman**: Menyimpan seluruh kelas vegetasi, dan HANYA membuang area non-vegetasi (Water, Built/Urban, Snow). |
| **1** | **Kesehatan Vegetasi (NDVI)** | MODIS MOD13Q1 via GEE (250m $\rightarrow$ 500m) | Mengukur fase kekeringan bahan bakar (*fuel drying*). Indikator paling vital untuk *Drought-driven*. | ✅ **Selesai.** Ditarik dari GEE. Diambil nilai *minimum* tahunan (`.min()`) untuk merekam momen puncak kekeringan. |
| **1** | **Peta Fungsi Gambut** | Geoportal BIG / Global Forest Watch (GFW) | Indikator absolut pendeteksi *Peat-driven escalation* (kebakaran bawah tanah yang sulit dipadamkan). | 🔄 **Persiapan Join.** Diunduh manual format `.shp` karena resolusi datanya raksasa dan berformat Polygon statis. |
| **2** | **Jaringan Jalan (Roads)** | Geofabrik OpenStreetMap (OSM) | Indikator utama *Human-induced Escalation* (Jarak kedekatan hotspot dengan akses manusia/pembukaan lahan). | 🔄 **Persiapan Join.** Menggunakan `.shp` dari Geofabrik karena *live query* API terbukti tidak mampu menangani besarnya pulau Sumatra. |
| **3** | **Kepadatan Penduduk** | WorldPop 100m via GEE (100m $\rightarrow$ 500m) | Pelengkap indikator intervensi manusia (*Human-induced*). | ✅ **Selesai.** Diekspor dari GEE dan disamakan resolusinya ke 500m. |

> 💡 **CATATAN TEKNIS (Rencana Tahap Integrasi Data):**
> *Bagaimana data-data raksasa di atas saling terhubung di Python?*
> 1. **Untuk Data dari GEE (CSV):** Karena kami sudah dengan cerdas merekayasa skrip GEE untuk memaksa semua output memiliki **skala yang seragam (500 meter)**, proses penggabungan data (Merge/Join) dengan data dasar `firms_all.csv` di Pandas akan sangat ringan, *seamless*, dan menghindari *loss* akibat ketidakcocokan resolusi piksel.
> 2. **Untuk Data Shapefile (Gambut & Jalan):** Kami tidak akan mengonversinya ke *raster*. Kami menggunakan pustaka `GeoPandas` untuk melakukan **Spatial Join** tingkat lanjut. Titik *hotspot* akan dicek menggunakan algoritma *Point-in-Polygon* (untuk Gambut) dan *Nearest Neighbor Distance* (untuk mengukur jarak presisi dalam satuan meter ke jalan OSM terdekat).

---

## 3. Metodologi Data Mining & AI

Kami tidak berhenti pada sekadar meracik model *Deep Learning* kotak hitam (Black Box). Karena kami menyasar instansi seperti BPBD, sistem peringatan harus sangat *Explainable* (bisa dijelaskan):

1. **Model Prediktif Cost-Sensitive (LightGBM):**
   Mengingat kasus ini adalah *Extreme Class Imbalance* (titik yang membesar sangat langka dibanding yang mati sendiri), kami melatih algoritma **LightGBM** dengan teknik `scale_pos_weight`. LightGBM dipilih karena performanya juara untuk data Tabular kompleks dan secara komputasi sangat ringan untuk implementasi operasional *Near Real-Time*.
2. **Algoritma Explainable AI (SHAP Interaction Values):**
   Ini adalah *Secret Weapon* kami. Alih-alih hanya mencari *feature importance* biasa (fitur apa yang penting), kami menggunakan kalkulus SHAP untuk mendeteksi **INTERAKSI** berpasangan antar fitur. (Contoh: Fitur *Peatland* mungkin biasa saja, tapi ketika berinteraksi secara bersamaan dengan "Curah Hujan = 0 selama 14 hari", kontribusinya pada probabilitas eskalasi akan meledak secara eksponensial).
3. **Clustering Archetypes & Association Rules (Apriori):**
   Dari matriks Interaksi SHAP tersebut, kami mengelompokkan hotspot (*Clustering*). Setelah tipe arketip terbentuk (misal: *Peat-driven, Wind-driven*), kami mengekstraksi logika manusia menggunakan algoritma **Apriori** (`mlxtend`) untuk menghasilkan urutan kausalitas baku *IF-THEN* (Rule Base).

---

## 4. Output Akhir & Kesesuaian dengan Society 5.0 (Novelty)
Sistem peringatan triase kami akan menghasilkan **Pattern Library (Katalog Aturan)** yang tidak pernah diimplementasikan oleh sistem konvensional.

Sebagai contoh simulasi operasional, ketika satelit mendeteksi klaster hotspot baru, sistem kami tidak hanya mencetak angka "Ada Titik Api", melainkan:
> *"🚨 Peringatan Triase: Klaster Hotspot A masuk ke kategori **Peat-driven**. (Faktor Pemicu: Berada di Lahan Gambut + Teridentifikasi sebagai Vegetasi Tergenang yang Mengering + NDVI < 0.3). Rekomendasi Intervensi: Kirimkan regu untuk melakukan pembasahan lahan/suntik gambut besar-besaran, pembuatan sekat bakar parit tidak akan efektif karena api menjalar di bawah tanah."*

Pendekatan membongkar genom api (*Escalome*) ini adalah manifestasi nyata dari **Society 5.0**: Menggabungkan kecerdasan ruang siber (AI/Data Mining satelit) ke dalam ruang fisik (Tindakan taktis patroli lapangan yang presisi) untuk menyelamatkan nyawa manusia dan meminimalisir emisi karbon di Indonesia.
