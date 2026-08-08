# DOKUMEN PEGANGAN MASTER — GEMASTIK XIX/2026 Divisi Penambangan Data
## "Decoding the Fire Escalome: Knowledge Discovery of Hidden Escalation Mechanisms in Indonesian Wildland Fires"

**Versi 4 — FINAL. Menggantikan semua versi sebelumnya. Tenggat: 14 Agustus 2026.**

---

## RINGKASAN 5W + 1H

| | Penjelasan |
|---|---|
| **WHAT** | Knowledge Discovery in Databases (KDD) untuk menemukan **Taksonomi Mekanisme Eskalasi Karhutla** (Escalation Archetypes) di Indonesia — bukan sekadar model prediksi biner. Output: taksonomi archetype (peat/wind/drought/human-driven) + Early-Warning Pattern Rules + Pattern Library terstruktur. |
| **WHY** | (1) Operasional: BPBD/Manggala Agni overload ribuan titik panas/hari, penanganan one-size-fits-all tidak efektif karena strategi padam peat-driven ≠ wind-driven. (2) Keilmuan: sistem existing (SiPongi) occurrence-based, belum ada yang membongkar heterogenitas mekanisme eskalasi secara terstruktur di konteks Indonesia. |
| **WHO** | Pengguna: BPBD & Manggala Agni (rekomendasi intervensi spesifik per archetype). Penerima manfaat ilmiah: komunitas Penambangan Data & Earth Observation. |
| **WHERE** | 5 provinsi prioritas: Riau, Sumsel, Jambi (BBOX Sumatra: 95°BT–109°BT, 6°LS–3°LU), Kalbar, Kalteng (BBOX Kalimantan: 108°BT–118°BT, 4°LS–4°LU) |
| **WHEN** | Data historis 6 tahun (1 Jan 2020 – Agustus 2026) untuk training & taxonomy discovery. Fase operasional: simulasi NRT (FIRMS NRT ~3 jam delay + Open-Meteo Forecast API). Tenggat kerja: 4–14 Agustus 2026. |
| **HOW** | Multi-source data integration → sanitasi anti-false-positive → feature engineering spasial-temporal-interaksi → cost-sensitive modeling → archetype clustering → association rule mining → Pattern Library. Detail lengkap di Bagian 6–8. |

---

## 1. Judul & Posisi Final

**Decoding the Fire Escalome: Knowledge Discovery of Hidden Escalation Mechanisms in Indonesian Wildland Fires**

Kata kunci yang harus konsisten dipakai di seluruh laporan:
- **Escalome** — istilah baru (analogi "genome"/"microbiome") untuk merujuk pada keseluruhan mekanisme eskalasi sebagai sistem yang bisa "dibongkar" (decoded), bukan diprediksi sebagai skor tunggal
- **Archetype** — bukan "kategori" atau "kelas" — istilah ini menegaskan tiap tipe punya *trigger path* (jalur pemicu) yang secara kualitatif berbeda, bukan cuma beda nilai fitur
- **Triase Prediktif** — prioritisasi patroli berbasis archetype + tingkat risiko, bukan sekadar skor probabilitas

---

## 2. Research Questions (final, jangan berubah lagi)

- **RQ1 (Early Signals):** Kombinasi sinyal spasial-temporal apa yang konsisten muncul sebelum eskalasi?
- **RQ2 (Anomalous Comparison):** Mengapa cluster dengan kondisi lingkungan serupa bisa punya nasib eskalasi berbeda?
- **RQ3 (Critical Transition):** Interaksi faktor apa yang jadi *tipping point* kritis sebelum eskalasi?
- **RQ4 (Escalation Archetypes):** Apakah ada taksonomi pola eskalasi berbeda (peat-driven, wind-driven, drought-driven, human-induced)?

**H1 — Hipotesis Paradoks Pareto (uji PALING PERTAMA, prasyarat semua bab lain):**
> Cluster yang bereskalasi signifikan berjumlah <10% dari total cluster aktif, tapi menyumbang >50% dari total akumulasi FRP di wilayah prioritas.

---

## 3. Definisi Konseptual Kunci (REVISI PENTING — baca sebelum kerja)

### 3a. Eskalasi — sekarang divalidasi ground truth, BUKAN cuma heuristik FIRMS
**Definisi lama (draft awal):** cluster kecil yang jumlah titik paniknya melonjak ≥3x dalam 3 hari — ini heuristik dari data FIRMS sendiri, rentan kritik "kenapa 3x?".

**Definisi final (upgrade):** cluster dinyatakan **eskalasi tervalidasi** jika:
1. Secara heuristik FIRMS menunjukkan lonjakan (persentil ke-95 dari distribusi rasio pertumbuhan — bukan angka bulat), **DAN**
2. **MODIS Burned Area (MCD64A1)** mengonfirmasi area tersebut benar-benar berkembang jadi area terbakar signifikan pada rentang tanggal yang sama (pakai atribut *burn date* per piksel dalam produk MCD64A1, bukan cuma info bulanan agregat)

Ini yang bikin label bukan lagi "self-referential" (FIRMS memprediksi FIRMS) — ada ground truth independen dari produk satelit tervalidasi NASA. Ini jauh lebih defensible di depan reviewer.

### 3b. Escalation Archetype
Pengelompokan tipe eskalasi berdasarkan *signature* faktor pemicu dominan, ditemukan lewat clustering pada ruang fitur interaksi (bukan ditentukan manual/asumsi):
- **Peat-driven:** gambut=True, curah hujan≈0 selama periode lama, eskalasi cenderung *smoldering* (lambat tapi luas)
- **Wind-driven:** kecepatan angin tinggi, kelembapan rendah, densitas tetangga naik cepat → eskalasi permukaan cepat
- **Drought-driven:** indikator ENSO/IOD kuat, defisit curah hujan jangka panjang
- **Human-induced:** jarak ke jalan dekat, dekat pemukiman/lahan garapan

### 3c. Triase Prediktif
Output akhir: skor risiko + label archetype + rule yang memicu → rekomendasi strategi intervensi spesifik (bukan generik "kirim tim pemadam").

---

## 4. Arsitektur Data — TIERING (jangan kerjakan 8 sumber setara!)

Status: FIRMS (394.000+ baris) & Open-Meteo (12.000+ baris), 2020–2026, **SUDAH SELESAI**.

| Tier | Sumber | Kegunaan | Akses | Kapan dikerjakan |
|---|---|---|---|---|
| **0 — Selesai** | FIRMS, Open-Meteo | Deteksi + cuaca | — | ✅ Done |
| **1 — Prioritas TERTINGGI** | **MODIS Burned Area (MCD64A1)** | Ground truth label eskalasi (revisi Bagian 3a) | Google Earth Engine: `MODIS/061/MCD64A1` | Hari 1–2 (mendefinisikan ulang label — semua bergantung ini) |
| **1** | Peta Gambut (KLHK/Wetlands International) | Indikator peat-driven | Shapefile: tanahair.indonesia.go.id / data.go.id | Hari 1–2 (statis, sekali join) |
| **1** | SRTM DEM | Elevasi, slope (arah rambat api) | GEE: `USGS/SRTMGL1_003` | Hari 1–2 (statis, murah) |
| **2 — Value tinggi** | Land Cover (Dynamic World, fallback ESA WorldCover 2021) | Filter false positive urban/industri + fuel type | GEE: `GOOGLE/DYNAMICWORLD/V1` atau `ESA/WorldCover/v200` | Hari 3–4 |
| **2** | NDVI | Kekeringan bahan bakar vegetasi | GEE: MODIS NDVI 16-day composite | Hari 3–4 |
| **3 — Kalau waktu cukup** | OpenStreetMap (jarak jalan) | Indikator human-induced | `osmnx` Python library | Hari 5 |
| **3** | WorldPop | Kepadatan penduduk, human-induced | worldpop.org direct download | Hari 5 |
| **4 — Opsional, boleh skip** | GHSL (permukiman) | Redundant dengan WorldPop+OSM | GEE: `JRC/GHSL/...` | Hanya jika Tier 1-3 selesai lebih cepat dari jadwal |

**Prinsip kerja:** kalau Tier 4 tidak sempat, itu TIDAK apa-apa — laporan tetap solid dengan Tier 0-3. Kalau MCD64A1 (Tier 1) molor, SEMUA jadwal berikutnya ikut molor karena label eskalasi final bergantung padanya — prioritaskan ini di atas segalanya di hari 1-2.

**Catatan teknis:** MCD64A1, Dynamic World, dan NDVI MODIS semuanya lebih realistis diakses lewat **Google Earth Engine Python API** (`earthengine-api`), bukan download manual HDF/GeoTIFF — ini skill baru yang belum pernah dipakai tim untuk FIRMS/Open-Meteo (yang REST API biasa). Alokasikan waktu di hari 1 untuk setup akun GEE (gratis untuk riset/edukasi, approval biasanya cepat) dan tes autentikasi sebelum mulai narik data sungguhan.

---

## 5. Pipeline Data Sanitasi (Anti False-Positive)

```
NASA FIRMS (VIIRS 375m)
    │  Deteksi anomali suhu permukaan
    ▼
Land Cover & Peatland Filtering (ESA WorldCover/Dynamic World + Peta Gambut)
    │  Keep: Hutan, Gambut, Semak Belukar
    │  Drop: Urban, Industri, Flare Oil & Gas
    ▼
Forest Fire Candidate Dataset
    │
    ▼
Burned Area Validation (MCD64A1) ──► Label Eskalasi Final (bukan heuristik semata)
    │
    ▼
Integrasi Cuaca + Fitur Spasial-Temporal + Topografi + Antropogenik
    │
    ▼
Pattern Discovery, Clustering & Archetype Rule Extraction
```

---

## 6. Metodologi Lengkap — Step by Step

### Step 1 — Grid & Agregasi (sudah ada kodenya)
Grid spasial 0.1° (~11km), agregasi harian hotspot count + FRP sum per sel. *(Reuse `02_feature_engineering_modeling.py` yang sudah ditulis & di-debug sebelumnya sebagai basis.)*

### Step 2 — Filtering Sanitasi
Overlay grid dengan Land Cover — buang sel yang dominan urban/industri. Overlay dengan peta gambut — tandai `is_peatland` per sel.

### Step 3 — Label Eskalasi Final (REVISI)
1. Hitung heuristik FIRMS (persentil ke-95 rasio pertumbuhan, bukan 3x tetap)
2. Cross-check dengan MCD64A1: apakah sel tsb tercatat *burned* pada window tanggal yang relevan
3. Label positif = kedua kondisi terpenuhi. Dokumentasikan berapa banyak kandidat heuristik yang TIDAK terkonfirmasi MCD64A1 (ini sendiri temuan menarik untuk RQ2 — kenapa FIRMS bilang eskalasi tapi tidak jadi burned area riil?)

### Step 4 — Feature Engineering Interaksi
Selain fitur dasar (lag, tetangga spasial, cuaca) yang sudah dirancang, tambahkan fitur INTERAKSI eksplisit:
- `precip_dry_streak × is_peatland` (indikator peat-driven)
- `windspeed × humidity_inverse × neighbor_density` (indikator wind-driven)
- `road_distance_inverse × population_density` (indikator human-induced)
- `enso_index × ndvi_dryness` (indikator drought-driven)

### Step 5 — Modeling Cost-Sensitive
LightGBM + `scale_pos_weight` (sama seperti sebelumnya) sebagai model prediksi utama — bukan untuk klaim akurasi, tapi sebagai fondasi untuk SHAP Interaction Values di Step 6.

### Step 6 — SHAP Interaction Values
Bukan SHAP feature importance biasa (itu cuma explainability) — pakai **SHAP Interaction Values** untuk mengidentifikasi PASANGAN fitur yang berinteraksi kuat memicu eskalasi. Ini input untuk Step 7 & 8.

### Step 7 — Archetype Clustering
Pada subset event eskalasi (label positif saja), lakukan clustering (misal K-Means atau hierarchical clustering) pada ruang fitur interaksi hasil Step 6 — temukan kluster alami. Cek apakah kluster yang terbentuk match dengan 4 archetype dugaan (peat/wind/drought/human), atau justru muncul archetype lain yang tidak terduga (ini akan jadi temuan yang LEBIH kuat kalau terjadi — data mining sejati menemukan, bukan mengonfirmasi asumsi).

### Step 8 — Association Rule Mining
Pakai `mlxtend` (Apriori + association_rules):
1. Diskritisasi fitur kunci (hasil dari SHAP interaction Step 6) jadi bin bermakna
2. Bentuk transaksi biner per cluster-hari
3. Jalankan Apriori, filter rules dengan consequent = "Escalation=Yes"
4. Urutkan by lift & confidence
5. Kelompokkan rules berdasarkan archetype hasil Step 7

Output contoh:
```
Rule (Peat-driven): Peatland=True ∧ Precip=0(14 hari) ∧ ΔFRP↑ 
    ⟹ Smoldering Escalation (support=6%, confidence=84%, lift=3.2)

Rule (Wind-driven): Windspeed>X ∧ Humidity<Y% ∧ NeighborDensity↑ 
    ⟹ Fast Surface Escalation (support=4%, confidence=87%, lift=4.1)
```

### Step 9 — Pattern Library / Escalation Knowledge Base (output final)
Susun katalog terstruktur berisi, untuk TIAP archetype:
- Nama & deskripsi archetype
- Rule defining (dari Step 8) lengkap dengan support/confidence/lift
- Kondisi lingkungan tipikal (ringkasan statistik)
- Rekomendasi strategi intervensi spesifik (sekat bakar cepat untuk wind-driven vs pembasahan lahan untuk peat-driven)
- Contoh kasus historis (1-2 cluster nyata dari data yang match archetype ini)

Ini yang jadi kontribusi utama laporan — bukan cuma tabel metrik model.

---

## 7. Arsitektur Sistem — Fase Training vs Operasional (NRT)

```
FASE 1: Offline Training & Taxonomy Discovery (Data 2020–2026)
FIRMS + Open-Meteo + Land Cover + Peat + Burned Area + NDVI + Topografi
    │  Archetypes & Rules terbentuk
    ▼
FASE 2: Near Real-Time (NRT) Triase Operasional (Simulasi)
FIRMS NRT (~3 jam delay) + Open-Meteo Forecast API
    │  Klasifikasi cluster baru ke archetype + skor risiko
    ▼
Output: peta triase ter-update, dengan rekomendasi strategi per archetype
```

Istilah presisi: **"near real-time"**, bukan real-time murni (lihat justifikasi di dokumen versi sebelumnya — keterbatasan inheren satelit polar-orbit, bukan kelemahan).

**Simulasi demo (hari 10, sebelum submit):** jalankan model+archetype classifier terhadap data NRT 7-10 hari terakhir, tunjukkan output triase riil sebagai bukti pipeline operasional bekerja pada data baru.

---

## 8. Kesejajaran Society 5.0

- **Human-centered:** keselamatan warga wilayah rawan asap/api, strategi intervensi presisi per archetype
- **CPS integration:** satelit (fisik) → archetype classifier (siber) → strategi patroli spesifik (aksi fisik)
- **Masalah terukur:** H1 (persentase cluster vs kontribusi FRP), jumlah rule tervalidasi per archetype
- **SDGs:** SDG 13 (Climate Action), SDG 15 (Life on Land)
- **Teknologi bermakna:** KDD lengkap (bukan cuma model prediksi), ground truth tervalidasi MCD64A1
- **Resilience:** triase presisi per archetype > triase generik

---

## 9. Timeline Lengkap (4–14 Agustus 2026)

| Tanggal | Fokus | Output |
|---|---|---|
| **4 Agu** | Setup GEE, uji H1 dengan data FIRMS riil, mulai tarik MCD64A1 untuk wilayah prioritas | H1 teruji + akses GEE siap |
| **5 Agu** | Selesaikan MCD64A1 + peta gambut + SRTM, bentuk label eskalasi final (Step 3) | Label tervalidasi ground truth |
| **6 Agu** | Land cover (Dynamic World/WorldCover) + NDVI, feature engineering interaksi (Step 4) | Dataset fitur lengkap |
| **7 Agu** | Modeling cost-sensitive + SHAP Interaction Values (Step 5-6) | Model + pasangan fitur interaksi kunci |
| **8 Agu** | Archetype clustering (Step 7) | Taksonomi archetype teridentifikasi |
| **9 Agu** | Association Rule Mining (Step 8) | Daftar rules per archetype |
| **10 Agu** | Pattern Library curation (Step 9) + OSM/WorldPop kalau sempat | Katalog Pattern Library lengkap |
| **11-12 Agu** | Penulisan Technical Report 8 bagian penuh | Naskah 100% |
| **13 Agu** | Simulasi demo NRT, uji similaritas, surat pernyataan, polish | Siap review |
| **14 Agu** | Review akhir, submit | Selesai |

---

## 10. Kerangka 8 Bagian Technical Report

1. **Judul** — Bagian 1
2. **Abstrak** — tulis TERAKHIR, sertakan angka H1 + jumlah archetype ditemukan + jumlah rules tervalidasi
3. **Pendahuluan** — gap occurrence vs escalation (SiPongi) + hasil H1 sebagai fakta pembuka
4. **Kajian Terkait** — riset prediksi hotspot existing (occurrence-based) vs posisi taksonomi archetype ini
5. **Solusi Usulan** — Bagian 6 (metodologi 9 step) lengkap, termasuk revisi label MCD64A1
6. **Hasil Eksperimen** — H1, metrik model (MCC/PR-AUC/F1), hasil clustering archetype, daftar rules
7. **Analisis** — interpretasi tiap archetype, validasi ulang RQ1-RQ4 dengan temuan konkret
8. **Kesimpulan** — Pattern Library sebagai kontribusi utama, keterbatasan jujur (jumlah archetype masih hasil clustering data historis, perlu validasi lapangan berkelanjutan)

---

## 11. Persiapan Presentasi ke Dosen — Q&A Terbaru

| Pertanyaan | Jawaban |
|---|---|
| "Apa bedanya dengan versi sebelumnya yang cuma prediksi?" | Shift dari single-target prediction ke taxonomy discovery — kontribusi utama adalah archetype + Pattern Library, model prediksi cuma alat bantu (lihat Step 5-9) |
| "Kenapa label pakai MCD64A1, bukan cuma FIRMS?" | FIRMS heuristik rentan false positive; MCD64A1 ground truth independen dari NASA — label jadi tervalidasi, bukan self-referential |
| "Bagaimana kalau archetype yang ditemukan clustering tidak match 4 dugaan awal (peat/wind/drought/human)?" | Itu justru temuan lebih kuat — laporan akan mendokumentasikan apa adanya, karena itu esensi data mining (menemukan, bukan mengonfirmasi asumsi) |
| "Association Rule Mining beda dari SHAP gimana?" | SHAP = explainability (kontribusi fitur ke 1 prediksi); Association Rules = knowledge extraction (pola IF-THEN general dari keseluruhan data) — dua hal berbeda, dipakai berurutan |
| "Datanya real-time?" | Training pakai histori 2020-2026; operasional pakai FIRMS NRT + Open-Meteo Forecast API, near real-time (~3 jam delay per lintasan satelit) — lihat Bagian 7 |
| "8 sumber data, sanggup semua dalam waktu segini?" | Tidak semua setara — lihat tiering Bagian 4. Tier 1 (MCD64A1, peat, DEM) wajib; Tier 3-4 (OSM, WorldPop, GHSL) opsional kalau waktu sisa |

---

## 12. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| GEE setup/autentikasi molor di hari 1 | Alokasikan blok waktu khusus hari 1 pagi, jangan digabung task lain |
| MCD64A1 resolusi bulanan tidak match window 3 hari | Pakai atribut *burn date* per piksel (bukan info bulanan agregat) untuk presisi harian |
| Archetype clustering tidak menghasilkan kluster jelas/interpretable | Coba beberapa jumlah kluster (elbow method), dan siapkan narasi jujur kalau hasilnya 2-3 archetype dominan, bukan 4 — itu tetap valid temuan |
| Waktu habis sebelum Tier 3 (OSM/WorldPop) selesai | Sudah diantisipasi di tiering — laporan tetap solid tanpa itu |
| H1 tidak terbukti timpang | Jangan paksakan narasi Pareto; RQ2 (kenapa cluster serupa beda nasib) tetap valid sebagai motivasi archetype tanpa perlu H1 ekstrem |

---

## 13. Checklist Eksekusi Final

- [ ] Akun GEE aktif, autentikasi Python API berhasil (hari 1)
- [ ] H1 diuji dengan data FIRMS riil, angka dikutip di Bab 1
- [ ] Label eskalasi final = heuristik FIRMS + validasi MCD64A1 (bukan cuma satu-satunya)
- [ ] Minimal Tier 1 (MCD64A1, peat, DEM) selesai sebelum lanjut ke modeling
- [ ] SHAP Interaction Values (bukan cuma feature importance biasa) dihasilkan
- [ ] Archetype clustering dijalankan, hasil didokumentasikan apa adanya
- [ ] Association Rule Mining menghasilkan minimal beberapa rules per archetype dengan support/confidence/lift dilaporkan
- [ ] Pattern Library tersusun sebagai bab/lampiran terstruktur, bukan sekadar daftar rules mentah
- [ ] Istilah "escalome", "archetype", "triase prediktif" konsisten di seluruh laporan
- [ ] Simulasi NRT dijalankan minimal sekali sebagai bukti operasional
- [ ] Q&A Bagian 11 dipahami (bukan dihafal)
