# Laporan Penyelesaian: Fase 2 (Spatio-Temporal Graph & Explainability)

Dokumen ini adalah *handover* (serah terima) resmi dari **Person B (Graph & Network Lead)** kepada **Person C (Machine Learning & Pattern Mining Lead)**. Di sini dirangkum seluruh tahapan, keputusan teknis, inovasi, serta *Kamus Output* lengkap agar tidak ada miskomunikasi mengenai isi setiap file yang dihasilkan dan cara menggunakannya untuk tahap selanjutnya.

---

## 1. Persiapan Data (Merge & Validasi)
**Masalah:** Data serah terima dari Person A terbagi di dua file — `cluster_master_features.csv` (berisi waktu kejadian) dan `tabular_master_final.csv` (berisi fitur fisik lengkap). Graf membutuhkan keduanya dalam satu tabel.
**Solusi & Proses:**
*   Kedua file di-*merge* via `cluster_id` menggunakan operasi `LEFT JOIN`.
*   File hasil merge disimpan ke folder baru `data_processed_B/` agar **tidak menyentuh file asli Person A sama sekali**.
*   Kolom `wind_alignment_score` dari `tabular_master_final.csv` digunakan langsung sebagai nilai cos(θ) Wind Alignment, karena sudah dihitung Person A dari data ERA5 (u10, v10) yang presisi — jauh lebih akurat daripada menghitung ulang.
**Hasil:** `cluster_master_PersonB.csv` — 2.888 baris × 29 kolom, 0 NaN di semua kolom kritis.

---

## 2. Konstruksi Graf Spasial-Temporal (Step 7)
**Masalah:** Hubungan antar kejadian kebakaran belum pernah dimodelkan secara eksplisit. Pertanyaan Pak Daniel: *"Titik panas A dan B ini terhubung KARENA APA?"* belum terjawab.
**Solusi & Proses:**
*   Dibangun Directed Graph G = (V, E) menggunakan `networkx`, di mana setiap node adalah satu cluster kebakaran.
*   Dua node membentuk *edge* (koneksi) HANYA JIKA: **(a)** jaraknya ≤ 5 km (dihitung via *haversine*, dipercepat KDTree pre-filter) DAN **(b)** waktunya berurutan 0 < Δt ≤ 3 hari dalam tahun yang sama.
*   Setiap *edge* memiliki bobot Wij yang didekomposisi dari 4 komponen:

| Komponen | Bobot (α/β/γ/δ) | Rumus |
|---|---|---|
| Spatial Decay (Kedekatan Jarak) | α = 0.25 | `exp(-0.5 × d_km)` |
| Wind Alignment (Angin ERA5) | β = 0.35 | `wind_alignment_score` node sumber |
| Slope Direction (Lereng DEM) | γ = 0.15 | `1 - exp(-0.05 × slope_target)` |
| Peatland Continuity (Gambut) | δ = 0.25 | 1.0 / 0.5 / 0.0 |

*   Dekomposisi per komponen ini menjadikan graf **transparan sejak awal** — setiap *edge* sudah tahu persis "alasan keberadaannya" tanpa perlu model *black box*.
**Hasil:** 7.882 kandidat spasial → **163 edge valid** setelah filter haversine + temporal.

---

## 3. Penemuan Komunitas & Archetype (Bonus Jembatan untuk Person C)
**Konteks:** Sesuai pembagian tugas, *Graph Community Detection* adalah bagian dari Step 10–12 milik Person C. Namun, iterasi awal dijalankan sebagai *"jembatan serah terima"* agar Person C bisa langsung memulai *Archetype Discovery* dan *Association Rules* tanpa harus membangun dari nol — serupa dengan Person A yang menyiapkan baseline LightGBM sebagai jembatan untuk Person B.
**Proses:**
*   Louvain Algorithm dijalankan pada `spatiotemporal_graph.pkl` untuk mendeteksi komunitas alami dari struktur koneksi graf.
*   Tiap komunitas dianalisis berdasarkan faktor *edge* yang paling dominan di dalamnya → ini yang menjadi label *archetype*.
**Hasil:** 6 komunitas signifikan (≥ 3 node) dengan 3 archetype berbeda:
*   **Wind-driven** (35%): Api menjalar karena tiupan angin searah.
*   **Proximity-driven** (35%): Api menjalar karena kedekatan jarak fisik semata.
*   **Peat-driven** (30%): Api menjalar lambat melalui sambungan gambut bawah tanah.

---

## 4. Graph Explainability (Step 9)
**Masalah:** Pak Daniel meminta: *"Faktor apa yang paling menentukan dua node terhubung?"* Ini perlu dijawab dengan angka, bukan asumsi.
**Solusi & Proses:**
*   Karena Wij sudah didekomposisi per komponen sejak Step 7, *explainability* adalah analisis statistik murni dari tabel *breakdown edge* — tidak membutuhkan GNN atau model *black box* sama sekali.
*   4 analisis dijalankan: distribusi faktor global, perbandingan faktor edge eskalasi vs non-eskalasi, profil faktor per komunitas, dan top-5 *edge* terkuat per faktor.
**Hasil & Temuan Utama:**

| Kelompok Edge | Faktor Dominan | Persentase |
|---|---|---|
| Semua edge (global) | Spatial Decay (Jarak) | 40.5% |
| Edge non-eskalasi | Spatial Decay (Jarak) | 42.8% |
| **Edge eskalasi (api raksasa)** | **Peatland Continuity (Gambut)** | **55.6%** |

> **Kesimpulan Emas:** Api biasa menular hanya karena kedekatan jarak. Namun api yang berkembang menjadi *bencana raksasa* (eskalasi), **55.6%** kasusnya disebabkan oleh sambungan gambut bawah tanah — bukan angin, bukan lereng.

---

## 5. KAMUS OUTPUT (Isi & Fungsi Setiap File)

Semua file berada di: `G:/My Drive/gemastik/karhutla/data_processed_B/`

### A. File Data Utama
- **`cluster_master_PersonB.csv`**: Dataset gabungan 2.888 cluster (29 kolom) — bahan baku untuk semua script Graf. Jangan dipakai untuk training model tabular; gunakan `tabular_master_final.csv` milik Person A untuk itu.
- **`spatiotemporal_graph.pkl`**: Objek graf NetworkX lengkap. Load dengan `pickle.load()`. Berisi 2.888 node (dengan atribut fitur) dan 163 *directed edge* (dengan atribut bobot Wij per komponen).

### B. File Analisis & Laporan
- **`edge_statistics.csv`**: Tabel 163 baris. Tiap baris = satu *edge* dengan kolom: `source_id`, `target_id`, `distance_km`, `delta_days`, `weight_total`, `spatial_decay`, `wind_alignment`, `slope_direction`, `peatland_continuity`, `dominant_factor`, `source_label`, `target_label`.
- **`community_analysis.csv`**: Tabel analisis per komunitas Louvain. Kolom: `community_id`, ukuran, % eskalasi, archetype, dan rata-rata nilai tiap komponen edge.
- **`graph_explainability_report.csv`**: Laporan akhir perbandingan faktor dominan antara edge eskalasi dan non-eskalasi.

### C. File Visualisasi
- **`graph_visualization.png`**: Peta visual graf spasial (node diplot berdasarkan koordinat lat/lon, edge diwarnai berdasarkan `dominant_factor`).
- **`graph_factor_comparison.png`**: Grafik batang perbandingan rata-rata nilai 4 komponen Wij antara edge eskalasi vs non-eskalasi.

---

## 6. Handover ke Person C

Halo Person C (dan agen AI Person C)! 👋

Tugas Fase 2 sudah rampung. Berikut hal yang perlu diperhatikan untuk melanjutkan ke Fase 3:

1.  **Bahan baku utamamu** adalah `tabular_master_final.csv` dari Person A (untuk LightGBM + SHAP) dan file-file di `data_processed_B/` dari saya (untuk melengkapi analisis graf).
2.  **Untuk Step 8 (LightGBM):** Person A sudah menyiapkan script `11_modeling_lightgbm.py` sebagai *baseline* — tinggal kamu jalankan dan kembangkan dengan *cost-sensitive learning*.
3.  **Untuk Step 9 Tabular (SHAP):** Jalankan analisis SHAP pada model LightGBM, lalu cocokkan hasilnya dengan temuan *Graph Explainability* saya untuk narasi yang kuat.
4.  **Untuk Step 10–12 (Archetype & Association Rules):** Saya sudah menjalankan iterasi awal Community Detection (Louvain). Hasilnya di `community_analysis.csv` bisa langsung kamu gunakan sebagai titik awal *Archetype Discovery* dan *Association Rule Mining*.

Semangat! Data sudah setajam mungkin. Saatnya kamu cetak gol di babak akhir! 🚀
