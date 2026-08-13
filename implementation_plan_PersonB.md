# Rencana Implementasi: Fase 2 (Spatio-Temporal Graph & Network Analysis)

Script-script `12b`, `13`, `14`, dan `15` ini dirancang sebagai kelanjutan langsung dari *output* Person A. Tujuannya adalah membuktikan bahwa data yang sudah disiapkan Person A dapat digunakan untuk memodelkan **mekanisme penularan api** secara spasial dan temporal — menjawab pertanyaan inti: *"Titik panas A dan B ini terhubung KARENA APA?"*

---

## 1. Persiapan Data — Merge Input dari Person A
Script `12b_merge_data_PersonB.py` menggabungkan dua file warisan Person A:
*   **`cluster_master_features.csv`** → menyumbang kolom identitas waktu: `year`, `start_day_of_year`.
*   **`tabular_master_final.csv`** → menyumbang seluruh fitur fisik: `slope`, `elevation`, `wind_alignment_score`, `windspeed_10m_max`, `is_peatland`, dll.
*   *Merge* dilakukan via `cluster_id` (LEFT JOIN). Output disimpan ke `data_processed_B/` agar **tidak menyentuh file asli Person A**.
*   Kolom `wind_alignment_score` digunakan langsung sebagai nilai cos(θ) Wind Alignment karena sudah dihitung dari data ERA5 (u10, v10) oleh Person A — tidak perlu dihitung ulang.

---

## 2. Konstruksi Graf Spasial-Temporal — Step 7
Script `13_bangun_spatiotemporal_graph.py` adalah inti dari Fase 2. Dua node membentuk *edge* (koneksi penularan api) HANYA JIKA:
*   **Syarat Spasial:** Jarak antar-cluster ≤ 5 km (dihitung via *haversine*, dipercepat *KDTree pre-filtering*).
*   **Syarat Temporal:** Selisih waktu 0 < Δt ≤ 3 hari, dalam tahun yang sama. Graf bersifat *directed* (berarah) — node yang muncul lebih awal menjadi *source*.

Bobot edge Wij dihitung dari 4 komponen fisik nyata:
*   **Spatial Decay** (α=0.25): `exp(-0.5 × d_km)` — makin dekat, makin kuat koneksinya.
*   **Wind Alignment** (β=0.35): `wind_alignment_score` dari ERA5 — makin searah angin, makin mudah api menjalar.
*   **Slope Direction** (γ=0.15): `1 - exp(-0.05 × slope_target)` — lereng curam mempercepat rambatan api ke atas.
*   **Peatland Continuity** (δ=0.25): bernilai `1.0` jika kedua node di atas gambut, `0.5` jika salah satu, `0.0` jika tidak ada.

---

## 3. Community Detection — Jembatan untuk Person C (Step 10 Person C)
Script `14_community_detection_archetype.py` menjalankan **Louvain Algorithm** pada graf hasil Step 7.
*   Sesuai pembagian tugas, langkah ini adalah bagian dari Step 10–12 milik Person C. Dijalankan lebih awal sebagai *jembatan serah terima* agar Person C tidak mulai dari nol.
*   Tiap komunitas dianalisis berdasarkan faktor *edge* dominan di dalamnya, menghasilkan label *archetype*: `Wind-driven`, `Proximity-driven`, atau `Peat-driven`.

---

## 4. Graph Explainability — Step 9 (Bagian Graf)
Script `15_graph_explainability_lite.py` menjawab pertanyaan: *"Faktor apa yang paling dominan memicu dua node terhubung?"*
*   Karena Wij sudah didekomposisi per komponen sejak Step 7, *explainability* adalah **analisis statistik murni** dari tabel *edge breakdown* — tidak membutuhkan GNNExplainer atau model *black box*.
*   Analisis dilakukan dalam 4 lapisan: distribusi global, perbandingan eskalasi vs non-eskalasi, profil per komunitas, dan *Top-5 Edge* terkuat per faktor.
*   **Temuan kunci:** Peatland Continuity (Gambut) mendominasi **55.6%** dari seluruh *edge* yang berujung pada cluster eskalasi.

---

## 5. Urutan Eksekusi (Sudah Selesai)
```
[SELESAI] python 12b_merge_data_PersonB.py   → cluster_master_PersonB.csv
[SELESAI] python 13_bangun_spatiotemporal_graph.py → spatiotemporal_graph.pkl + edge_statistics.csv
[SELESAI] python 14_community_detection_archetype.py → community_analysis.csv
[SELESAI] python 15_graph_explainability_lite.py → graph_explainability_report.csv + visualisasi
```
