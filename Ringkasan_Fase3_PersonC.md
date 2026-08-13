# Laporan Penyelesaian: Fase 3 (Machine Learning & Pattern Mining)

Dokumen ini adalah *handover* (serah terima) resmi dari **Person C (Machine Learning & Pattern Mining Lead)** kepada tahap penulisan laporan teknis tim. Di sini dirangkum seluruh tahapan, keputusan teknis, temuan kritis, serta *Kamus Output* lengkap agar tidak ada miskomunikasi mengenai isi setiap file yang dihasilkan dan cara menggunakannya dalam laporan akhir.

---

## 1. Persiapan Data & Audit (Step 8A)

**Masalah:** Dataset dari Person B (cluster_master_PersonB.csv) perlu diverifikasi ulang sebelum dimasukkan ke model — memastikan tidak ada leakage, fitur future, atau kolom yang tidak boleh digunakan sebagai predictor.

**Solusi & Proses:**
*   Audit menyeluruh terhadap 29 kolom cluster_master_PersonB.csv (2.888 baris).
*   Ditemukan bahwa growth_ratio adalah **target-derived future feature** — ia dihitung dari perbandingan hotspot H0 vs H+1/H+3, yang berarti **tidak tersedia di saat prediksi (H0)**. Temuan ini dikonfirmasi via audit temporal (STEP9B_TEMPORAL_VALIDITY_REPORT.md).
*   Dataset akhir siap modeling menggunakan **20 fitur fisik murni H0** yang tidak mengandung informasi masa depan.

**Hasil:** step8_features_ready.csv — 2.888 baris, 20 fitur + 1 label + identitas. Proporsi positif: 153/2888 = **5.3%** (extreme class imbalance).

---

## 2. LightGBM Cost-Sensitive Modeling (Step 8B)

**Masalah:** Extreme class imbalance (5.3% positif) membuat model naif memprediksi semua sebagai negatif.

**Solusi & Proses:**
*   **Dua model dilatih** secara bersamaan:
    *   **Model I (Early-Warning):** tanpa growth_ratio. Ini model operasional yang sah untuk prediksi H0.
    *   **Model II (Diagnostic/Post-Hoc):** dengan growth_ratio. Hanya untuk analisis retrospektif.
*   Kedua model menggunakan scale_pos_weight (cost-sensitive) untuk menangani imbalance.
*   Evaluasi menggunakan PR-AUC, F1-Score (macro), dan MCC.

**Keputusan Metodologis Krusial:**
Model operasional early-warning = Model I (tanpa growth_ratio). Model II hanya referensi diagnostik.

**Hasil:** step8_test_predictions_no_gr.csv + step8_lightgbm_no_growth_ratio_comparison.txt

---

## 3. SHAP Analysis (Step 9A, 9B, 9C)

**Step 9A - SHAP Global:** Feature importance global Model I. total_hotspots = fitur paling berpengaruh.

**Step 9B - Temporal Validity Audit (Temuan Kritis):**
*   growth_ratio menggunakan data H+1..H+3 yang tidak tersedia di H0.
*   Kesimpulan: growth_ratio adalah **target-derived future feature**, hanya valid untuk post-hoc analysis.
*   Laporan: STEP9B_TEMPORAL_VALIDITY_REPORT.md

**Step 9C - SHAP Interaction Values:**
*   15 pasangan fitur dengan nilai interaksi SHAP terkuat.
*   **Top pair:** total_hotspots x population_density (0.052) - sinyal Human-induced.
*   **Top drought pair:** total_hotspots x fuel_danger_index (0.047).
*   Output: step9c_shap_interaction_top15.csv

---

## 4. Archetype Discovery (Step 10A - K-Means)

**Temuan Kritis:** K=2 terbaik, Silhouette=0.1591 (struktur lemah).

Framing yang tepat: **archetype spectrum/gradient** bukan 4 kategori diskrit. Data tidak mendukung 4 archetype yang rigid — ini temuan valid, bukan kegagalan. Konsisten dengan proposal yang menyebut archetype sebagai pola yang ditemukan, bukan dipaksakan.

**Hasil:** step10a_cluster_profiles.csv + STEP10A_ARCHETYPE_DISCOVERY_REPORT.md

---

## 5. Graph Community Detection (Step 10B)

**Temuan:**
*   163 edges / 2.888 nodes = densitas ~0.000039 (sangat sparse).
*   94% komunitas adalah isolated nodes.
*   Mendukung interpretasi bahwa kebakaran umumnya independen secara spatiotemporal.

**Hasil:** step10b_community_profiles.csv + step10b_archetype_assignments.csv + STEP10B_GRAPH_ARCHETYPE_REPORT.md

---

## 6. Association Rule Mining (Step 11 & 11A)

**Step 11:** 495 valid rules (lift>=2). 76 STRONG rules. 15 Pattern Library candidates.

**Koreksi Terminologi Kritis (Step 11A):**
Klaim "baseline lift = 18.9x" di Step 11 adalah **kesalahan terminologi**.
- True baseline lift = 1.0 (random)
- 18.9x = theoretical maximum (jika conf=1.0)
- Top rules kita: lift 3.8-5.4x = bermakna, terminologinya saja yang dikoreksi.

**Hasil validasi:**
*   5-fold CV: semua 15 candidates STABLE (CV < 0.15).
*   Cutpoint sensitivity: semua sensitif terhadap P25/75 boundary.
*   Grade: semua **B (Exploratory)** - Grade A = 0.
*   154 rules Unclear dipertahankan, tidak dipaksa ke archetype.

**hotspot_low (total_hotspots <= 2):**
*   Esc rate: 16.64% vs baseline 5.30% = **lift 3.14x, CV=0.02 (sangat stabil)**.
*   Interpretasi: cluster kecil + kondisi berbahaya = "surprising escalation" = early-warning sejati.
*   Bukan kausal - association dengan kondisi lingkungan.

---

## 7. Pattern Library Final (Step 12)

**15 Pattern Final (Grade B - Exploratory):**

| Archetype | Count | Lift Terbaik |
|---|---|---|
| Drought-driven | 4 | 5.13 (fuel_danger_high & hotspot_low & road_mid) |
| Multi-factor | 4 | 4.75 (hotspot_low & ndvi_low & wind_high) |
| Peat-driven | 3 | 4.09 (hotspot_low & peat_yes & rain14d_med) |
| Human-induced | 2 | 5.44 (hotspot_low & lon_west & pop_dense) |
| Wind-driven | 2 | 4.56 (hotspot_low & lon_west & wind_high) |

Pattern lift tertinggi: hotspot_low & lon_west & pop_dense (lift=5.44, Human-induced)
Pattern coverage tertinggi: hotspot_low & ndvi_low & peat_yes (cov=7.2%, Multi-factor)

**Constraints yang dipatuhi:**
- growth_ratio tidak ada di Pattern Library
- Semua fitur dari H0 saja
- Association != causation (tidak ada klaim kausal)
- hotspot_low threshold: total_hotspots <= 2 (P33, konsisten)

---

## 8. KAMUS OUTPUT

### Script Python - Penamaan GitHub
Teman-teman menggunakan format: NN_deskripsi_singkat.py (file Python bernomor urut).
Script Person C dilanjutkan dari nomor 16 (Person B sampai script 15):

| File Saat Ini | Rename untuk GitHub | Fungsi |
|---|---|---|
| step8_lightgbm_baseline.py | 16_lightgbm_cost_sensitive.py | Training Model I & II |
| step9a_shap_global.py | 17_shap_global_analysis.py | SHAP feature importance |
| step9b_temporal_audit.py | 18_temporal_validity_audit.py | Audit growth_ratio |
| step9c_shap_interaction.py | 19_shap_interaction_values.py | SHAP interaction pairs |
| step10a_archetype_clustering.py | 20_archetype_clustering_kmeans.py | K-Means archetype |
| step10b_graph_archetype.py | 21_graph_community_archetype.py | Graph community detection |
| step11_association_rules.py | 22_association_rule_mining.py | Apriori mining |
| step11a_rule_validation.py | 23_rule_validation_stability.py | Validasi & grading rules |
| step12_pattern_library.py | 24_pattern_library_integration.py | Pattern Library synthesis |

**APAKAH RENAME MEMENGARUHI HASIL/CODE?**
**TIDAK.** Semua script membaca/menulis ke path absolut (G:\My Drive\FireEscalome\).
Nama file script tidak digunakan di dalam kode. Rename 100% aman.

### File yang Push ke GitHub (ikuti format teman - .py saja)
Push: 16_lightgbm_cost_sensitive.py s/d 24_pattern_library_integration.py
Push: Ringkasan_Fase3_PersonC.md (file ini)
TIDAK push: file .csv besar (>50KB), file .pkl, file .txt raw output

### Data Output Utama (Satu Folder)
Disarankan buat folder output_PersonC/ berisi:
- step12_pattern_library_final.csv (UTAMA - 15 patterns)
- step9c_shap_interaction_top15.csv
- step10a_cluster_profiles.csv
- step11_top_rules.csv
- step11a_rule_stability.csv
- step12_pattern_library_summary.png
- step9c_shap_interaction_heatmap.png
- step9a_shap_summary.png
- Semua STEP*.md laporan (10 file)

---

## 9. Hal Krusial untuk Technical Report

### WAJIB masuk laporan:
1. Koreksi growth_ratio sebagai future feature (STEP9B) - membedakan karya ini dari analisis naif
2. Framing "archetype spectrum" bukan 4 kategori diskrit - jujur terhadap data
3. hotspot_low sebagai early-warning signal yang stabil (lift 3.14x, CV=0.02)
4. Pattern Library Grade B - stabil di CV, perlu validasi eksternal. Jangan overclaim.
5. Dua model yang berbeda (Early-Warning vs Diagnostic) - perbedaan fundamental

### TIDAK boleh diklaim:
- Association rules sebagai kausalitas
- growth_ratio sebagai predictor H0
- Pattern Library siap deployment operasional
- 4 archetype diskrit yang rigid
- Baseline lift = 18.9x (koreksi: baseline=1.0; top rules=5.44x)

---

*Fase 3 selesai: 11 Agustus 2026 | Person C - FireEscalome GEMASTIK XIX/2026*
*Siap serah terima untuk penulisan Technical Report.*
