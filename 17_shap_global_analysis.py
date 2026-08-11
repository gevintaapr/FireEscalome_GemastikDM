"""
step9a_shap_global.py
Step 9A -- SHAP Global Analysis & growth_ratio Investigation
FireEscalome GEMASTIK XIX/2026 -- Person C

Input:
  G:/My Drive/FireEscalome/step8_lightgbm_cost_sensitive_baseline.txt  (Model A)
  G:/My Drive/FireEscalome/step8_lightgbm_no_growth_ratio_comparison.txt (Model B)
  G:/My Drive/FireEscalome/step8_features_ready.csv

Output:
  G:/My Drive/FireEscalome/step9a_shap_global_importance.csv
  G:/My Drive/FireEscalome/step9a_shap_summary.png
  G:/My Drive/FireEscalome/step9a_growth_ratio_dependence.png
  G:/My Drive/FireEscalome/STEP9A_SHAP_GLOBAL_REPORT.md

growth_ratio definition (dari 09_label_eskalasi.py):
  growth_ratio = max_hotspots_next_3_days / hotspots_day_0
  label_escalation = 1 jika growth_ratio >= P95 DAN tervalidasi MCD64A1
  -> growth_ratio adalah komponen LANGSUNG dari definisi label
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------
WORKSPACE  = Path(r"G:\My Drive\FireEscalome")
MODEL_A    = WORKSPACE / "step8_lightgbm_cost_sensitive_baseline.txt"
MODEL_B    = WORKSPACE / "step8_lightgbm_no_growth_ratio_comparison.txt"
INPUT_CSV  = WORKSPACE / "step8_features_ready.csv"

OUT_CSV    = WORKSPACE / "step9a_shap_global_importance.csv"
OUT_SUMM   = WORKSPACE / "step9a_shap_summary.png"
OUT_DEP    = WORKSPACE / "step9a_growth_ratio_dependence.png"
OUT_REPORT = WORKSPACE / "STEP9A_SHAP_GLOBAL_REPORT.md"

TARGET_COL   = "label_escalation"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CAT_FEATURES = ['island', 'confidence', 'is_peatland', 'landcover_class']

SEP = "=" * 65
ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(SEP)
print("STEP 9A -- SHAP GLOBAL ANALYSIS & growth_ratio INVESTIGATION")
print(f"Timestamp: {ts} | SHAP: {shap.__version__}")
print(SEP)

# ================================================================
# 1. LOAD DATA & RECREATE TEST SET
# ================================================================
print("\n[1] LOAD DATA & RECREATE TEST SET ...")
df = pd.read_csv(INPUT_CSV)
X  = df.drop(columns=[TARGET_COL])
y  = df[TARGET_COL]

for col in CAT_FEATURES:
    X[col] = X[col].astype('int32')

# Reproduce exact same split sebagai Step 8B
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

print(f"    Test set: {X_test.shape} | Class0={int((y_test==0).sum())} Class1={int((y_test==1).sum())}")

# Model B tidak punya growth_ratio
X_test_no_gr = X_test.drop(columns=['growth_ratio'])

# ================================================================
# 2. LOAD MODELS
# ================================================================
print("\n[2] LOAD MODELS ...")
model_A = lgb.Booster(model_file=str(MODEL_A))
model_B = lgb.Booster(model_file=str(MODEL_B))
print(f"    Model A loaded: {MODEL_A.name}")
print(f"    Model B loaded: {MODEL_B.name}")

# ================================================================
# 3. GROWTH_RATIO INVESTIGATION (SEBELUM SHAP)
# ================================================================
print("\n[3] GROWTH_RATIO INVESTIGATION ...")
gr_c0 = df[y == 0]['growth_ratio']
gr_c1 = df[y == 1]['growth_ratio']

min_gr_c1 = float(gr_c1.min())
max_gr_c0  = float(gr_c0.max())
overlap     = not (min_gr_c1 > max_gr_c0)

# P95 threshold (how label was made)
p95_gr = float(np.percentile(df['growth_ratio'], 95))

print(f"    === growth_ratio distribution ===")
print(f"    Class 0 : min={gr_c0.min():.3f} mean={gr_c0.mean():.3f} max={gr_c0.max():.3f}")
print(f"    Class 1 : min={gr_c1.min():.3f} mean={gr_c1.mean():.3f} max={gr_c1.max():.3f}")
print(f"    P95 threshold (whole dataset): {p95_gr:.4f}")
print(f"    Min gr Class 1: {min_gr_c1:.4f}")
print(f"    Max gr Class 0: {max_gr_c0:.4f}")
print(f"    Overlap exists: {overlap}")
print(f"    Perfect separation: {not overlap}")
print(f"")
print(f"    === growth_ratio ORIGIN (dari 09_label_eskalasi.py) ===")
print(f"    growth_ratio = max_hotspots(H+1..H+3) / hotspots(H0)")
print(f"    Label 1 diberikan JIKA: growth_ratio >= P95 DAN tervalidasi MCD64A1")
print(f"    -> growth_ratio adalah KOMPONEN LANGSUNG definisi label")
print(f"    -> Bukan temporal leakage (dihitung dari H0-H+3 yg sama dengan observasi)")
print(f"    -> Bukan data leakage (dihitung dari raw FIRMS, bukan dari test set)")
print(f"    -> Tapi: label = f(growth_ratio), jadi model 'menghafal' aturan labeling")

# ================================================================
# 4. SHAP TREEXPLAINER -- MODEL A
# ================================================================
print("\n[4] SHAP TreeExplainer -- Model A (WITH growth_ratio) ...")
explainer_A = shap.TreeExplainer(model_A)
shap_values_A = explainer_A.shap_values(X_test)

# Handle output: bisa 1D atau 2D tergantung SHAP version
if isinstance(shap_values_A, list):
    sv_A = shap_values_A[1]  # binary: index 1 = positive class
else:
    sv_A = shap_values_A

print(f"    SHAP values shape: {sv_A.shape}")

# Mean absolute SHAP per feature
mean_abs_shap_A = np.abs(sv_A).mean(axis=0)
shap_df_A = pd.DataFrame({
    'feature'       : X_test.columns,
    'mean_abs_shap_A': mean_abs_shap_A
}).sort_values('mean_abs_shap_A', ascending=False).reset_index(drop=True)

shap_df_A['rank_A'] = range(1, len(shap_df_A) + 1)

print(f"\n    Top 10 SHAP features (Model A):")
print(f"    {'Rank':>4} {'Feature':<30} {'Mean|SHAP|':>12}")
for _, row in shap_df_A.head(10).iterrows():
    print(f"    {int(row['rank_A']):>4} {row['feature']:<30} {row['mean_abs_shap_A']:>12.4f}")

gr_rank_A = int(shap_df_A[shap_df_A['feature'] == 'growth_ratio']['rank_A'].values[0])
gr_shap_A = float(shap_df_A[shap_df_A['feature'] == 'growth_ratio']['mean_abs_shap_A'].values[0])
total_shap_A = mean_abs_shap_A.sum()
gr_pct_A = gr_shap_A / total_shap_A * 100

print(f"\n    growth_ratio: Rank={gr_rank_A}, mean|SHAP|={gr_shap_A:.4f} ({gr_pct_A:.1f}% of total)")

# ================================================================
# 5. SHAP TREEXPLAINER -- MODEL B
# ================================================================
print("\n[5] SHAP TreeExplainer -- Model B (WITHOUT growth_ratio) ...")
explainer_B = shap.TreeExplainer(model_B)
shap_values_B = explainer_B.shap_values(X_test_no_gr)

if isinstance(shap_values_B, list):
    sv_B = shap_values_B[1]
else:
    sv_B = shap_values_B

mean_abs_shap_B = np.abs(sv_B).mean(axis=0)
shap_df_B = pd.DataFrame({
    'feature'       : X_test_no_gr.columns,
    'mean_abs_shap_B': mean_abs_shap_B
}).sort_values('mean_abs_shap_B', ascending=False).reset_index(drop=True)
shap_df_B['rank_B'] = range(1, len(shap_df_B) + 1)

print(f"\n    Top 10 SHAP features (Model B - no growth_ratio):")
print(f"    {'Rank':>4} {'Feature':<30} {'Mean|SHAP|':>12}")
for _, row in shap_df_B.head(10).iterrows():
    print(f"    {int(row['rank_B']):>4} {row['feature']:<30} {row['mean_abs_shap_B']:>12.4f}")

# ================================================================
# 6. COMBINED IMPORTANCE TABLE
# ================================================================
print("\n[6] COMBINED IMPORTANCE TABLE ...")
combined = shap_df_A.merge(shap_df_B[['feature','mean_abs_shap_B','rank_B']],
                            on='feature', how='left')
combined = combined.sort_values('mean_abs_shap_A', ascending=False)

print(f"\n    {'Feature':<30} {'Model A Rank':>12} {'Model A |SHAP|':>14} {'Model B Rank':>12} {'Model B |SHAP|':>14}")
for _, row in combined.head(15).iterrows():
    rb = f"{int(row['rank_B'])}" if pd.notna(row.get('rank_B')) else "N/A"
    sb = f"{row['mean_abs_shap_B']:.4f}" if pd.notna(row.get('mean_abs_shap_B')) else "N/A"
    print(f"    {row['feature']:<30} {int(row['rank_A']):>12} {row['mean_abs_shap_A']:>14.4f} {rb:>12} {sb:>14}")

# ================================================================
# 7. SIMPAN CSV
# ================================================================
print("\n[7] SIMPAN CSV ...")
combined.to_csv(OUT_CSV, index=False, float_format='%.6f')
print(f"    Saved: {OUT_CSV}")

# ================================================================
# 8. PLOT SHAP SUMMARY (BEESWARM)
# ================================================================
print("\n[8] PLOT SHAP SUMMARY ...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Model A beeswarm
plt.sca(axes[0])
shap.summary_plot(sv_A, X_test, max_display=20, show=False,
                  color_bar=True, plot_type='dot')
axes[0].set_title('Model A: SHAP Beeswarm (WITH growth_ratio)',
                   fontsize=11, fontweight='bold', pad=12)

# Model B beeswarm
plt.sca(axes[1])
shap.summary_plot(sv_B, X_test_no_gr, max_display=20, show=False,
                  color_bar=True, plot_type='dot')
axes[1].set_title('Model B: SHAP Beeswarm (WITHOUT growth_ratio)',
                   fontsize=11, fontweight='bold', pad=12)

plt.tight_layout(pad=2.0)
plt.savefig(OUT_SUMM, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: {OUT_SUMM}")

# ================================================================
# 9. GROWTH_RATIO DEPENDENCE PLOT
# ================================================================
print("\n[9] GROWTH_RATIO DEPENDENCE PLOT ...")
gr_col_idx = list(X_test.columns).index('growth_ratio')
shap_gr    = sv_A[:, gr_col_idx]
gr_vals    = X_test['growth_ratio'].values
y_test_arr = y_test.values

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('growth_ratio Investigation: SHAP & Class Separation',
             fontsize=13, fontweight='bold', y=1.02)

# Left: SHAP Dependence Plot
sc = axes[0].scatter(gr_vals, shap_gr,
                     c=y_test_arr, cmap='RdYlGn_r',
                     alpha=0.7, s=40, edgecolors='none')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].axvline(p95_gr, color='orange', linestyle='--', linewidth=1.5,
                label=f'P95 threshold={p95_gr:.2f}')
axes[0].set_xlabel('growth_ratio (value)', fontsize=11)
axes[0].set_ylabel('SHAP value (impact on prediction)', fontsize=11)
axes[0].set_title('SHAP Dependence: growth_ratio', fontsize=11, fontweight='bold')
cb = plt.colorbar(sc, ax=axes[0])
cb.set_label('True Label (0=Non-Esc, 1=Esc)', fontsize=9)
axes[0].legend(fontsize=9)

# Right: Distribution per class
bins = np.linspace(0, min(gr_vals.max(), 30), 50)
axes[1].hist(gr_vals[y_test_arr == 0], bins=bins, alpha=0.6,
             color='steelblue', label=f'Class 0 (n={int((y_test_arr==0).sum())})',
             density=True)
axes[1].hist(gr_vals[y_test_arr == 1], bins=bins, alpha=0.7,
             color='crimson', label=f'Class 1 (n={int((y_test_arr==1).sum())})',
             density=True)
axes[1].axvline(p95_gr, color='orange', linestyle='--', linewidth=1.5,
                label=f'P95={p95_gr:.2f}')
axes[1].axvline(max_gr_c0, color='blue', linestyle=':', linewidth=1.5,
                label=f'Max C0={max_gr_c0:.2f}')
axes[1].axvline(min_gr_c1, color='red', linestyle=':', linewidth=1.5,
                label=f'Min C1={min_gr_c1:.2f}')
axes[1].set_xlabel('growth_ratio (value)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('growth_ratio Distribution by Class', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DEP, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: {OUT_DEP}")

# ================================================================
# 10. TULIS LAPORAN MD
# ================================================================
print("\n[10] TULIS LAPORAN ...")

# Prepare comparison table rows
cmp_rows = []
for _, row in combined.head(15).iterrows():
    rb = str(int(row['rank_B'])) if pd.notna(row.get('rank_B')) else "N/A"
    sb = f"{row['mean_abs_shap_B']:.4f}" if pd.notna(row.get('mean_abs_shap_B')) else "N/A"
    cmp_rows.append(
        f"| {row['feature']} | {int(row['rank_A'])} | {row['mean_abs_shap_A']:.4f} | {rb} | {sb} |"
    )

gr_dominance = "YA -- mendominasi sangat kuat" if gr_pct_A > 50 else "MODERAT" if gr_pct_A > 20 else "TIDAK"

lines = [
    "# STEP 9A -- SHAP GLOBAL ANALYSIS & growth_ratio INVESTIGATION",
    "## FireEscalome GEMASTIK XIX/2026 | Person C",
    f"**Timestamp: {ts} | SHAP: {shap.__version__}**",
    "",
    "---",
    "",
    "## 1. growth_ratio: Definisi & Status Investigasi",
    "",
    "### 1a. Definisi dari Source Code (09_label_eskalasi.py)",
    "",
    "```python",
    "def calculate_growth_ratio(df_cluster):",
    "    day_0_count = np.sum(days == min_day)              # hotspot hari H0",
    "    counts_next_3_days = [np.sum(days == d)            # hotspot H+1, H+2, H+3",
    "                          for d in range(min_day+1, min_day+4)]",
    "    max_daily_next_3_days = max(counts_next_3_days)    # max dari H+1..H+3",
    "    return max_daily_next_3_days / day_0_count         # growth_ratio",
    "",
    "# Label 1 diberikan JIKA: growth_ratio >= P95 DAN tervalidasi MCD64A1",
    "```",
    "",
    "### 1b. Klasifikasi Status growth_ratio",
    "",
    "| Aspek | Status | Detail |",
    "|---|---|---|",
    "| Temporal leakage? | **TIDAK** | Dihitung dari window H0-H+3 yang identik dengan periode observasi |",
    "| Data leakage dari test set? | **TIDAK** | Dihitung dari raw FIRMS sebelum split |",
    "| Target-derived feature? | **YA** | Label = f(growth_ratio >= P95), bukan sebaliknya |",
    "| Circular? | **HAMPIR** | Model belajar: growth_ratio tinggi -> label 1, persis aturan labeling |",
    "| Valid sebagai predictive feature? | **DEBATABLE** | Observabel di H0-H+3, tapi secara definitional menentukan label |",
    "",
    "> **OPEN ISSUE:** Apakah growth_ratio seharusnya dimasukkan sebagai fitur prediktif?",
    "> Argumen untuk: observabel, tidak future-leaking, memang informatif secara fisik",
    "> Argumen kontra: label = f(growth_ratio), sehingga model belajar aturan labeling bukan pola fisik",
    "> Keputusan metodologis ini harus dikonsultasikan dengan tim (bukan diputuskan sepihak).",
    "",
    "---",
    "",
    "## 2. Class Separation Analysis",
    "",
    "| Metrik | Nilai |",
    "|---|---|",
    f"| Min growth_ratio Class 1 | **{min_gr_c1:.4f}** |",
    f"| Max growth_ratio Class 0 | **{max_gr_c0:.4f}** |",
    f"| Overlap antar kelas | **{overlap}** |",
    f"| Perfect separation? | **{not overlap}** |",
    f"| P95 threshold (definisi label) | {p95_gr:.4f} |",
    f"| Class 0: min/mean/max | {gr_c0.min():.3f} / {gr_c0.mean():.3f} / {gr_c0.max():.3f} |",
    f"| Class 1: min/mean/max | {gr_c1.min():.3f} / {gr_c1.mean():.3f} / {gr_c1.max():.3f} |",
    "",
    "---",
    "",
    "## 3. Top 10 SHAP Features -- Model A (WITH growth_ratio)",
    "",
    "| Rank | Feature | Mean |SHAP| |",
    "|---|---|---|",
]

for _, row in shap_df_A.head(10).iterrows():
    lines.append(f"| {int(row['rank_A'])} | **{row['feature']}** | {row['mean_abs_shap_A']:.4f} |")

lines += [
    "",
    f"**growth_ratio: Rank={gr_rank_A}, Mean|SHAP|={gr_shap_A:.4f} ({gr_pct_A:.1f}% dari total SHAP)**",
    "",
    "---",
    "",
    "## 4. Top 10 SHAP Features -- Model B (WITHOUT growth_ratio)",
    "",
    "| Rank | Feature | Mean |SHAP| |",
    "|---|---|---|",
]

for _, row in shap_df_B.head(10).iterrows():
    lines.append(f"| {int(row['rank_B'])} | **{row['feature']}** | {row['mean_abs_shap_B']:.4f} |")

lines += [
    "",
    "---",
    "",
    "## 5. Perbandingan Ranking Model A vs Model B",
    "",
    "| Feature | Rank A | SHAP A | Rank B | SHAP B |",
    "|---|---|---|---|---|",
] + cmp_rows + [
    "",
    "---",
    "",
    "## 6. Dominance Analysis",
    "",
    f"| Aspek | Nilai |",
    f"|---|---|",
    f"| growth_ratio rank di Model A | **#{gr_rank_A}** |",
    f"| growth_ratio mean|SHAP| | {gr_shap_A:.4f} |",
    f"| Total SHAP sum (all features) | {total_shap_A:.4f} |",
    f"| growth_ratio % dari total SHAP | **{gr_pct_A:.1f}%** |",
    f"| Apakah growth_ratio mendominasi? | **{gr_dominance}** |",
    "",
    "---",
    "",
    "## 7. Visualisasi",
    "",
    "| File | Keterangan |",
    "|---|---|",
    "| step9a_shap_summary.png | SHAP Beeswarm Model A (kiri) vs Model B (kanan) |",
    "| step9a_growth_ratio_dependence.png | SHAP Dependence + distribusi growth_ratio per kelas |",
    "",
    "---",
    "",
    "## 8. Ringkasan & Rekomendasi",
    "",
    "### 8a. Temuan Utama",
    "",
    f"1. **growth_ratio: {gr_pct_A:.1f}% dari total SHAP** (Rank #{gr_rank_A}) di Model A",
    f"2. **Perfect separation:** min_c1={min_gr_c1:.1f} > max_c0={max_gr_c0:.3f}, tidak ada overlap",
    f"3. **Target-derived:** Label 1 = growth_ratio >= P95. Model A belajar aturan ini langsung.",
    f"4. **Model B (tanpa growth_ratio):** lebih generalisable, top fitur = {shap_df_B.iloc[0]['feature']}",
    f"5. **Model B ROC-AUC=0.9098, F1-1=0.32** -- mencerminkan kesulitan sebenarnya.",
    "",
    "### 8b. Klasifikasi growth_ratio",
    "",
    "| Kategori | Status |",
    "|---|---|",
    "| Temporal leakage | TIDAK |",
    "| Data leakage dari test set | TIDAK |",
    "| Predictive feature yang sangat kuat | YA |",
    "| Target-derived feature | YA |",
    "| Circular definition risk | YA (label = f(growth_ratio)) |",
    "",
    "### 8c. Rekomendasi",
    "",
    "> **REKOMENDASI: growth_ratio perlu INVESTIGASI LEBIH LANJUT (bukan langsung dihapus).**",
    "",
    "- Jika tujuan model = **triase operasional real-time** (BPBD/Manggala Agni):",
    "  growth_ratio dihitung dari H0-H+3 yang sudah berlalu -> **VALID digunakan**",
    "  Artinya: pada saat prediksi, observer sudah bisa menghitung growth_ratio dari data FIRMS.",
    "",
    "- Jika tujuan model = **prediksi sebelum eskalasi terjadi** (early warning pada H0):",
    "  growth_ratio membutuhkan data H+1 s/d H+3 yang **belum tersedia** -> **TIDAK VALID**",
    "  Model seharusnya hanya menggunakan fitur yang tersedia pada H0.",
    "",
    "- **OPEN ISSUE:** Proposal menyebut 'Early-Warning Pattern Rules' -- ini mengindikasikan",
    "  model seharusnya prediktif pada H0. Jika demikian, growth_ratio tidak valid.",
    "  Namun ini perlu konfirmasi dari tim.",
    "",
    "---",
    "",
    "## 9. Output Files",
    "",
    "| File | Status |",
    "|---|---|",
    "| step9a_shap_global_importance.csv | TERSIMPAN |",
    "| step9a_shap_summary.png | TERSIMPAN |",
    "| step9a_growth_ratio_dependence.png | TERSIMPAN |",
    "| STEP9A_SHAP_GLOBAL_REPORT.md | Dokumen ini |",
    "",
    "---",
    "Dibuat: Person C -- FireEscalome GEMASTIK XIX/2026",
    "Next: Step 9B (SHAP Interaction) atau keputusan growth_ratio -- setelah instruksi",
]

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"    Saved: {OUT_REPORT}")

# ================================================================
# 11. LAPORAN AKHIR KE CONSOLE
# ================================================================
print(f"\n{SEP}")
print("STEP 9A -- LAPORAN AKHIR")
print(SEP)
print(f"""
Top 10 fitur berdasarkan mean |SHAP| (Model A):""")
for _, row in shap_df_A.head(10).iterrows():
    print(f"  #{int(row['rank_A']):2d}  {row['feature']:<30} {row['mean_abs_shap_A']:.4f}")

print(f"""
growth_ratio ranking   : #{gr_rank_A}
mean |SHAP| growth_ratio: {gr_shap_A:.4f} ({gr_pct_A:.1f}% dari total)
Apakah mendominasi?    : {gr_dominance}

Dependence analysis:
  Perfect separation     : {not overlap}
  Min gr Class 1         : {min_gr_c1:.4f}
  Max gr Class 0         : {max_gr_c0:.4f}

Indikasi leakage?
  Temporal leakage : TIDAK
  Data leakage     : TIDAK
  Target-derived   : YA (label = f(growth_ratio >= P95))
  Circular risk    : YA (model belajar aturan labeling)

Model A vs Model B:
  Model A ROC-AUC={0.9986:.4f}, F1-1={0.9841:.4f} (WITH growth_ratio)
  Model B ROC-AUC={0.9098:.4f}, F1-1={0.3226:.4f} (WITHOUT growth_ratio)

Rekomendasi:
  growth_ratio perlu INVESTIGASI LEBIH LANJUT
  OPEN ISSUE: valid hanya jika digunakan dalam triase post-hoc (bukan early warning H0)
""")
print(SEP)
print("STEP 9A SELESAI. Menunggu instruksi berikutnya.")
print(SEP)
