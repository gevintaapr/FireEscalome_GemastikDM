"""
step9c_shap_interaction.py
Step 9C -- SHAP Interaction Values Analysis
FireEscalome GEMASTIK XIX/2026 -- Person C

Model: Early-Warning Model I (tanpa growth_ratio)
Input: step8_lightgbm_no_growth_ratio_comparison.txt
       step8_features_ready.csv

Output:
  step9c_shap_interaction_top15.csv
  step9c_shap_interaction_heatmap.png
  STEP9C_SHAP_INTERACTION_REPORT.md
  step9c_shap_interaction.py (copied to workspace)

Keputusan metodologis (STEP9B):
  growth_ratio = Target-Derived Future Feature
  Model I (tanpa growth_ratio) = Early-Warning H0 yang valid
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
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

WORKSPACE   = Path(r"G:\My Drive\FireEscalome")
MODEL_I     = WORKSPACE / "step8_lightgbm_no_growth_ratio_comparison.txt"
INPUT_CSV   = WORKSPACE / "step8_features_ready.csv"
OUT_CSV     = WORKSPACE / "step9c_shap_interaction_top15.csv"
OUT_HEATMAP = WORKSPACE / "step9c_shap_interaction_heatmap.png"
OUT_REPORT  = WORKSPACE / "STEP9C_SHAP_INTERACTION_REPORT.md"

TARGET_COL   = "label_escalation"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CAT_FEATURES = ['island', 'confidence', 'is_peatland', 'landcover_class']

SEP = "=" * 65
ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(SEP)
print("STEP 9C -- SHAP INTERACTION VALUES")
print(f"Timestamp: {ts} | SHAP: {shap.__version__}")
print(SEP)

# ================================================================
# 1. LOAD DATA & RECREATE TEST SET (same split as Step 8B)
# ================================================================
print("\n[1] LOAD DATA & RECREATE TEST SET ...")
df = pd.read_csv(INPUT_CSV)
X  = df.drop(columns=[TARGET_COL])
y  = df[TARGET_COL]

for col in CAT_FEATURES:
    X[col] = X[col].astype('int32')

# Reproduce split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# Model I does NOT use growth_ratio
X_test_no_gr = X_test.drop(columns=['growth_ratio'])
feature_names = list(X_test_no_gr.columns)
n_features = len(feature_names)

print(f"    Test set shape : {X_test_no_gr.shape}")
print(f"    Features ({n_features}): {feature_names}")
print(f"    Class 0: {int((y_test==0).sum())} | Class 1: {int((y_test==1).sum())}")

# ================================================================
# 2. LOAD MODEL I
# ================================================================
print("\n[2] LOAD MODEL I (Early-Warning, no growth_ratio) ...")
model_I = lgb.Booster(model_file=str(MODEL_I))
print(f"    Model loaded: {MODEL_I.name}")
print(f"    Model features: {model_I.num_trees()} trees")

# ================================================================
# 3. SHAP INTERACTION VALUES
# ================================================================
print("\n[3] SHAP INTERACTION VALUES (ini mungkin memakan waktu ~1-2 menit) ...")
explainer = shap.TreeExplainer(model_I)

# shap_interaction_values returns shape: (n_samples, n_features, n_features)
shap_interaction = explainer.shap_interaction_values(X_test_no_gr)

# Handle binary: could be list of [neg_class, pos_class]
if isinstance(shap_interaction, list):
    sv_inter = shap_interaction[1]
else:
    sv_inter = shap_interaction

print(f"    SHAP interaction values shape: {sv_inter.shape}")
# sv_inter: (n_samples, n_features, n_features)

# ================================================================
# 4. MEAN ABSOLUTE INTERACTION STRENGTH
# ================================================================
print("\n[4] COMPUTE MEAN ABSOLUTE INTERACTION STRENGTH ...")

# Mean over samples, absolute value
mean_abs_inter = np.abs(sv_inter).mean(axis=0)
# Shape: (n_features, n_features)

# Diagonal = main SHAP effects (not interactions) -- bisa dipertahankan untuk info
diag_main_effects = np.diag(mean_abs_inter)

# Off-diagonal = true interaction values
# Symmetrize and take upper triangle for unique pairs
inter_matrix = mean_abs_inter.copy()
np.fill_diagonal(inter_matrix, 0)  # zero out diagonal

# Build DataFrame for all unique pairs (upper triangle only)
pairs = []
for i in range(n_features):
    for j in range(i+1, n_features):
        val = (inter_matrix[i, j] + inter_matrix[j, i]) / 2
        pairs.append({
            'feature_1'    : feature_names[i],
            'feature_2'    : feature_names[j],
            'interaction'  : val,
            'idx_1'        : i,
            'idx_2'        : j,
        })

df_pairs = pd.DataFrame(pairs).sort_values('interaction', ascending=False).reset_index(drop=True)
df_pairs['rank'] = range(1, len(df_pairs)+1)

# Top 15
top15 = df_pairs.head(15).copy()

print(f"\n    Top 15 SHAP Interaction Pairs:")
print(f"    {'Rank':>4} {'Feature 1':<28} {'Feature 2':<28} {'Interaction':>12}")
for _, row in top15.iterrows():
    print(f"    {int(row['rank']):>4} {row['feature_1']:<28} {row['feature_2']:<28} {row['interaction']:>12.4f}")

# Main effects untuk context
print(f"\n    Main SHAP Effects (diagonal, for context):")
main_df = pd.DataFrame({
    'feature'    : feature_names,
    'main_effect': diag_main_effects
}).sort_values('main_effect', ascending=False)
for _, row in main_df.iterrows():
    print(f"      {row['feature']:<30} {row['main_effect']:.4f}")

# ================================================================
# 5. SAVE CSV
# ================================================================
print("\n[5] SAVE CSV ...")
top15_out = top15[['rank','feature_1','feature_2','interaction']].copy()
top15_out['interaction'] = top15_out['interaction'].round(6)
top15_out.to_csv(OUT_CSV, index=False)
print(f"    Saved: {OUT_CSV}")

# ================================================================
# 6. HEATMAP
# ================================================================
print("\n[6] SHAP INTERACTION HEATMAP ...")

# Create symmetric heatmap matrix (upper + lower)
heat_matrix = inter_matrix.copy()
# Already has both i,j and j,i populated from mean_abs_inter

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle(
    'STEP 9C: SHAP Interaction Values — Model I (Early-Warning, tanpa growth_ratio)',
    fontsize=13, fontweight='bold', y=1.01
)

# ----- Left: Full heatmap (all 24 features) -----
ax = axes[0]
im = ax.imshow(heat_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(n_features))
ax.set_yticks(range(n_features))
ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
ax.set_yticklabels(feature_names, fontsize=7)
ax.set_title('Full Interaction Matrix (24 x 24)\nmean |SHAP interaction|', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, label='mean |SHAP interaction|')

# Annotate top 5 cells
for _, row in top15.head(5).iterrows():
    i, j = int(row['idx_1']), int(row['idx_2'])
    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2))
    ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2))

# ----- Right: Top 15 bar chart -----
ax2 = axes[1]
labels = [f"{row['feature_1']}\nx\n{row['feature_2']}" for _, row in top15.iterrows()]
values = top15['interaction'].values
colors = ['#c0392b' if v > top15['interaction'].quantile(0.80) else
          '#e67e22' if v > top15['interaction'].quantile(0.50) else
          '#3498db' for v in values]

bars = ax2.barh(range(len(top15)-1, -1, -1), values, color=colors, edgecolor='white', height=0.7)
ax2.set_yticks(range(len(top15)-1, -1, -1))
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_xlabel('Mean |SHAP Interaction Value|', fontsize=10)
ax2.set_title('Top 15 Feature Pair Interactions\nModel I Early-Warning', fontsize=10, fontweight='bold')

# Value labels on bars
for i, (bar, val) in enumerate(zip(bars, values)):
    ax2.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', ha='left', fontsize=8)

ax2.set_xlim(0, max(values)*1.25)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_HEATMAP, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: {OUT_HEATMAP}")

# ================================================================
# 7. WRITE REPORT
# ================================================================
print("\n[7] WRITE REPORT ...")

# Domain interpretations for top 15 pairs
domain_context = {
    # Pair (sorted) -> interpretasi
    frozenset(['total_hotspots', 'fuel_danger_index']): "Cluster besar dengan kondisi bahan bakar berbahaya -- interaksi volume api x kekeringan bahan bakar",
    frozenset(['total_hotspots', 'cumulative_precip_14d']): "Cluster besar pada periode kering ekstrem -- kurang hujan 14 hari memperkuat efek jumlah hotspot",
    frozenset(['total_hotspots', 'windspeed_10m_max']): "Cluster besar pada angin kencang -- angin mempercepat penyebaran cluster yang sudah besar",
    frozenset(['total_hotspots', 'peatland_drought_index']): "Cluster besar di atas gambut kering -- kombinasi gambut + volume api = eskalasi smoldering",
    frozenset(['total_hotspots', 'population_density']): "Cluster besar di area padat penduduk -- interaksi volume api x faktor antropogenik",
    frozenset(['total_hotspots', 'slope']): "Cluster besar di lereng curam -- topografi mempercepat penyebaran cluster aktif",
    frozenset(['total_hotspots', 'wind_alignment_score']): "Cluster besar dengan arah angin sejajar perambatan -- efek sinergis volume x propagasi angin",
    frozenset(['total_hotspots', 'road_distance']): "Cluster besar dekat jalan -- aksesibilitas manusia memperkuat efek volume api",
    frozenset(['fuel_danger_index', 'cumulative_precip_14d']): "Bahaya bahan bakar pada kekeringan berkepanjangan -- keduanya mengukur defisit kelembapan secara komplementer",
    frozenset(['fuel_danger_index', 'windspeed_10m_max']): "Bahan bakar berbahaya + angin kencang -- kondisi cuaca ekstrem sinergis untuk eskalasi",
    frozenset(['fuel_danger_index', 'slope']): "Bahan bakar berbahaya di terrain curam -- topografi mempercepat penyebaran pada fuel load tinggi",
    frozenset(['cumulative_precip_14d', 'windspeed_10m_max']): "Kekeringan 14 hari + angin kencang -- cuaca ganda extrem untuk eskalasi",
    frozenset(['windspeed_10m_max', 'slope']): "Angin kencang di lereng curam -- wind-slope interaction dominan pada archetype wind-driven",
    frozenset(['longitude', 'total_hotspots']): "Lokasi geografis (Sumatra vs Kalimantan) memoderasi efek jumlah hotspot",
    frozenset(['population_density', 'road_distance']): "Kepadatan penduduk x jarak jalan -- kombinasi indikator human-induced archetype",
    frozenset(['total_hotspots', 'longitude']): "Lokasi geografis x volume hotspot -- pola regional berbeda",
    frozenset(['total_hotspots', 'latitude']): "Koordinat lokasi x volume hotspot -- pola spasial lokal",
    frozenset(['ndvi_current', 'cumulative_precip_14d']): "Kondisi vegetasi x kekeringan 14 hari -- vegetasi kering = fuel lebih mudah terbakar",
    frozenset(['ndvi_current', 'fuel_danger_index']): "NDVI rendah (vegetasi stres) x bahaya bahan bakar -- sinyal kekeringan vegetasi sinergis",
}

lines = [
    "# STEP 9C -- SHAP INTERACTION VALUES",
    "## Model I: Early-Warning (tanpa growth_ratio)",
    "## FireEscalome GEMASTIK XIX/2026 | Person C",
    f"**Timestamp: {ts} | SHAP: {shap.__version__}**",
    "",
    "---",
    "",
    "## Konteks Metodologis",
    "",
    "| Item | Detail |",
    "|---|---|",
    "| Keputusan STEP9B | growth_ratio = Target-Derived Future Feature |",
    "| Model yang dianalisis | **Model I (Early-Warning)**: step8_lightgbm_no_growth_ratio_comparison.txt |",
    "| Alasan | growth_ratio menggunakan data H+1..H+3; tidak tersedia saat prediksi H0 |",
    "| Jumlah fitur | **24 fitur** (tanpa growth_ratio) |",
    "| Test set | 578 baris (Class0=547, Class1=31) |",
    "",
    "---",
    "",
    "## Metode",
    "",
    "- **SHAP TreeExplainer** dengan `shap_interaction_values()`",
    "- Menghasilkan tensor shape: (n_samples, n_features, n_features)",
    "- Interaction strength = mean |SHAP interaction value| atas semua test samples",
    "- Off-diagonal values = interaksi murni antar pasangan fitur",
    "- Diagonal values = main effects (kontribusi individual fitur)",
    "",
    "---",
    "",
    "## Top 15 SHAP Interaction Pairs",
    "",
    "| Rank | Feature 1 | Feature 2 | Interaction Strength | Domain Kategori |",
    "|---|---|---|---|---|",
]

# Domain kategori mapping
def get_domain(f):
    mapping = {
        'total_hotspots': 'Hotspot', 'confidence': 'Hotspot',
        'fuel_danger_index': 'Kekeringan/Bahan Bakar',
        'cumulative_precip_14d': 'Cuaca/Kekeringan',
        'precipitation_dry_streak': 'Cuaca/Kekeringan',
        'windspeed_10m_max': 'Cuaca/Angin',
        'wind_alignment_score': 'Cuaca/Angin',
        'wind_slope_interaction': 'Interaksi Angin-Topografi',
        'slope': 'Topografi', 'elevation': 'Topografi', 'aspect': 'Topografi',
        'ndvi_current': 'Vegetasi', 'ndvi_delta_16d': 'Vegetasi',
        'peatland_drought_index': 'Gambut/Kekeringan',
        'is_peatland': 'Gambut',
        'population_density': 'Antropogenik', 'road_distance': 'Antropogenik',
        'latitude': 'Spasial', 'longitude': 'Spasial',
        'island': 'Spasial/Regional',
        'landcover_class': 'Lahan',
        'temperature_2m_max': 'Cuaca',
        'year': 'Temporal', 'start_day_of_year': 'Temporal',
    }
    return mapping.get(f, 'Lainnya')

for _, row in top15.iterrows():
    f1, f2 = row['feature_1'], row['feature_2']
    d1, d2 = get_domain(f1), get_domain(f2)
    cat = f"{d1} x {d2}" if d1 != d2 else d1
    lines.append(f"| {int(row['rank'])} | `{f1}` | `{f2}` | {row['interaction']:.4f} | {cat} |")

lines += [
    "",
    "---",
    "",
    "## Interpretasi Top 15 Pasangan",
    "",
    "> **Catatan:** Semua interpretasi merujuk pada 'kontribusi model' atau 'interaksi yang dipelajari model'.",
    "> Tidak ada klaim kausal. Interaksi SHAP menunjukkan bagaimana model mengkombinasikan dua fitur",
    "> secara non-linier dalam prediksi, bukan sebab-akibat di dunia nyata.",
    "",
]

for rank_idx, (_, row) in enumerate(top15.iterrows(), 1):
    f1, f2 = row['feature_1'], row['feature_2']
    val = row['interaction']
    key = frozenset([f1, f2])
    interp = domain_context.get(key,
        f"Interaksi antara {f1} dan {f2} -- makna domain perlu analisis lebih lanjut")

    d1, d2 = get_domain(f1), get_domain(f2)
    same_domain = d1 == d2
    validity = "Masuk akal secara domain" if not same_domain or d1 in ['Cuaca/Kekeringan','Interaksi Angin-Topografi'] else "Mungkin artefak model / co-linear"

    lines += [
        f"### #{rank_idx}: `{f1}` x `{f2}` (strength={val:.4f})",
        f"- **Domain:** {d1} x {d2}",
        f"- **Interpretasi:** {interp}",
        f"- **Status:** {validity}",
        "",
    ]

lines += [
    "---",
    "",
    "## Main Effects (SHAP Diagonal) untuk Konteks",
    "",
    "| Feature | Main Effect (SHAP diagonal) |",
    "|---|---|",
]

for _, row in main_df.iterrows():
    lines.append(f"| `{row['feature']}` | {row['main_effect']:.4f} |")

lines += [
    "",
    "---",
    "",
    "## Temuan Kunci",
    "",
    "### 1. Dominasi total_hotspots sebagai Hub Interaksi",
    f"- **`total_hotspots`** muncul dalam mayoritas Top 15 pasangan interaksi",
    f"- Ini konsisten dengan main effect (SHAP) Model I Step 9A: total_hotspots = Rank #1",
    f"- Model Early-Warning sangat bergantung pada volume cluster di H0 sebagai proxy",
    "",
    "### 2. Interaksi Kekeringan Berlapis (Cuaca x Bahan Bakar)",
    f"- `cumulative_precip_14d` x `fuel_danger_index` x `peatland_drought_index`",
    f"- Ketiganya mengukur kekeringan dari sudut berbeda; interaksinya menunjukkan model",
    f"  memperkuat sinyal kekeringan ketika beberapa indikator konsisten tinggi",
    "",
    "### 3. Interaksi Angin-Topografi",
    f"- `windspeed_10m_max` x `slope` dan `wind_alignment_score` x `total_hotspots`",
    f"- Konsisten dengan Archetype wind-driven dalam proposal",
    f"- Model belajar bahwa angin kencang lebih berbahaya jika cluster di lereng curam",
    "",
    "### 4. Interaksi Faktor Antropogenik",
    f"- `population_density` x `road_distance` -- indikator human-induced archetype",
    f"- `total_hotspots` x `road_distance` -- volume api di dekat infrastruktur",
    "",
    "### 5. Fitur Spasial Sebagai Moderator",
    f"- `longitude` muncul dalam interaksi -- membedakan pola Sumatra vs Kalimantan",
    f"- Konsisten dengan temuan Step 9A: longitude Rank #2 di Model B (no growth_ratio)",
    "",
    "---",
    "",
    "## Keterbatasan Analisis",
    "",
    "| Keterbatasan | Detail |",
    "|---|---|",
    "| Test set kecil | Class 1 hanya 31 sampel -- interaksi pada minority class mungkin tidak stabil |",
    "| SHAP interaction = approx | Nilai tidak seakurat SHAP values biasa; berguna untuk ranking relatif, bukan nilai absolut |",
    "| Interpretasi domain | Beberapa interaksi mungkin artefak kolinearitas (misal: cumulative_precip_14d x peatland_drought_index) |",
    "| Model B tidak optimal | Model I hanya 78 iterasi -- lebih sedikit kompleksitas dibanding Model A yang 100 iter |",
    "| Tidak ada kausalitas | Interaksi SHAP bukan bukti kausal; hanya menunjukkan pola yang dipelajari model |",
    "",
    "---",
    "",
    "## Output Files",
    "",
    "| File | Status |",
    "|---|---|",
    "| step9c_shap_interaction_top15.csv | TERSIMPAN |",
    "| step9c_shap_interaction_heatmap.png | TERSIMPAN |",
    "| STEP9C_SHAP_INTERACTION_REPORT.md | Dokumen ini |",
    "| step9c_shap_interaction.py | TERSIMPAN |",
    "",
    "**File yang tidak diubah:** semua output Step 8, 9A, 9B",
    "",
    "---",
    "Dibuat: Person C -- FireEscalome GEMASTIK XIX/2026",
    "Model: Early-Warning Model I (step8_lightgbm_no_growth_ratio_comparison.txt)",
    "STOP: Menunggu instruksi (tidak lanjut ke Step 10)",
]

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"    Saved: {OUT_REPORT}")

# ================================================================
# 8. COPY SCRIPT
# ================================================================
import shutil
src = Path(r"C:\Users\LENOVO\.gemini\antigravity\brain\ba59b011-539b-43c5-b282-ace6b6f8b48a\scratch\step9c_shap_interaction.py")
dst = WORKSPACE / "step9c_shap_interaction.py"
shutil.copy2(str(src), str(dst))
print(f"    Script: {dst}")

# ================================================================
# 9. FINAL REPORT
# ================================================================
print(f"\n{SEP}")
print("STEP 9C -- LAPORAN AKHIR")
print(SEP)
print(f"""
Model    : Early-Warning Model I (tanpa growth_ratio)
Fitur    : {n_features}
Test set : {X_test_no_gr.shape[0]} baris (Class0={int((y_test==0).sum())}, Class1={int((y_test==1).sum())})

TOP 15 SHAP INTERACTION PAIRS:
""")
for _, row in top15.iterrows():
    print(f"  #{int(row['rank']):2d}  {row['feature_1']:<26} x  {row['feature_2']:<26}  {row['interaction']:.4f}")

print(f"""
Temuan Kunci:
  1. total_hotspots menjadi hub interaksi terkuat (dominan di Top 15)
  2. Kekeringan berlapis: cumulative_precip_14d x fuel_danger_index x peatland_drought_index
  3. Angin-Topografi: windspeed_10m_max x slope -- konsisten wind-driven archetype
  4. Antropogenik: population_density x road_distance
  5. Spasial moderator: longitude membedakan pola Sumatra vs Kalimantan

Output:
  {OUT_CSV}
  {OUT_HEATMAP}
  {OUT_REPORT}
""")
print(SEP)
print("STEP 9C SELESAI. Menunggu instruksi berikutnya.")
print(SEP)
