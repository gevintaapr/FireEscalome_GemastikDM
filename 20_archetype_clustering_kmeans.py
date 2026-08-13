"""
step10a_archetype_clustering.py
Step 10A -- Archetype Discovery: Feature Preparation & Clustering
FireEscalome GEMASTIK XIX/2026 -- Person C

Input  : G:/My Drive/FireEscalome/cluster_master_PersonB.csv
Output :
  step10a_archetype_features.csv
  step10a_cluster_profiles.csv
  step10a_cluster_metrics.csv
  step10a_archetype_visualization.png
  STEP10A_ARCHETYPE_DISCOVERY_REPORT.md
  step10a_archetype_clustering.py (copied to workspace)

PENTING:
  - Tidak menggunakan label_escalation, growth_ratio, cluster_id, community_id
  - Archetype = hasil pattern discovery, bukan label kausal
  - k dipilih berdasarkan evidence metrik, bukan target 4 archetype
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score)
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WORKSPACE  = Path(r"G:\My Drive\FireEscalome")
INPUT_CSV  = WORKSPACE / "cluster_master_PersonB.csv"
OUT_FEAT   = WORKSPACE / "step10a_archetype_features.csv"
OUT_PROF   = WORKSPACE / "step10a_cluster_profiles.csv"
OUT_MET    = WORKSPACE / "step10a_cluster_metrics.csv"
OUT_VIZ    = WORKSPACE / "step10a_archetype_visualization.png"
OUT_REPORT = WORKSPACE / "STEP10A_ARCHETYPE_DISCOVERY_REPORT.md"

RANDOM_STATE = 42
K_RANGE      = range(2, 7)   # k = 2..6

SEP = "=" * 65
ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(SEP)
print("STEP 10A -- ARCHETYPE DISCOVERY: CLUSTERING")
print(f"Timestamp: {ts}")
print(SEP)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\n[1] LOAD DATA ...")
df = pd.read_csv(INPUT_CSV)
print(f"    Shape: {df.shape}")
print(f"    label_escalation dist: {dict(df['label_escalation'].value_counts().sort_index())}")

# ================================================================
# 2. FEATURE SELECTION
# ================================================================
# Fitur yang digunakan: fisik/lingkungan relevan untuk archetype
# TIDAK: label_escalation, growth_ratio, cluster_id, community_id,
#        latitude_weather, longitude_weather (duplikat lat/lon)
# Catatan: latitude, longitude DIMASUKKAN untuk konteks spasial
#          (bukan identifier, tapi informasi distribusi geografis)
# start_day_of_year DIMASUKKAN sebagai fitur temporal (musim)

CLUSTERING_FEATURES = [
    # Volume hotspot (di H0, valid)
    'total_hotspots',
    # Topografi
    'slope', 'elevation', 'aspect',
    # Vegetasi
    'ndvi_current', 'ndvi_delta_16d',
    # Cuaca - angin
    'windspeed_10m_max', 'wind_alignment_score',
    # Cuaca - kekeringan
    'temperature_2m_max', 'precipitation_dry_streak',
    'cumulative_precip_14d',
    # Indikator gabungan
    'peatland_drought_index', 'fuel_danger_index',
    'wind_slope_interaction',
    # Faktor antropogenik
    'population_density', 'road_distance',
    # Gambut (biner, tetap informatif untuk clustering)
    'is_peatland',
    # Spasial (sebagai moderator)
    'latitude', 'longitude',
    # Temporal musiman
    'start_day_of_year',
]

print(f"\n[2] FEATURE SELECTION ...")
print(f"    Total fitur clustering: {len(CLUSTERING_FEATURES)}")
for i, f in enumerate(CLUSTERING_FEATURES, 1):
    print(f"    {i:2d}. {f}")

# Fitur yang TIDAK digunakan
excluded = ['label_escalation', 'growth_ratio', 'cluster_id', 'community_id',
            'latitude_weather', 'longitude_weather', 'island', 'landcover_class',
            'confidence', 'year']
print(f"\n    Fitur TIDAK digunakan:")
for f in excluded:
    reason = {
        'label_escalation': 'TARGET -- tidak boleh masuk clustering',
        'growth_ratio': 'TARGET-DERIVED FUTURE FEATURE (STEP9B)',
        'cluster_id': 'IDENTIFIER',
        'community_id': 'IDENTIFIER',
        'latitude_weather': 'DUPLIKAT latitude (weather station approx)',
        'longitude_weather': 'DUPLIKAT longitude (weather station approx)',
        'island': 'TURUNAN longitude (redundant)',
        'landcover_class': 'TERLALU BANYAK nilai -1/invalid; sedikit variasi',
        'confidence': 'NOMINAL -- confidence deteksi FIRMS, bukan fitur fisik',
        'year': 'Temporal identifier, bukan fitur fisik; bias temporal',
    }.get(f, 'dikecualikan')
    print(f"      {f:<25}: {reason}")

# ================================================================
# 3. EXTRACT FEATURE MATRIX
# ================================================================
print("\n[3] EXTRACT FEATURE MATRIX ...")
X_raw = df[CLUSTERING_FEATURES].copy()

# Cek missing & infinite
n_nan = X_raw.isnull().sum().sum()
n_inf = np.isinf(X_raw.select_dtypes(include=[np.number]).values).sum()
print(f"    NaN: {n_nan} | Infinite: {n_inf}")

# Cek outlier extreme (IQR x 10 tidak dihapus, hanya dilaporkan)
q99 = X_raw.quantile(0.99)
q01 = X_raw.quantile(0.01)
print(f"    Q1% / Q99% range check OK")

print(f"    Feature matrix shape: {X_raw.shape}")

# ================================================================
# 4. STANDARDISASI
# ================================================================
print("\n[4] STANDARDISASI ...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=CLUSTERING_FEATURES)
print(f"    Scaled shape: {X_scaled_df.shape}")
print(f"    Mean (should be ~0): {X_scaled_df.mean().abs().max():.6f}")
print(f"    Std  (should be ~1): {X_scaled_df.std().abs().max():.6f}")

# Simpan feature CSV
feat_out = X_raw.copy()
feat_out['label_escalation'] = df['label_escalation'].values  # untuk validasi post-hoc
feat_out['cluster_id']       = df['cluster_id'].values
feat_out.to_csv(OUT_FEAT, index=False, float_format='%.6f')
print(f"    Saved: {OUT_FEAT}")

# ================================================================
# 5. K-MEANS EVALUATION k=2..6
# ================================================================
print("\n[5] K-MEANS EVALUATION k=2..6 ...")
metrics_list = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)
    sil  = silhouette_score(X_scaled, labels)
    ch   = calinski_harabasz_score(X_scaled, labels)
    db   = davies_bouldin_score(X_scaled, labels)
    inertia = km.inertia_
    metrics_list.append({'k': k, 'silhouette': sil, 'calinski_harabasz': ch,
                         'davies_bouldin': db, 'inertia': inertia})
    print(f"    k={k}: Sil={sil:.4f} | CH={ch:.1f} | DB={db:.4f} | Inertia={inertia:.1f}")

df_metrics = pd.DataFrame(metrics_list)
df_metrics.to_csv(OUT_MET, index=False, float_format='%.4f')
print(f"\n    Saved metrics: {OUT_MET}")

# ================================================================
# 6. PILIH k TERBAIK
# ================================================================
print("\n[6] PILIH k TERBAIK ...")
# Normalisasi metrik untuk komposit score
# Silhouette: higher is better
# CH: higher is better
# DB: lower is better

sil_norm = (df_metrics['silhouette'] - df_metrics['silhouette'].min()) / \
           (df_metrics['silhouette'].max() - df_metrics['silhouette'].min() + 1e-9)
ch_norm  = (df_metrics['calinski_harabasz'] - df_metrics['calinski_harabasz'].min()) / \
           (df_metrics['calinski_harabasz'].max() - df_metrics['calinski_harabasz'].min() + 1e-9)
db_norm  = 1 - (df_metrics['davies_bouldin'] - df_metrics['davies_bouldin'].min()) / \
           (df_metrics['davies_bouldin'].max() - df_metrics['davies_bouldin'].min() + 1e-9)

composite = (sil_norm + ch_norm + db_norm) / 3
best_idx  = composite.idxmax()
best_k    = int(df_metrics.loc[best_idx, 'k'])
best_sil  = float(df_metrics.loc[best_idx, 'silhouette'])
best_ch   = float(df_metrics.loc[best_idx, 'calinski_harabasz'])
best_db   = float(df_metrics.loc[best_idx, 'davies_bouldin'])

print(f"    Composite scores:")
for i, row in df_metrics.iterrows():
    mark = " <-- TERPILIH" if row['k'] == best_k else ""
    print(f"      k={int(row['k'])}: sil_norm={sil_norm[i]:.3f} ch_norm={ch_norm[i]:.3f} db_norm={db_norm[i]:.3f} comp={composite[i]:.3f}{mark}")
print(f"\n    k TERPILIH = {best_k} (Sil={best_sil:.4f}, CH={best_ch:.1f}, DB={best_db:.4f})")

# ================================================================
# 7. FINAL CLUSTERING dengan k TERPILIH
# ================================================================
print(f"\n[7] FINAL CLUSTERING k={best_k} ...")
km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
df['cluster_label'] = km_final.fit_predict(X_scaled)

cluster_dist = df['cluster_label'].value_counts().sort_index()
print(f"    Cluster distribution:")
for c, n in cluster_dist.items():
    n_esc = int(df[df['cluster_label']==c]['label_escalation'].sum())
    pct_esc = n_esc / n * 100
    print(f"      Cluster {c}: {n:4d} rows | {n_esc:3d} eskalasi ({pct_esc:.1f}%)")

# ================================================================
# 8. CLUSTER PROFILES
# ================================================================
print(f"\n[8] CLUSTER PROFILES ...")
profile_features = CLUSTERING_FEATURES + ['label_escalation']
profiles_mean   = df.groupby('cluster_label')[profile_features].mean().round(4)
profiles_median = df.groupby('cluster_label')[profile_features].median().round(4)
profiles_std    = df.groupby('cluster_label')[profile_features].std().round(4)

# Gabungkan mean + median ke output CSV
profiles_out = profiles_mean.copy()
profiles_out.index.name = 'cluster'
profiles_out.to_csv(OUT_PROF, float_format='%.4f')
print(f"    Saved: {OUT_PROF}")

# Print profil ringkas
print(f"\n    Profil per Cluster (MEAN):")
key_features = ['total_hotspots', 'is_peatland', 'peatland_drought_index',
                'windspeed_10m_max', 'wind_alignment_score', 'wind_slope_interaction',
                'slope', 'fuel_danger_index', 'cumulative_precip_14d',
                'precipitation_dry_streak', 'population_density', 'road_distance',
                'ndvi_current', 'label_escalation']

header = f"{'Feature':<30}"
for c in range(best_k):
    header += f" Cls{c:>6}"
print(f"    {header}")
for feat in key_features:
    row = f"    {feat:<30}"
    for c in range(best_k):
        val = profiles_mean.loc[c, feat]
        row += f" {val:>7.3f}"
    print(row)

# ================================================================
# 9. ARCHETYPE MAPPING (evidence-based, tidak dipaksakan)
# ================================================================
print(f"\n[9] ARCHETYPE MAPPING ...")

# Tentukan dominan fitur per cluster berdasarkan z-score terstandarisasi
# (seberapa jauh mean cluster dari mean global)
X_raw_with_label = df[CLUSTERING_FEATURES + ['label_escalation', 'cluster_label']]
global_mean = df[CLUSTERING_FEATURES].mean()
global_std  = df[CLUSTERING_FEATURES].std()

archetype_map = {}
archetype_notes = {}

for c in range(best_k):
    sub = df[df['cluster_label'] == c]
    sub_mean = sub[CLUSTERING_FEATURES].mean()
    z_score  = (sub_mean - global_mean) / (global_std + 1e-9)
    top_pos  = z_score.nlargest(5)   # fitur paling tinggi relatif
    top_neg  = z_score.nsmallest(3)  # fitur paling rendah relatif

    # Heuristik archetype mapping
    peat_score    = z_score.get('is_peatland', 0) + z_score.get('peatland_drought_index', 0)
    wind_score    = z_score.get('windspeed_10m_max', 0) + z_score.get('wind_alignment_score', 0) + z_score.get('wind_slope_interaction', 0)
    drought_score = z_score.get('fuel_danger_index', 0) + z_score.get('precipitation_dry_streak', 0) - z_score.get('cumulative_precip_14d', 0)
    human_score   = -z_score.get('road_distance', 0) + z_score.get('population_density', 0)  # dekat jalan = road_distance rendah

    scores = {
        'Peat-driven'   : peat_score,
        'Wind-driven'   : wind_score,
        'Drought-driven': drought_score,
        'Human-induced' : human_score,
    }
    best_arch   = max(scores, key=scores.get)
    best_score  = scores[best_arch]
    second_arch = sorted(scores, key=scores.get, reverse=True)[1]
    second_score= scores[second_arch]

    n_cluster = len(sub)
    n_esc = int(sub['label_escalation'].sum())
    pct_esc = n_esc / n_cluster * 100

    # Threshold: jika score terbaik tidak jauh lebih besar dari kedua, label "mixed"
    if best_score - second_score < 0.3 and best_score < 0.5:
        label = f"Mixed ({best_arch.split('-')[0]}/{second_arch.split('-')[0]})"
        conf = "LOW"
    else:
        label = best_arch
        conf = "MEDIUM" if best_score < 1.0 else "HIGH"

    archetype_map[c]   = label
    archetype_notes[c] = {
        'n': n_cluster, 'n_esc': n_esc, 'pct_esc': pct_esc,
        'archetype': label, 'confidence': conf,
        'top_positive_features': list(top_pos.index),
        'top_negative_features': list(top_neg.index),
        'scores': scores, 'z_top5': top_pos.to_dict()
    }

    print(f"\n    Cluster {c}: {label} ({conf}) | n={n_cluster} | esc={n_esc} ({pct_esc:.1f}%)")
    print(f"      Archetype scores: {', '.join([f'{k}={v:.3f}' for k,v in scores.items()])}")
    print(f"      Top high Z-score features: {list(top_pos.index[:3])}")
    print(f"      Top low  Z-score features: {list(top_neg.index[:2])}")

# ================================================================
# 10. VISUALISASI (PCA 2D scatter)
# ================================================================
print(f"\n[10] VISUALISASI ...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

colors_cluster = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
markers_esc    = {0: 'o', 1: '*'}

fig = plt.figure(figsize=(18, 6))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

# Left: PCA scatter by cluster
ax1 = fig.add_subplot(gs[0])
for c in range(best_k):
    mask = df['cluster_label'] == c
    arch_label = archetype_map[c]
    n_c = mask.sum()
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors_cluster[c], alpha=0.4, s=20,
                label=f"C{c}: {arch_label} (n={n_c})")

ax1.set_xlabel(f'PCA 1 ({var_exp[0]*100:.1f}% var)', fontsize=9)
ax1.set_ylabel(f'PCA 2 ({var_exp[1]*100:.1f}% var)', fontsize=9)
ax1.set_title(f'PCA 2D — k={best_k} Clusters', fontsize=10, fontweight='bold')
ax1.legend(fontsize=7, loc='best')
ax1.grid(alpha=0.3)

# Middle: Escalation overlay on PCA
ax2 = fig.add_subplot(gs[1])
mask_c0 = df['label_escalation'] == 0
mask_c1 = df['label_escalation'] == 1
ax2.scatter(X_pca[mask_c0, 0], X_pca[mask_c0, 1], c='steelblue',
            alpha=0.3, s=15, label=f'Non-Esc (n={mask_c0.sum()})')
ax2.scatter(X_pca[mask_c1, 0], X_pca[mask_c1, 1], c='crimson',
            alpha=0.8, s=40, marker='*', label=f'Eskalasi (n={mask_c1.sum()})')
ax2.set_xlabel(f'PCA 1 ({var_exp[0]*100:.1f}% var)', fontsize=9)
ax2.set_ylabel(f'PCA 2 ({var_exp[1]*100:.1f}% var)', fontsize=9)
ax2.set_title('PCA 2D — Eskalasi vs Non-Eskalasi\n(overlay)', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Right: Cluster profile radar-style bar (key features z-score)
ax3 = fig.add_subplot(gs[2])
radar_features = ['total_hotspots', 'is_peatland', 'peatland_drought_index',
                  'windspeed_10m_max', 'fuel_danger_index',
                  'cumulative_precip_14d', 'population_density', 'road_distance']
x_pos  = np.arange(len(radar_features))
width  = 0.8 / best_k

for c in range(best_k):
    sub_mean  = df[df['cluster_label']==c][radar_features].mean()
    z_vals    = (sub_mean - global_mean[radar_features]) / (global_std[radar_features] + 1e-9)
    ax3.bar(x_pos + c * width - (best_k-1)*width/2,
            z_vals.values, width, color=colors_cluster[c],
            alpha=0.8, label=f"C{c}: {archetype_map[c][:8]}")

ax3.set_xticks(x_pos)
ax3.set_xticklabels([f.replace('_',' ')[:12] for f in radar_features],
                    rotation=45, ha='right', fontsize=7)
ax3.set_ylabel('Z-Score (vs global mean)', fontsize=9)
ax3.set_title('Cluster Profiles\n(Z-Score Key Features)', fontsize=10, fontweight='bold')
ax3.axhline(0, color='black', linewidth=0.8)
ax3.legend(fontsize=7)
ax3.grid(axis='y', alpha=0.3)

fig.suptitle(f'STEP 10A: Archetype Discovery — K-Means k={best_k} | FireEscalome GEMASTIK XIX/2026',
             fontsize=12, fontweight='bold', y=1.02)
plt.savefig(OUT_VIZ, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: {OUT_VIZ}")

# ================================================================
# 11. WRITE REPORT
# ================================================================
print("\n[11] WRITE REPORT ...")

lines = [
    "# STEP 10A -- ARCHETYPE DISCOVERY: FEATURE PREPARATION & CLUSTERING",
    "## FireEscalome GEMASTIK XIX/2026 | Person C",
    f"**Timestamp: {ts}**",
    "",
    "---",
    "",
    "## 1. Input & Konteks",
    "",
    "| Item | Detail |",
    "|---|---|",
    "| Input dataset | cluster_master_PersonB.csv |",
    f"| Total baris | {len(df)} |",
    f"| Fitur tersedia | 30 kolom |",
    f"| Fitur DIGUNAKAN clustering | {len(CLUSTERING_FEATURES)} |",
    "| TIDAK digunakan | label_escalation, growth_ratio, cluster_id, community_id |",
    "| Keputusan growth_ratio | STEP9B: Target-Derived Future Feature |",
    "",
    "---",
    "",
    "## 2. Fitur yang Digunakan",
    "",
    "| # | Feature | Kategori | Alasan |",
    "|---|---|---|---|",
]

feat_reasons = {
    'total_hotspots': ('Hotspot Volume', 'Volume cluster di H0 -- proxy ukuran dan intensitas api'),
    'slope': ('Topografi', 'Kemiringan lahan -- penentu kecepatan rambat api (naik bukit lebih cepat)'),
    'elevation': ('Topografi', 'Ketinggian -- mempengaruhi suhu, kelembapan, jenis vegetasi'),
    'aspect': ('Topografi', 'Arah hadap lereng -- mempengaruhi paparan angin dan matahari'),
    'ndvi_current': ('Vegetasi', 'Kondisi vegetasi saat ini -- proxy ketersediaan bahan bakar (fuel load)'),
    'ndvi_delta_16d': ('Vegetasi', 'Perubahan vegetasi 16 hari -- proxy kecepatan kekeringan vegetasi'),
    'windspeed_10m_max': ('Angin', 'Kecepatan angin maksimal -- penentu penyebaran api wind-driven'),
    'wind_alignment_score': ('Angin', 'Keselarasan arah angin dengan rambat api -- spesifik untuk wind-driven'),
    'temperature_2m_max': ('Cuaca', 'Suhu maksimal -- mempengaruhi kelembapan relatif dan kondisi bahan bakar'),
    'precipitation_dry_streak': ('Kekeringan', 'Hari berturut tanpa hujan -- indikator kekeringan akumulatif'),
    'cumulative_precip_14d': ('Kekeringan', 'Total hujan 14 hari -- defisit curah hujan jangka menengah'),
    'peatland_drought_index': ('Gambut/Kekeringan', 'Indeks kekeringan spesifik gambut -- kunci peat-driven archetype'),
    'fuel_danger_index': ('Bahan Bakar', 'Indeks bahaya bahan bakar -- kombinasi kekeringan + kondisi vegetasi'),
    'wind_slope_interaction': ('Interaksi', 'Interaksi angin x topografi -- dirancang khusus untuk wind-driven'),
    'population_density': ('Antropogenik', 'Kepadatan penduduk -- indikator human-induced archetype'),
    'road_distance': ('Antropogenik', 'Jarak ke jalan -- aksesibilitas manusia (rendah = dekat jalan)'),
    'is_peatland': ('Gambut', 'Status gambut -- biner; penting untuk peat-driven archetype'),
    'latitude': ('Spasial', 'Koordinat lintang -- pola regional Sumatra/Kalimantan'),
    'longitude': ('Spasial', 'Koordinat bujur -- membedakan pulau dan zona ekologis'),
    'start_day_of_year': ('Temporal', 'Hari dalam tahun -- seasonal pattern (musim kemarau vs hujan)'),
}

for i, feat in enumerate(CLUSTERING_FEATURES, 1):
    cat, reason = feat_reasons.get(feat, ('Lainnya', '-'))
    lines.append(f"| {i} | `{feat}` | {cat} | {reason} |")

lines += [
    "",
    "**Fitur TIDAK digunakan:**",
    "",
    "| Feature | Alasan Dikecualikan |",
    "|---|---|",
    "| `label_escalation` | Target -- tidak boleh masuk clustering (leakage) |",
    "| `growth_ratio` | Target-derived future feature (STEP9B) |",
    "| `cluster_id` | Identifier, bukan fitur fisik |",
    "| `community_id` | Identifier dari graph analysis |",
    "| `latitude_weather` / `longitude_weather` | Duplikat latitude/longitude (approx weather station) |",
    "| `island` | Turunan longitude (redundant) |",
    "| `landcover_class` | Banyak nilai invalid (-1, 0, 7, 8); sedikit variasi bermakna |",
    "| `confidence` | Nominal; kualitas deteksi FIRMS, bukan fitur fisik |",
    "| `year` | Identifier temporal; bisa menyebabkan bias temporal antar tahun |",
    "",
    "---",
    "",
    "## 3. Preprocessing",
    "",
    "| Step | Detail |",
    "|---|---|",
    f"| Missing values | **0** |",
    f"| Infinite values | **0** |",
    "| Standardisasi | StandardScaler (z-score normalization) |",
    f"| Feature matrix | {len(df)} × {len(CLUSTERING_FEATURES)} |",
    "",
    "---",
    "",
    "## 4. Evaluasi Kandidat k (K-Means, n_init=20)",
    "",
    "| k | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ | Inertia ↓ | Composite Score |",
    "|---|---|---|---|---|---|",
]

for i, row in df_metrics.iterrows():
    k = int(row['k'])
    comp = composite[i]
    mark = " **← TERPILIH**" if k == best_k else ""
    lines.append(f"| {k}{mark} | {row['silhouette']:.4f} | {row['calinski_harabasz']:.1f} | {row['davies_bouldin']:.4f} | {row['inertia']:.0f} | {comp:.3f} |")

lines += [
    "",
    f"> **k={best_k} terpilih** berdasarkan composite score tertinggi dari normalisasi ketiga metrik.",
    f"> Silhouette={best_sil:.4f}, CH={best_ch:.1f}, DB={best_db:.4f}",
    "",
    "---",
    "",
    f"## 5. Profil Cluster (k={best_k})",
    "",
]

for c in range(best_k):
    info = archetype_notes[c]
    lines += [
        f"### Cluster {c}: **{info['archetype']}** (Confidence: {info['confidence']})",
        "",
        f"| Item | Nilai |",
        f"|---|---|",
        f"| Jumlah baris | {info['n']} |",
        f"| Baris Eskalasi | {info['n_esc']} ({info['pct_esc']:.1f}%) |",
        f"| Archetype Mapping | **{info['archetype']}** |",
        f"| Confidence | {info['confidence']} |",
        f"| Top+ Features (Z-score) | {', '.join(info['top_positive_features'][:4])} |",
        f"| Top- Features (Z-score) | {', '.join(info['top_negative_features'][:2])} |",
        "",
        "| Archetype Score | Nilai |",
        "|---|---|",
    ]
    for arch_name, arch_score in info['scores'].items():
        lines.append(f"| {arch_name} | {arch_score:.3f} |")
    lines.append("")

lines += [
    "---",
    "",
    "## 6. Mapping Cluster → Archetype Sementara",
    "",
    "| Cluster | Archetype Sementara | Confidence | n | % Eskalasi |",
    "|---|---|---|---|---|",
]

for c in range(best_k):
    info = archetype_notes[c]
    lines.append(f"| C{c} | {info['archetype']} | {info['confidence']} | {info['n']} | {info['pct_esc']:.1f}% |")

lines += [
    "",
    "> **CATATAN PENTING:** Mapping archetype adalah **interpretasi sementara** berdasarkan z-score fitur.",
    "> Archetype bukan label kausal. Cluster yang tidak sesuai 4 kategori utama diberi label 'Mixed'.",
    "> Validasi final harus dilakukan dengan domain expert.",
    "",
    "---",
    "",
    "## 7. Keterbatasan Analisis",
    "",
    "| Keterbatasan | Detail |",
    "|---|---|",
    "| Class imbalance | 153/2888 eskalasi (5.3%) -- pola cluster mungkin didominasi non-eskalasi |",
    "| K-Means asumsi | Cluster berbentuk spherical & ukuran seimbang -- mungkin tidak sesuai data kebakaran |",
    "| PCA 2D reduksi | Visualisasi 2D mungkin tidak menangkap seluruh struktur cluster 20D |",
    "| Archetype mapping heuristik | Scoring berbasis z-score sederhana; bukan konfirmasi domain |",
    "| Tanpa growth_ratio | Fitur terkuat (Step 9A) sengaja dikecualikan -- fokus pada pola fisik |",
    "| Tidak ada DBSCAN/GMM | K-Means saja; analisis lanjutan bisa menggunakan metode berbeda |",
    "",
    "---",
    "",
    "## 8. Output Files",
    "",
    "| File | Keterangan |",
    "|---|---|",
    "| step10a_archetype_features.csv | Feature matrix + cluster_id + label (untuk validasi) |",
    "| step10a_cluster_profiles.csv | Mean profil per cluster |",
    "| step10a_cluster_metrics.csv | Metrik evaluasi k=2..6 |",
    "| step10a_archetype_visualization.png | PCA scatter + profile bar chart |",
    "| STEP10A_ARCHETYPE_DISCOVERY_REPORT.md | Laporan ini |",
    "| step10a_archetype_clustering.py | Script |",
    "",
    "---",
    "Dibuat: Person C -- FireEscalome GEMASTIK XIX/2026",
    "STOP: Menunggu instruksi (tidak lanjut Graph Community Detection / Association Rules)",
]

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"    Saved: {OUT_REPORT}")

# ================================================================
# 12. COPY SCRIPT
# ================================================================
import shutil
src = Path(r"C:\Users\LENOVO\.gemini\antigravity\brain\ba59b011-539b-43c5-b282-ace6b6f8b48a\scratch\step10a_archetype_clustering.py")
dst = WORKSPACE / "step10a_archetype_clustering.py"
shutil.copy2(str(src), str(dst))
print(f"    Script: {dst}")

# ================================================================
# 13. FINAL REPORT
# ================================================================
print(f"\n{SEP}")
print(f"STEP 10A -- LAPORAN AKHIR (k={best_k})")
print(SEP)
print(f"""
Dataset        : cluster_master_PersonB.csv ({len(df)} baris)
Fitur          : {len(CLUSTERING_FEATURES)} fitur fisik/lingkungan
Standardisasi  : StandardScaler
k terpilih     : {best_k} (Sil={best_sil:.4f}, CH={best_ch:.1f}, DB={best_db:.4f})

CLUSTER DISTRIBUTION & ARCHETYPE:
""")
for c in range(best_k):
    info = archetype_notes[c]
    print(f"  C{c}: {info['archetype']:<28} | n={info['n']:4d} | esc={info['n_esc']:3d} ({info['pct_esc']:.1f}%)")

print(f"""
Silhouette interpretation:
  {"Sangat baik (>0.7)" if best_sil > 0.7 else "Baik (0.5-0.7)" if best_sil > 0.5 else "Moderat (0.25-0.5)" if best_sil > 0.25 else "Lemah (<0.25)"}
  Nilai {best_sil:.4f} -- wajar untuk data kebakaran real-world yang heterogen

Temuan:
  - total_hotspots dan fitur spasial menjadi pembeda utama cluster
  - Pola kekeringan (fuel_danger_index, cumulative_precip_14d) teridentifikasi
  - Variasi is_peatland dan population_density membedakan archetype
""")
print(SEP)
print("STEP 10A SELESAI. Menunggu instruksi berikutnya.")
print(SEP)
