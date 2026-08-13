"""
STEP 13 — SIMULASI NEAR REAL-TIME (NRT)
FireEscalome — GEMASTIK XIX/2026

Simulasi pipeline prediksi end-to-end menggunakan data 2025-2026
sebagai proxy Near Real-Time (NRT).

Cara kerja NRT sesungguhnya:
  - FIRMS NRT API → cluster baru hari ini
  - Open-Meteo Forecast API → cuaca hari ini
  - → Model I → skor risiko + archetype
  - → Output triase peta

Simulasi ini:
  - Ambil data 2025-2026 dari cluster_master_PersonB.csv (310 cluster)
  - Anggap sebagai "cluster baru yang masuk hari ini"
  - Jalankan Model I (retrain dari data 2020-2024 sebagai training)
  - Assign archetype dari Pattern Library
  - Output: tabel triase + peta visualisasi

CATATAN: Model I tidak menggunakan growth_ratio (temporally valid)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
WORKSPACE = Path("G:/My Drive/FireEscalome")

FEATURE_COLS = [
    'island', 'year', 'start_day_of_year', 'latitude', 'longitude',
    'is_peatland', 'landcover_class', 'confidence', 'total_hotspots',
    'windspeed_10m_max', 'wind_alignment_score', 'slope', 'elevation',
    'aspect', 'ndvi_current', 'ndvi_delta_16d', 'temperature_2m_max',
    'precipitation_dry_streak', 'cumulative_precip_14d',
    'peatland_drought_index', 'fuel_danger_index',
    'wind_slope_interaction', 'population_density', 'road_distance'
]
TARGET = 'label_escalation'

ARCHETYPE_COLORS = {
    'Peat-driven':    '#8B4513',
    'Wind-driven':    '#4169E1',
    'Drought-driven': '#FF8C00',
    'Human-induced':  '#DC143C',
    'Multi-factor':   '#9932CC',
    'Low-risk':       '#90EE90'
}

print("=" * 65)
print("STEP 13 — SIMULASI NRT FIREESCALOME")
print(f"Waktu simulasi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/6] Loading data...")
df = pd.read_csv(WORKSPACE / "cluster_master_PersonB.csv")

# Split: 2020-2024 = training historical | 2025-2026 = NRT simulation
df_train = df[df['year'] <= 2024].copy()
df_nrt   = df[df['year'] >= 2025].copy()

print(f"  Historical (training): {len(df_train):,} cluster (2020-2024)")
print(f"  NRT simulation:        {len(df_nrt):,} cluster (2025-2026)")
print(f"  NRT eskalasi nyata:    {df_nrt[TARGET].sum()} cluster")

# Encode categorical string columns
from sklearn.preprocessing import LabelEncoder
df_all = pd.concat([df_train, df_nrt], ignore_index=True)
for col in ['island', 'confidence']:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))
df_train_enc = df_all[df_all['year'] <= 2024].copy()
df_nrt_enc   = df_all[df_all['year'] >= 2025].copy()

X_train = df_train_enc[FEATURE_COLS]
y_train = df_train_enc[TARGET]
X_nrt   = df_nrt_enc[FEATURE_COLS]
y_nrt   = df_nrt[TARGET]  # ground truth (tersimpan, tidak dipakai saat prediksi)

# ─────────────────────────────────────────────
# 2. TRAIN MODEL I (Early-Warning, tanpa growth_ratio)
# ─────────────────────────────────────────────
print("\n[2/6] Training Model I (Early-Warning, no growth_ratio)...")
try:
    import lightgbm as lgb
    from sklearn.metrics import (classification_report, precision_recall_curve,
                                  average_precision_score, f1_score)

    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / pos

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=spw,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    print(f"  Model trained | scale_pos_weight={spw:.1f}")

except ImportError:
    print("  LightGBM tidak tersedia. Gunakan RandomForest fallback...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (classification_report, precision_recall_curve,
                                  average_precision_score, f1_score)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("  RandomForest trained (fallback)")

# ─────────────────────────────────────────────
# 3. PREDIKSI NRT
# ─────────────────────────────────────────────
print("\n[3/6] Running NRT predictions...")
proba = model.predict_proba(X_nrt)[:, 1]
pred  = (proba >= 0.5).astype(int)

# Threshold optimal: F1-based
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.05)
f1s = [f1_score(y_nrt, (proba >= t).astype(int), zero_division=0) for t in thresholds]
best_thresh = thresholds[np.argmax(f1s)]
pred_opt = (proba >= best_thresh).astype(int)

print(f"  Threshold default (0.5): {pred.sum()} alert clusters")
print(f"  Threshold optimal ({best_thresh:.2f}): {pred_opt.sum()} alert clusters")

ap = average_precision_score(y_nrt, proba)
print(f"  PR-AUC (vs ground truth): {ap:.4f}")

# ─────────────────────────────────────────────
# 4. ASSIGN ARCHETYPE dari Pattern Library
# ─────────────────────────────────────────────
print("\n[4/6] Assigning archetypes from Pattern Library...")

def assign_archetype_nrt(row, prob):
    """
    Rule-based archetype assignment berdasarkan Pattern Library Step 12.
    Threshold: P33 dari training data.
    """
    hotspot_low  = row['total_hotspots'] <= 2
    peat_yes     = row['is_peatland'] == 1
    ndvi_low     = row['ndvi_current'] < 0.3
    wind_high    = row['windspeed_10m_max'] > 12
    pop_dense    = row['population_density'] > 500
    road_mid     = row['road_distance'] < 5000
    fuel_danger  = row['fuel_danger_index'] > 10
    lon_west     = row['longitude'] < 107
    rain_low     = row['precipitation_dry_streak'] >= 3
    drought_idx  = row['peatland_drought_index'] > 5

    if prob < 0.15:
        return 'Low-risk'

    # Hitung sinyal per archetype
    human_score   = int(pop_dense) + int(road_mid) + int(lon_west)
    drought_score = int(fuel_danger) + int(ndvi_low) + int(rain_low) + int(drought_idx)
    peat_score    = int(peat_yes) + int(hotspot_low) + int(drought_idx)
    wind_score    = int(wind_high) + int(lon_west) + int(hotspot_low)

    scores = {
        'Human-induced':  human_score,
        'Drought-driven': drought_score,
        'Peat-driven':    peat_score,
        'Wind-driven':    wind_score,
    }
    top_score = max(scores.values())
    top_types = [k for k, v in scores.items() if v == top_score]

    if len(top_types) > 1:
        return 'Multi-factor'
    return top_types[0]

df_nrt = df_nrt.copy()
df_nrt['risk_score']  = proba
df_nrt['alert']       = pred_opt
df_nrt['archetype']   = df_nrt.apply(
    lambda r: assign_archetype_nrt(r, r['risk_score']), axis=1)

# Risk level
def risk_level(p):
    if p >= 0.7: return 'CRITICAL'
    if p >= 0.5: return 'HIGH'
    if p >= 0.3: return 'MEDIUM'
    if p >= 0.15: return 'LOW'
    return 'MINIMAL'

df_nrt['risk_level'] = df_nrt['risk_score'].apply(risk_level)

# ─────────────────────────────────────────────
# 5. TABEL TRIASE
# ─────────────────────────────────────────────
print("\n[5/6] Generating triase output...")

alert_df = df_nrt[df_nrt['alert'] == 1].copy()
alert_df = alert_df.sort_values('risk_score', ascending=False)

triase_cols = ['cluster_id', 'year', 'start_day_of_year', 'island',
               'latitude', 'longitude', 'total_hotspots',
               'risk_score', 'risk_level', 'archetype',
               'is_peatland', 'windspeed_10m_max', 'fuel_danger_index']
triase_out = alert_df[triase_cols].copy()
triase_out['risk_score'] = triase_out['risk_score'].round(4)

triase_out.to_csv(WORKSPACE / "step13_nrt_triase_output.csv", index=False)
print(f"  Alert clusters: {len(triase_out)}")
print(f"\n  TOP 10 HIGHEST RISK:")
print(triase_out[['cluster_id','island','latitude','longitude',
                   'risk_score','risk_level','archetype']].head(10).to_string(index=False))

print(f"\n  ARCHETYPE DISTRIBUTION (Alert clusters):")
arch_dist = alert_df['archetype'].value_counts()
for arch, cnt in arch_dist.items():
    pct = cnt / len(alert_df) * 100
    print(f"    {arch:<20} {cnt:3d} clusters ({pct:.1f}%)")

print(f"\n  RISK LEVEL SUMMARY:")
risk_dist = df_nrt['risk_level'].value_counts()
for lvl in ['CRITICAL','HIGH','MEDIUM','LOW','MINIMAL']:
    cnt = risk_dist.get(lvl, 0)
    print(f"    {lvl:<10} {cnt:3d} clusters")

# ─────────────────────────────────────────────
# 6. VISUALISASI PETA TRIASE
# ─────────────────────────────────────────────
print("\n[6/6] Generating NRT triase map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#0a0a1a')

for ax in axes:
    ax.set_facecolor('#0a0a1a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

# — Panel 1: Risk Score Map
ax1 = axes[0]
sc = ax1.scatter(
    df_nrt['longitude'], df_nrt['latitude'],
    c=df_nrt['risk_score'],
    cmap='YlOrRd', s=40, alpha=0.8,
    vmin=0, vmax=1, edgecolors='none'
)
# Highlight alerts
ax1.scatter(
    alert_df['longitude'], alert_df['latitude'],
    c='white', s=120, alpha=0.9,
    edgecolors='red', linewidths=1.5,
    marker='*', label=f'Alert ({len(alert_df)})', zorder=5
)
cb = plt.colorbar(sc, ax=ax1, shrink=0.8)
cb.set_label('Risk Score', color='white', fontsize=10)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
ax1.set_title('NRT Risk Score Map\n(2025–2026 Simulation)', 
              color='white', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Longitude', color='white', fontsize=9)
ax1.set_ylabel('Latitude', color='white', fontsize=9)
ax1.legend(loc='lower right', framealpha=0.3, fontsize=8,
           facecolor='#1a1a2e', labelcolor='white')
ax1.grid(True, alpha=0.15, color='gray')

# — Panel 2: Archetype Map
ax2 = axes[1]
for arch, color in ARCHETYPE_COLORS.items():
    subset = df_nrt[df_nrt['archetype'] == arch]
    if len(subset) > 0:
        size = 80 if arch != 'Low-risk' else 20
        alpha = 0.9 if arch != 'Low-risk' else 0.3
        ax2.scatter(subset['longitude'], subset['latitude'],
                   c=color, s=size, alpha=alpha,
                   label=f'{arch} ({len(subset)})',
                   edgecolors='none', zorder=3)

# Mark escalation ground truth
esc_gt = df_nrt[df_nrt[TARGET] == 1]
ax2.scatter(esc_gt['longitude'], esc_gt['latitude'],
           c='none', s=200, edgecolors='yellow',
           linewidths=2, marker='o', label=f'Actual Escalation ({len(esc_gt)})',
           zorder=6, alpha=0.8)

ax2.set_title('NRT Archetype Classification\n(Pattern Library Assignment)', 
              color='white', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Longitude', color='white', fontsize=9)
ax2.set_ylabel('Latitude', color='white', fontsize=9)
ax2.legend(loc='lower right', framealpha=0.4, fontsize=7,
           facecolor='#1a1a2e', labelcolor='white', ncol=1)
ax2.grid(True, alpha=0.15, color='gray')

fig.suptitle(
    f'FireEscalome — NRT Triase Simulation | {len(df_nrt)} Clusters | '
    f'{len(alert_df)} Alerts | PR-AUC={ap:.3f}',
    color='white', fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(WORKSPACE / "step13_nrt_triase_map.png",
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  Saved: step13_nrt_triase_map.png")

# ─────────────────────────────────────────────
# 7. SIMPAN NRT FULL OUTPUT
# ─────────────────────────────────────────────
nrt_full = df_nrt[['cluster_id','year','start_day_of_year','island',
                    'latitude','longitude','total_hotspots','is_peatland',
                    'windspeed_10m_max','fuel_danger_index','precipitation_dry_streak',
                    'risk_score','risk_level','archetype','alert',TARGET]].copy()
nrt_full.to_csv(WORKSPACE / "step13_nrt_full_output.csv", index=False)

print("\n" + "=" * 65)
print("STEP 13 SELESAI — OUTPUT:")
print(f"  step13_nrt_triase_output.csv  ({len(triase_out)} alert clusters)")
print(f"  step13_nrt_full_output.csv    ({len(nrt_full)} all NRT clusters)")
print(f"  step13_nrt_triase_map.png     (peta triase dual panel)")
print("=" * 65)
print(f"\nSIMULASI NRT BERHASIL")
print(f"Pipeline end-to-end: Data → Model I → Archetype → Triase Map")
print(f"Waktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
