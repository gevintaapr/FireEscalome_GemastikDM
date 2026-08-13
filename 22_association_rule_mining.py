"""
step11_association_rules.py
Step 11 -- Association Rule Mining: Early-Warning Pattern Rules
FireEscalome GEMASTIK XIX/2026 -- Person C

Metodologi:
  - Apriori + association_rules (mlxtend)
  - Antecedent: fitur fisik H0 (NO growth_ratio, NO label_escalation)
  - Consequent: Escalation=Yes
  - Diskritisasi: quantile-based (tertile) -- tidak memaksa threshold manual
  - Evaluasi: lift + confidence + support + coverage positif
  - Archetype profiling: Peat / Wind / Drought / Human / Mixed (data-driven)
  - Class imbalance: baseline 5.3% → lift > 3 bermakna

Input (semua dari G:/My Drive/FireEscalome/):
  cluster_master_PersonB.csv  (full dataset, 2888 baris, 153 eskalasi)
  step9c_shap_interaction_top15.csv
  step10a_archetype_features.csv
  step10a_cluster_profiles.csv
  step10b_archetype_assignments.csv

Output:
  step11_association_rules.csv       -- semua valid rules
  step11_top_rules.csv               -- top 20 rules by lift
  step11_archetype_rules.csv         -- rules dikelompokkan per archetype
  step11_rule_summary.png            -- visualisasi
  STEP11_ASSOCIATION_RULE_MINING_REPORT.md
  step11_association_rules.py        -- copy ke workspace
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

WORKSPACE = Path(r"G:\My Drive\FireEscalome")
RANDOM_STATE = 42

ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
SEP = "=" * 65

print(SEP)
print("STEP 11 -- ASSOCIATION RULE MINING: EARLY-WARNING PATTERNS")
print(f"Timestamp: {ts}")
print(SEP)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\n[1] LOAD DATA ...")

df = pd.read_csv(WORKSPACE / "cluster_master_PersonB.csv")
df_shap = pd.read_csv(WORKSPACE / "step9c_shap_interaction_top15.csv")

n_total = len(df)
n_pos   = int(df['label_escalation'].sum())
n_neg   = n_total - n_pos
base_rate = n_pos / n_total

print(f"    Dataset: {n_total} baris | Eskalasi={n_pos} ({base_rate*100:.2f}%) | Non={n_neg}")
print(f"    SHAP interaction top15 loaded: {len(df_shap)} pairs")

# Also load archetype assignments
df_arch = pd.read_csv(WORKSPACE / "step10b_archetype_assignments.csv")
comm_to_arch = dict(zip(df_arch['community_id'], df_arch['archetype']))

# ================================================================
# 2. FEATURE SELECTION (SHAP-GUIDED)
# ================================================================
print("\n[2] FEATURE SELECTION ...")

# Features from SHAP top15 interactions (Step 9C)
# EXCLUDE: growth_ratio (target-derived future feature, STEP9B)
# EXCLUDE: label_escalation (only as consequent)
SHAP_FEATURES = [
    'total_hotspots',        # hub interaction (10/15 pairs)
    'fuel_danger_index',     # drought/fuel signal
    'cumulative_precip_14d', # drought/rainfall signal
    'windspeed_10m_max',     # wind signal
    'wind_alignment_score',  # wind directionality
    'wind_slope_interaction',# wind-terrain coupling
    'slope',                 # topography
    'population_density',    # human-induced signal
    'road_distance',         # human-induced (inverted)
    'precipitation_dry_streak', # drought
    'longitude',             # spatial moderator
]

# Also include key archetype features not in top15 SHAP but domain-relevant
EXTRA_FEATURES = [
    'is_peatland',           # peat-driven archetype marker (binary)
    'peatland_drought_index',# peat drought signal
    'ndvi_current',          # vegetation stress
]

CANDIDATE_FEATURES = SHAP_FEATURES + EXTRA_FEATURES
print(f"    SHAP-guided features   : {len(SHAP_FEATURES)}")
print(f"    Extra domain features  : {len(EXTRA_FEATURES)}")
print(f"    Total candidates       : {len(CANDIDATE_FEATURES)}")
print(f"    Excluded (by design)   : growth_ratio (target-derived), label_escalation (consequent only)")

# ================================================================
# 3. DISCRETISATION (quantile-based, data-driven)
# ================================================================
print("\n[3] DISCRETIZATION (quantile tertile -- no manual threshold) ...")

# Strategy:
# - Binary features (is_peatland): keep as-is → "peat=yes" / "peat=no"
# - Continuous features: tertile split using pd.qcut on TRAINING portion only
# - For road_distance: inverted naming (low distance = near road = human signal)
# - Labels chosen to be domain-meaningful

BINARY_FEATURES = {'is_peatland': ('peat_yes', 'peat_no')}

# Quantile cutpoints computed from FULL dataset (all 2888)
# Note: in a strict ML sense we'd use training only, but for descriptive ARM
# we use the full dataset since we are doing pattern discovery, not prediction
# This is standard practice in ARM literature

items_df = pd.DataFrame(index=df.index)
cutpoints_log = {}

def safe_qcut(series, q=3, labels=None, retbins=False):
    """qcut with duplicate handling"""
    try:
        result = pd.qcut(series, q=q, labels=labels, duplicates='drop', retbins=retbins)
        return result
    except Exception as e:
        # Fallback: cut by equal width
        result = pd.cut(series, bins=q, labels=labels[:len(pd.cut(series, bins=q).cat.categories)] if labels else None, retbins=retbins)
        return result

for feat in CANDIDATE_FEATURES:
    if feat not in df.columns:
        print(f"    SKIP (not in dataset): {feat}")
        continue

    series = df[feat]

    if feat in BINARY_FEATURES:
        pos_label, neg_label = BINARY_FEATURES[feat]
        items_df[pos_label] = (series == 1)
        items_df[neg_label] = (series == 0)
        cutpoints_log[feat] = f"binary: 1={pos_label}, 0={neg_label}"
        continue

    # Domain-aware label naming
    label_map = {
        'total_hotspots'        : ('hotspot_high', 'hotspot_med', 'hotspot_low'),
        'fuel_danger_index'     : ('fuel_danger_high', 'fuel_danger_med', 'fuel_danger_low'),
        'cumulative_precip_14d' : ('rain14d_high', 'rain14d_med', 'rain14d_low'),
        'windspeed_10m_max'     : ('wind_high', 'wind_med', 'wind_low'),
        'wind_alignment_score'  : ('wind_align_high', 'wind_align_med', 'wind_align_low'),
        'wind_slope_interaction': ('windslope_high', 'windslope_med', 'windslope_low'),
        'slope'                 : ('slope_steep', 'slope_mod', 'slope_flat'),
        'population_density'    : ('pop_dense', 'pop_mod', 'pop_sparse'),
        'road_distance'         : ('road_near', 'road_mid', 'road_far'),  # note: near=low distance
        'precipitation_dry_streak': ('dry_streak_long', 'dry_streak_med', 'dry_streak_short'),
        'longitude'             : ('lon_east', 'lon_mid', 'lon_west'),
        'peatland_drought_index': ('peat_drought_high', 'peat_drought_med', 'peat_drought_low'),
        'ndvi_current'          : ('ndvi_high', 'ndvi_med', 'ndvi_low'),
    }

    labels = label_map.get(feat, (f'{feat}_high', f'{feat}_med', f'{feat}_low'))

    # For road_distance, low values = near road = human signal → invert ordering
    if feat == 'road_distance':
        labels = ('road_far', 'road_mid', 'road_near')
        cat = safe_qcut(series, q=3, labels=labels)
    elif feat == 'longitude':
        # West=Sumatra (low lon), East=Kalimantan (high lon)
        labels = ('lon_west', 'lon_mid', 'lon_east')
        cat = safe_qcut(series, q=3, labels=labels)
    else:
        cat = safe_qcut(series, q=3, labels=labels)

    # Handle NaN from qcut
    if hasattr(cat, 'cat'):
        for lbl in cat.cat.categories:
            items_df[str(lbl)] = (cat == lbl)
        try:
            _, bins = pd.qcut(series, q=3, duplicates='drop', retbins=True)
            cutpoints_log[feat] = f"P33={bins[1]:.3f}, P67={bins[2]:.3f}"
        except:
            cutpoints_log[feat] = "quantile bins"
    else:
        print(f"    WARNING: {feat} discretization returned non-categorical; skipping")

# Add target: Escalation=Yes / No
items_df['Escalation_YES'] = df['label_escalation'].astype(bool)
items_df['Escalation_NO']  = (df['label_escalation'] == 0)

# Add community_id and archetype for stratified analysis
items_df['community_id']    = df['community_id'].values
items_df['label_escalation']= df['label_escalation'].values

print(f"    Discretized items: {len([c for c in items_df.columns if c not in ['community_id','label_escalation']])}")
print(f"    Cutpoints log:")
for feat, cp in cutpoints_log.items():
    print(f"      {feat:<30}: {cp}")

# ================================================================
# 4. APRIORI FREQUENT ITEMSETS
# ================================================================
print("\n[4] APRIORI FREQUENT ITEMSETS ...")

# Exclude meta columns from ARM
ARM_COLS = [c for c in items_df.columns if c not in ['community_id', 'label_escalation']]
X_bool   = items_df[ARM_COLS].astype(bool)

# min_support calibration:
# Base rate = 5.3% → even 10% of positive cases = 15 rows → support = 15/2888 = 0.0052
# Use min_support=0.005 to capture rare-but-meaningful patterns
# Also run with 0.01 for higher quality rules

MIN_SUPPORT = 0.005   # ~14 instances (floor for exploratory rules)
MIN_SUPPORT_STRONG = 0.015  # ~43 instances (floor for strong rules)

print(f"    min_support (exploratory) : {MIN_SUPPORT} ({int(MIN_SUPPORT*n_total)} instances)")
print(f"    min_support (strong)      : {MIN_SUPPORT_STRONG} ({int(MIN_SUPPORT_STRONG*n_total)} instances)")

freq_itemsets = apriori(X_bool, min_support=MIN_SUPPORT, use_colnames=True, max_len=4)
print(f"    Frequent itemsets found   : {len(freq_itemsets)}")

# ================================================================
# 5. ASSOCIATION RULES
# ================================================================
print("\n[5] ASSOCIATION RULES (consequent = Escalation_YES) ...")

rules_all = association_rules(freq_itemsets, metric='confidence', min_threshold=0.05,
                               num_itemsets=len(freq_itemsets))
print(f"    Total rules (all consequents): {len(rules_all)}")

# Filter: only consequent = {Escalation_YES}
rules_all['consequents_str'] = rules_all['consequents'].apply(lambda x: frozenset(x))
rules_esc = rules_all[
    rules_all['consequents'].apply(lambda x: frozenset(x) == frozenset(['Escalation_YES']))
].copy()
print(f"    Rules with Escalation_YES as consequent: {len(rules_esc)}")

# Remove rules where antecedent contains Escalation_NO (trivial exclusion)
rules_esc = rules_esc[
    ~rules_esc['antecedents'].apply(lambda x: 'Escalation_NO' in x or 'Escalation_YES' in x)
].copy()
print(f"    After removing trivial/circular antecedents: {len(rules_esc)}")

# Add additional metrics
rules_esc['n_transactions'] = (rules_esc['support'] * n_total).round(0).astype(int)
rules_esc['n_positive_covered'] = (rules_esc['support'] * n_total * rules_esc['confidence'] / rules_esc['confidence']).round(0)
# Correct: n_positive_covered = n_transactions * confidence
rules_esc['n_positive_covered'] = (rules_esc['n_transactions'] * rules_esc['confidence']).round(0).astype(int)
rules_esc['coverage_of_positives_pct'] = (rules_esc['n_positive_covered'] / n_pos * 100).round(2)
rules_esc['antecedents_str'] = rules_esc['antecedents'].apply(lambda x: ' & '.join(sorted(x)))

# Classify support strength
rules_esc['strength'] = 'EXPLORATORY'
rules_esc.loc[rules_esc['support'] >= MIN_SUPPORT_STRONG, 'strength'] = 'STRONG'

# Remove very weak rules (lift < 2 — less than 2x baseline)
rules_esc_filtered = rules_esc[rules_esc['lift'] >= 2.0].copy()
print(f"    After lift >= 2.0 filter: {len(rules_esc_filtered)}")

# Sort by composite score: lift * confidence * log(support*n+1)
import numpy as np
rules_esc_filtered['composite_score'] = (
    rules_esc_filtered['lift'] *
    rules_esc_filtered['confidence'] *
    np.log1p(rules_esc_filtered['n_transactions'])
)
rules_esc_filtered = rules_esc_filtered.sort_values('composite_score', ascending=False).reset_index(drop=True)

# ================================================================
# 6. ARCHETYPE TAGGING
# ================================================================
print("\n[6] ARCHETYPE TAGGING ...")

# Domain keyword mapping → archetype
ARCHETYPE_KEYWORDS = {
    'Peat-driven'   : ['peat', 'peat_yes', 'peat_drought'],
    'Wind-driven'   : ['wind_high', 'wind_align_high', 'windslope_high'],
    'Drought-driven': ['dry_streak_long', 'rain14d_low', 'fuel_danger_high', 'ndvi_low'],
    'Human-induced' : ['pop_dense', 'road_near'],
    'Multi-factor'  : [],  # assigned when multiple archetype keywords present
}

def assign_archetype(antecedents_str):
    counts = {}
    for arch, kws in ARCHETYPE_KEYWORDS.items():
        if arch == 'Multi-factor':
            continue
        cnt = sum(1 for kw in kws if kw in antecedents_str)
        if cnt > 0:
            counts[arch] = cnt
    if not counts:
        return 'Unclear'
    if len(counts) >= 2:
        return 'Multi-factor'
    return max(counts, key=counts.get)

rules_esc_filtered['archetype'] = rules_esc_filtered['antecedents_str'].apply(assign_archetype)

arch_counts = rules_esc_filtered['archetype'].value_counts()
print(f"    Rules by archetype:")
for arch, cnt in arch_counts.items():
    print(f"      {arch:<20}: {cnt} rules")

# ================================================================
# 7. CHECK DOMAIN VALIDITY
# ================================================================
print("\n[7] DOMAIN VALIDITY TAGGING ...")

def check_domain_validity(ant_str):
    """
    Return 'VALID', 'WEAK', or 'QUESTIONABLE' based on domain rules:
    - Escalation should come with HIGH hotspot or HIGH danger or LOW rain or peat
    - QUESTIONABLE: rain_high + drought signal (contradictory)
    """
    # Contradictory signals
    if ('rain14d_high' in ant_str and 'dry_streak_long' in ant_str):
        return 'QUESTIONABLE (contradictory: high rain + long dry streak)'
    if ('fuel_danger_low' in ant_str):
        return 'QUESTIONABLE (low fuel danger → should inhibit escalation)'
    if ('ndvi_high' in ant_str and 'fuel_danger_high' in ant_str):
        return 'WEAK (high NDVI + high fuel danger is unusual)'
    # Strong positive signals
    positive_signals = ['hotspot_high', 'fuel_danger_high', 'dry_streak_long',
                        'rain14d_low', 'peat_yes', 'wind_high', 'pop_dense', 'road_near']
    n_pos_sig = sum(1 for ps in positive_signals if ps in ant_str)
    if n_pos_sig >= 1:
        return 'VALID'
    return 'VALID (weak signal pattern)'

rules_esc_filtered['domain_validity'] = rules_esc_filtered['antecedents_str'].apply(check_domain_validity)

# ================================================================
# 8. OVERLAP / REDUNDANCY CHECK
# ================================================================
print("\n[8] OVERLAP / REDUNDANCY ...")
# Mark rules where antecedent is a superset of a higher-lift rule (dominated)
# Simple approximation: check if antecedent of rule i contains antecedent of rule j (j has higher lift)
top_rules = rules_esc_filtered.head(30)
dominated = set()
for i in range(len(top_rules)):
    for j in range(len(top_rules)):
        if i == j: continue
        ant_i = set(top_rules.iloc[i]['antecedents'])
        ant_j = set(top_rules.iloc[j]['antecedents'])
        # If i's antecedent is superset of j and j has higher lift → i is dominated
        if ant_i.issuperset(ant_j) and top_rules.iloc[j]['lift'] >= top_rules.iloc[i]['lift']:
            dominated.add(i)

rules_esc_filtered['is_dominated'] = False
for idx in dominated:
    if idx < len(rules_esc_filtered):
        rules_esc_filtered.iloc[idx, rules_esc_filtered.columns.get_loc('is_dominated')] = True

print(f"    Dominated rules (superset of higher-lift rule): {len(dominated)}")

# ================================================================
# 9. OUTPUT CSVs
# ================================================================
print("\n[9] SAVE CSVs ...")

SAVE_COLS = ['antecedents_str', 'support', 'confidence', 'lift',
             'n_transactions', 'n_positive_covered', 'coverage_of_positives_pct',
             'composite_score', 'strength', 'archetype', 'domain_validity', 'is_dominated',
             'leverage', 'conviction', 'zhangs_metric']

# All valid rules
rules_save = rules_esc_filtered[SAVE_COLS].round(4)
rules_save.to_csv(WORKSPACE / "step11_association_rules.csv", index=False)
print(f"    step11_association_rules.csv: {len(rules_save)} rules")

# Top 20 by composite score (already sorted)
top20 = rules_esc_filtered.head(20)[SAVE_COLS].round(4)
top20.to_csv(WORKSPACE / "step11_top_rules.csv", index=False)
print(f"    step11_top_rules.csv: {len(top20)} rules")

# Per archetype
arch_rules_list = []
for arch in ['Peat-driven', 'Wind-driven', 'Drought-driven', 'Human-induced', 'Multi-factor', 'Unclear']:
    subset = rules_esc_filtered[rules_esc_filtered['archetype'] == arch].head(10)
    if len(subset) > 0:
        sub_save = subset[SAVE_COLS].copy()
        sub_save.insert(0, 'archetype_group', arch)
        arch_rules_list.append(sub_save)

if arch_rules_list:
    df_arch_rules = pd.concat(arch_rules_list, ignore_index=True).round(4)
    df_arch_rules.to_csv(WORKSPACE / "step11_archetype_rules.csv", index=False)
    print(f"    step11_archetype_rules.csv: {len(df_arch_rules)} rules")

# ================================================================
# 10. VISUALISASI
# ================================================================
print("\n[10] VISUALISASI ...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('STEP 11: Association Rule Mining — Early-Warning Pattern Rules\n'
             'FireEscalome GEMASTIK XIX/2026 | Person C (Model I, no growth_ratio)',
             fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

colors_arch = {
    'Peat-driven'   : '#8B4513',
    'Wind-driven'   : '#1E90FF',
    'Drought-driven': '#FF8C00',
    'Human-induced' : '#2ECC71',
    'Multi-factor'  : '#9B59B6',
    'Unclear'       : '#95A5A6',
}

# Panel 1: Top 15 rules by lift (scatter: support vs confidence, color=lift)
ax1 = fig.add_subplot(gs[0, :2])
top_vis = rules_esc_filtered.head(40)
sc = ax1.scatter(top_vis['support'], top_vis['confidence'],
                 c=top_vis['lift'], cmap='YlOrRd', s=top_vis['n_transactions']*3,
                 alpha=0.8, edgecolors='gray', linewidth=0.5)
plt.colorbar(sc, ax=ax1, label='Lift')
ax1.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='conf=0.30')
ax1.axvline(MIN_SUPPORT_STRONG, color='blue', linestyle='--', alpha=0.5,
            label=f'support={MIN_SUPPORT_STRONG} (strong)')
ax1.set_xlabel('Support', fontsize=10)
ax1.set_ylabel('Confidence', fontsize=10)
ax1.set_title('Rules: Support vs Confidence (color=Lift, size=n_transactions)\nTop 40 by composite score',
              fontsize=10, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Panel 2: Archetype distribution of rules
ax2 = fig.add_subplot(gs[0, 2])
arch_labels = [a for a in colors_arch if a in arch_counts.index]
arch_vals   = [arch_counts.get(a, 0) for a in arch_labels]
arch_colors = [colors_arch[a] for a in arch_labels]
bars = ax2.barh(arch_labels, arch_vals, color=arch_colors, edgecolor='white')
ax2.set_xlabel('Number of Rules', fontsize=9)
ax2.set_title('Rules by Archetype\n(lift≥2 filtered)', fontsize=10, fontweight='bold')
for bar, val in zip(bars, arch_vals):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)
ax2.grid(axis='x', alpha=0.3)

# Panel 3: Top 15 rules by lift -- horizontal bar
ax3 = fig.add_subplot(gs[1, :2])
top15_vis = rules_esc_filtered.head(15).copy()
short_labels = [
    f"#{i+1}: {row['antecedents_str'][:55]}..." if len(row['antecedents_str']) > 55
    else f"#{i+1}: {row['antecedents_str']}"
    for i, (_, row) in enumerate(top15_vis.iterrows())
]
bar_colors = [colors_arch.get(row['archetype'], '#95A5A6')
              for _, row in top15_vis.iterrows()]
hbars = ax3.barh(range(len(top15_vis)-1, -1, -1), top15_vis['lift'].values,
                 color=bar_colors, alpha=0.85, edgecolor='white')
ax3.set_yticks(range(len(top15_vis)-1, -1, -1))
ax3.set_yticklabels(short_labels, fontsize=7)
ax3.set_xlabel('Lift', fontsize=10)
ax3.set_title('Top 15 Rules by Lift\n(color = archetype)', fontsize=10, fontweight='bold')
for i, (bar, (_, row)) in enumerate(zip(hbars, top15_vis.iterrows())):
    ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f"lift={row['lift']:.1f} | conf={row['confidence']:.2f} | n={int(row['n_transactions'])}",
             va='center', fontsize=7)
ax3.set_xlim(0, top15_vis['lift'].max() * 1.35)
ax3.axvline(1/base_rate, color='red', linestyle='--', alpha=0.5,
            label=f'lift={1/base_rate:.0f} (100% prec)')
ax3.legend(fontsize=7)
ax3.grid(axis='x', alpha=0.3)

# Panel 4: Coverage of positive cases by top rules
ax4 = fig.add_subplot(gs[1, 2])
top10_cov = rules_esc_filtered.head(10).copy()
cov_labels = [f"Rule #{i+1}" for i in range(len(top10_cov))]
cov_vals   = top10_cov['coverage_of_positives_pct'].values
ax4.barh(range(len(top10_cov)-1, -1, -1), cov_vals,
         color='#3498DB', alpha=0.8, edgecolor='white')
ax4.set_yticks(range(len(top10_cov)-1, -1, -1))
ax4.set_yticklabels(cov_labels, fontsize=9)
ax4.set_xlabel('% Positive Cases Covered', fontsize=9)
ax4.set_title('Coverage of Escalation Cases\n(Top 10 Rules)', fontsize=10, fontweight='bold')
ax4.axvline(10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
ax4.legend(fontsize=8)
ax4.grid(axis='x', alpha=0.3)

plt.savefig(WORKSPACE / "step11_rule_summary.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: step11_rule_summary.png")

# ================================================================
# 11. WRITE REPORT
# ================================================================
print("\n[11] WRITE REPORT ...")

top_by_lift = rules_esc_filtered.nlargest(20, 'lift')
top_by_conf = rules_esc_filtered.nlargest(20, 'confidence')
n_strong    = (rules_esc_filtered['strength'] == 'STRONG').sum()
n_explor    = (rules_esc_filtered['strength'] == 'EXPLORATORY').sum()
n_valid     = (rules_esc_filtered['domain_validity'] == 'VALID').sum()

lines = [
    "# STEP 11 -- ASSOCIATION RULE MINING: EARLY-WARNING PATTERN RULES",
    "## FireEscalome GEMASTIK XIX/2026 | Person C",
    f"**Timestamp: {ts}**",
    "",
    "---",
    "",
    "## 1. Konteks Metodologis",
    "",
    "| Item | Detail |",
    "|---|---|",
    "| Dataset | cluster_master_PersonB.csv (2888 baris) |",
    f"| Total positif (eskalasi) | {n_pos} ({base_rate*100:.2f}%) |",
    "| Model basis | Early-Warning Model I (tanpa growth_ratio) |",
    "| Antecedent sumber | SHAP top15 interaction (Step 9C) + domain features |",
    "| Consequent | Escalation=YES |",
    "| Algoritma | Apriori + association_rules (mlxtend) |",
    f"| min_support exploratory | {MIN_SUPPORT} (~{int(MIN_SUPPORT*n_total)} instances) |",
    f"| min_support strong | {MIN_SUPPORT_STRONG} (~{int(MIN_SUPPORT_STRONG*n_total)} instances) |",
    "| Diskritisasi | Quantile tertile (data-driven, tidak memaksa threshold manual) |",
    "| growth_ratio | TIDAK digunakan (STEP9B: target-derived future feature) |",
    "",
    "---",
    "",
    "## 2. Ringkasan Rules",
    "",
    "| Metrik | Nilai |",
    "|---|---|",
    f"| Frequent itemsets | {len(freq_itemsets)} |",
    f"| Rules total (semua consequent) | {len(rules_all)} |",
    f"| Rules dengan Escalation_YES | {len(rules_esc)} |",
    f"| Setelah filter antecedent trivial | {len(rules_esc)} |",
    f"| Setelah filter lift >= 2.0 | {len(rules_esc_filtered)} |",
    f"| STRONG rules (support >= {MIN_SUPPORT_STRONG}) | {n_strong} |",
    f"| EXPLORATORY rules | {n_explor} |",
    f"| Domain VALID | {n_valid} |",
    "",
    "---",
    "",
    "## 3. Top 20 Rules by Lift",
    "",
    "| Rank | Antecedents | Support | Confidence | Lift | n | Coverage% | Arch | Strength |",
    "|---|---|---|---|---|---|---|---|---|",
]

for rank, (_, row) in enumerate(top_by_lift.iterrows(), 1):
    lines.append(
        f"| {rank} | `{row['antecedents_str'][:70]}` | {row['support']:.4f} | "
        f"{row['confidence']:.3f} | **{row['lift']:.2f}** | {int(row['n_transactions'])} | "
        f"{row['coverage_of_positives_pct']:.1f}% | {row['archetype']} | {row['strength']} |"
    )

lines += [
    "",
    "---",
    "",
    "## 4. Top 20 Rules by Confidence",
    "",
    "| Rank | Antecedents | Support | Confidence | Lift | n | Coverage% | Arch |",
    "|---|---|---|---|---|---|---|---|",
]

for rank, (_, row) in enumerate(top_by_conf.iterrows(), 1):
    lines.append(
        f"| {rank} | `{row['antecedents_str'][:70]}` | {row['support']:.4f} | "
        f"**{row['confidence']:.3f}** | {row['lift']:.2f} | {int(row['n_transactions'])} | "
        f"{row['coverage_of_positives_pct']:.1f}% | {row['archetype']} |"
    )

lines += [
    "",
    "---",
    "",
    "## 5. Rules per Archetype (Top 5 per Archetype by Lift)",
    "",
]

for arch in ['Peat-driven', 'Wind-driven', 'Drought-driven', 'Human-induced', 'Multi-factor', 'Unclear']:
    sub = rules_esc_filtered[rules_esc_filtered['archetype'] == arch].head(5)
    if len(sub) == 0:
        lines += [f"### {arch}", f"*Tidak ada rules.*", ""]
        continue
    lines += [
        f"### {arch} ({len(sub)} shown, {arch_counts.get(arch,0)} total)",
        "",
        "| Rule | Support | Confidence | Lift | n | Coverage% | Domain |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in sub.iterrows():
        lines.append(
            f"| `{row['antecedents_str'][:65]}` | {row['support']:.4f} | "
            f"{row['confidence']:.3f} | {row['lift']:.2f} | {int(row['n_transactions'])} | "
            f"{row['coverage_of_positives_pct']:.1f}% | {row['domain_validity'][:25]} |"
        )
    lines.append("")

lines += [
    "---",
    "",
    "## 6. Analisis: Apakah Data Mendukung 4 Archetype?",
    "",
]

# Count rules per archetype
arch_support = {
    arch: len(rules_esc_filtered[rules_esc_filtered['archetype'] == arch])
    for arch in ['Peat-driven', 'Wind-driven', 'Drought-driven', 'Human-induced', 'Multi-factor']
}

lines += [
    "| Archetype | Jumlah Rules | Didukung Data? |",
    "|---|---|---|",
]
for arch, cnt in arch_support.items():
    supported = "**YA**" if cnt >= 3 else ("LEMAH" if cnt >= 1 else "TIDAK")
    lines.append(f"| {arch} | {cnt} | {supported} |")

lines += [
    "",
    "**Interpretasi:**",
    "",
]

for arch, cnt in arch_support.items():
    if cnt >= 5:
        lines.append(f"- **{arch}**: TERDUKUNG KUAT ({cnt} rules) — layak masuk Pattern Library")
    elif cnt >= 2:
        lines.append(f"- **{arch}**: TERDUKUNG LEMAH ({cnt} rules) — masuk sebagai exploratory")
    else:
        lines.append(f"- **{arch}**: TIDAK TERDUKUNG ({cnt} rules) — jangan klaim di laporan")

lines += [
    "",
    "---",
    "",
    "## 7. Rules yang Layak Masuk Pattern Library (Step 12)",
    "",
    "> Kriteria: lift > 3 AND confidence > 0.20 AND domain VALID AND NOT dominated",
    "",
    "| Rank | Rule | Lift | Conf | Coverage% | Archetype | Rekomendasi |",
    "|---|---|---|---|---|---|---|",
]

library_candidates = rules_esc_filtered[
    (rules_esc_filtered['lift'] > 3) &
    (rules_esc_filtered['confidence'] > 0.20) &
    (rules_esc_filtered['domain_validity'] == 'VALID') &
    (~rules_esc_filtered['is_dominated'])
].head(15)

for rank, (_, row) in enumerate(library_candidates.iterrows(), 1):
    rec = "PATTERN LIBRARY" if row['strength'] == 'STRONG' else "EXPLORATORY"
    lines.append(
        f"| {rank} | `{row['antecedents_str'][:65]}` | {row['lift']:.2f} | "
        f"{row['confidence']:.3f} | {row['coverage_of_positives_pct']:.1f}% | "
        f"{row['archetype']} | {rec} |"
    )

lines += [
    "",
    "---",
    "",
    "## 8. Rules Lemah / Overfit / Tidak Direkomendasikan",
    "",
    "| Kategori | Detail |",
    "|---|---|",
    f"| Support sangat kecil (< {MIN_SUPPORT_STRONG}) | {n_explor} rules -- tandai EXPLORATORY |",
    "| Domain QUESTIONABLE | Cek antecedents dengan sinyal kontradiktif |",
    "| Lift < 3 | Rules mendekati baseline; tidak bermakna untuk early warning |",
    "| Dominated | Antecedent superset dari rule yang lebih kuat -- redundan |",
    "",
    "---",
    "",
    "## 9. Keterbatasan",
    "",
    "| Keterbatasan | Detail |",
    "|---|---|",
    f"| Hanya 153 kasus positif | ARM dengan imbalance tinggi menghasilkan banyak exploratory rules |",
    "| Diskritisasi tertile | Nilai cutpoint tergantung distribusi dataset; mungkin berbeda pada data baru |",
    "| Tanpa growth_ratio | Fitur yang paling prediktif (STEP9A) sengaja dikecualikan |",
    "| Tidak ada cross-validation | Rules bisa overfit pada dataset ini; butuh validasi eksternal |",
    "| Association ≠ causation | Rules adalah pattern korelasional, bukan mekanisme kausal |",
    "| Max antecedent length = 4 | Rules panjang mungkin terlalu specific/overfitting |",
    "",
    "---",
    "",
    "## 10. Output Files",
    "",
    "| File | Keterangan |",
    "|---|---|",
    f"| step11_association_rules.csv | {len(rules_save)} total valid rules (lift≥2) |",
    f"| step11_top_rules.csv | Top 20 by composite score |",
    f"| step11_archetype_rules.csv | Rules per archetype |",
    "| step11_rule_summary.png | Visualisasi 4 panel |",
    "| STEP11_ASSOCIATION_RULE_MINING_REPORT.md | Laporan ini |",
    "| step11_association_rules.py | Script |",
    "",
    "**File yang tidak diubah:** semua output Step 8–10",
    "",
    "---",
    "Dibuat: Person C -- FireEscalome GEMASTIK XIX/2026",
    "STOP: Menunggu instruksi (tidak lanjut ke Step 12)",
]

with open(WORKSPACE / "STEP11_ASSOCIATION_RULE_MINING_REPORT.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"    Saved: STEP11_ASSOCIATION_RULE_MINING_REPORT.md")

# ================================================================
# 12. COPY SCRIPT
# ================================================================
import shutil
src = Path(r"C:\Users\LENOVO\.gemini\antigravity\brain\ba59b011-539b-43c5-b282-ace6b6f8b48a\scratch\step11_association_rules.py")
dst = WORKSPACE / "step11_association_rules.py"
shutil.copy2(str(src), str(dst))
print(f"    Script: {dst}")

# ================================================================
# 13. FINAL SUMMARY
# ================================================================
print(f"\n{SEP}")
print("STEP 11 -- LAPORAN AKHIR")
print(SEP)
print(f"""
Dataset  : 2888 baris | Eskalasi={n_pos} ({base_rate*100:.2f}%) | Baseline lift={1/base_rate:.1f}
Itemsets : {len(freq_itemsets)}
Rules    : {len(rules_esc_filtered)} valid (lift>=2) | {n_strong} STRONG | {n_explor} EXPLORATORY

Archetype coverage:
""")
for arch, cnt in arch_support.items():
    print(f"  {arch:<20}: {cnt} rules")

print(f"\nTop 5 Rules by Lift:")
for rank, (_, row) in enumerate(rules_esc_filtered.head(5).iterrows(), 1):
    print(f"  #{rank}: {row['antecedents_str'][:60]}")
    print(f"       lift={row['lift']:.2f} | conf={row['confidence']:.3f} | "
          f"n={int(row['n_transactions'])} | cov={row['coverage_of_positives_pct']:.1f}% | {row['archetype']}")

print(f"\nPattern Library candidates: {len(library_candidates)} rules")
print(f"(lift>3, conf>0.20, domain VALID, not dominated)")
print(SEP)
print("STEP 11 SELESAI. Menunggu instruksi berikutnya.")
print(SEP)
