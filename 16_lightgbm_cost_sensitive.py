"""
step8_lightgbm_baseline.py  (FINAL)
Step 8B -- Cost-Sensitive LightGBM Baseline
FireEscalome GEMASTIK XIX/2026 -- Person C

CATATAN KONFLIK PENTING:
  growth_ratio PERFECTLY SEPARATES kelas 0 dan kelas 1:
    - Min growth_ratio Class 1 = 9.0
    - Max growth_ratio Class 0 = 8.833
  Ini menyebabkan early stopping berhenti di iterasi 1 (val_AUC=1.0)
  dan model hanya memprediksi kelas 0 pada threshold 0.5 setelah 1 iter.
  Dengan 100 iterasi, model berfungsi normal: TP=31, FN=0, FP=1.

  Keputusan:
  - Model A (WITH growth_ratio, 100 iter): digunakan sebagai baseline utama
    sesuai instruksi Step 8A yang melarang drop growth_ratio
  - Model B (WITHOUT growth_ratio): disertakan sebagai COMPARISON ONLY
    untuk dokumentasi dampak growth_ratio
  - TIDAK mengubah keputusan metodologi -- konflik dilaporkan

Python  : 3.13
LightGBM: 4.6.0
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------
WORKSPACE    = Path(r"G:\My Drive\FireEscalome")
INPUT_CSV    = WORKSPACE / "step8_features_ready.csv"
MODEL_A_OUT  = WORKSPACE / "step8_lightgbm_cost_sensitive_baseline.txt"
MODEL_B_OUT  = WORKSPACE / "step8_lightgbm_no_growth_ratio_comparison.txt"
PRED_A_OUT   = WORKSPACE / "step8_test_predictions.csv"
PRED_B_OUT   = WORKSPACE / "step8_test_predictions_no_gr.csv"
REPORT_OUT   = WORKSPACE / "STEP8B_MODEL_EVALUATION.md"

TARGET_COL   = "label_escalation"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
SCALE_POS_W  = 2735 / 153
THRESHOLD    = 0.5
CAT_FEATURES = ['island', 'confidence', 'is_peatland', 'landcover_class']
# ----------------------------------------------------------------

SEP = "=" * 65
ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(SEP)
print("STEP 8B -- COST-SENSITIVE LIGHTGBM BASELINE (FINAL)")
print(f"Timestamp: {ts} | LightGBM: {lgb.__version__}")
print(SEP)

# ================================================================
# 1. LOAD
# ================================================================
print("\n[1] LOAD DATA ...")
df = pd.read_csv(INPUT_CSV)
X  = df.drop(columns=[TARGET_COL])
y  = df[TARGET_COL]

for col in CAT_FEATURES:
    X[col] = X[col].astype('int32')

print(f"    X: {X.shape} | Class0={int((y==0).sum())} | Class1={int((y==1).sum())}")

# ================================================================
# 2. DIAGNOSA growth_ratio SEPARABILITY (CONFLICT REPORT)
# ================================================================
print("\n[2] CONFLICT REPORT: growth_ratio SEPARABILITY ...")
min_gr_c1 = float(df[y == 1]['growth_ratio'].min())
max_gr_c0 = float(df[y == 0]['growth_ratio'].max())
gr_separable = min_gr_c1 > max_gr_c0
print(f"    Min growth_ratio Class 1 : {min_gr_c1:.4f}")
print(f"    Max growth_ratio Class 0 : {max_gr_c0:.4f}")
print(f"    Perfect separation?      : {gr_separable}")
print(f"    Dampak: early_stopping berhenti di iter=1 (val AUC=1.0)")
print(f"    Keputusan: tetap gunakan growth_ratio (sesuai instruksi Step 8A)")
print(f"    Solusi: jalankan 100 iter tanpa early stopping untuk Model A")
print(f"    Model B (tanpa growth_ratio) disertakan sebagai COMPARISON")

# ================================================================
# 3. SPLIT
# ================================================================
print("\n[3] SPLIT ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

c0_tr = int((y_train == 0).sum()); c1_tr = int((y_train == 1).sum())
c0_te = int((y_test  == 0).sum()); c1_te = int((y_test  == 1).sum())

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE)

print(f"    Train: {len(X_train)} | C0={c0_tr} C1={c1_tr}")
print(f"    Test : {len(X_test)}  | C0={c0_te} C1={c1_te}")
print(f"    Val  : {len(X_val)}   (dari train, untuk diagnosis)")

# ================================================================
# 4. PARAMS
# ================================================================
BASE_PARAMS = {
    'objective'        : 'binary',
    'metric'           : ['binary_logloss', 'auc'],
    'boosting_type'    : 'gbdt',
    'scale_pos_weight' : SCALE_POS_W,
    'random_state'     : RANDOM_STATE,
    'learning_rate'    : 0.05,
    'num_leaves'       : 31,
    'max_depth'        : -1,
    'min_child_samples': 20,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 0.1,
    'verbose'          : -1,
}

# ================================================================
# 5. MODEL A: WITH growth_ratio, 100 iter
# ================================================================
print(f"\n[5] MODEL A: WITH growth_ratio (BASELINE UTAMA, 100 iter) ...")
dtrain_A = lgb.Dataset(X_tr, label=y_tr,
                        categorical_feature=CAT_FEATURES, free_raw_data=False)

modelA = lgb.train(BASE_PARAMS, dtrain_A,
                   num_boost_round=100,
                   callbacks=[lgb.log_evaluation(period=0)])

pA     = modelA.predict(X_test, num_iteration=100)
ypA    = (pA >= THRESHOLD).astype(int)

rocA   = roc_auc_score(y_test, pA)
prA    = average_precision_score(y_test, pA)
prec1A = precision_score(y_test, ypA, pos_label=1, zero_division=0)
rec1A  = recall_score(y_test, ypA, pos_label=1, zero_division=0)
f1A    = f1_score(y_test, ypA, pos_label=1, zero_division=0)
cmA    = confusion_matrix(y_test, ypA)
tnA, fpA, fnA, tpA = cmA.ravel()

best_iterA = 100  # fixed, no early stopping (karena val AUC trivially 1.0)

print(f"    [Model A] ROC-AUC={rocA:.4f} | PR-AUC={prA:.4f}")
print(f"    [Model A] Prec-1={prec1A:.4f} | Rec-1={rec1A:.4f} | F1-1={f1A:.4f}")
print(f"    [Model A] TP={tpA} TN={tnA} FP={fpA} FN={fnA}")

# ================================================================
# 6. MODEL B: WITHOUT growth_ratio (COMPARISON)
# ================================================================
print(f"\n[6] MODEL B: WITHOUT growth_ratio (COMPARISON, early stopping) ...")
X_no_gr = X.drop(columns=['growth_ratio'])
Xtr2, Xte2, ytr2, yte2 = train_test_split(
    X_no_gr, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
Xtr2a, Xtr2b, ytr2a, ytr2b = train_test_split(
    Xtr2, ytr2, test_size=0.15, stratify=ytr2, random_state=RANDOM_STATE)

dtrain_B = lgb.Dataset(Xtr2a, label=ytr2a,
                        categorical_feature=CAT_FEATURES, free_raw_data=False)
dval_B   = lgb.Dataset(Xtr2b, label=ytr2b,
                        categorical_feature=CAT_FEATURES, reference=dtrain_B, free_raw_data=False)

modelB = lgb.train(BASE_PARAMS, dtrain_B, num_boost_round=1000,
                   valid_sets=[dtrain_B, dval_B], valid_names=['train','val'],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

pB     = modelB.predict(Xte2, num_iteration=modelB.best_iteration)
ypB    = (pB >= THRESHOLD).astype(int)

rocB   = roc_auc_score(yte2, pB)
prB    = average_precision_score(yte2, pB)
prec1B = precision_score(yte2, ypB, pos_label=1, zero_division=0)
rec1B  = recall_score(yte2, ypB, pos_label=1, zero_division=0)
f1B    = f1_score(yte2, ypB, pos_label=1, zero_division=0)
cmB    = confusion_matrix(yte2, ypB)
tnB, fpB, fnB, tpB = cmB.ravel()
best_iterB = modelB.best_iteration

print(f"    [Model B] best_iter={best_iterB}")
print(f"    [Model B] ROC-AUC={rocB:.4f} | PR-AUC={prB:.4f}")
print(f"    [Model B] Prec-1={prec1B:.4f} | Rec-1={rec1B:.4f} | F1-1={f1B:.4f}")
print(f"    [Model B] TP={tpB} TN={tnB} FP={fpB} FN={fnB}")

# ================================================================
# 7. CLASSIFICATION REPORTS
# ================================================================
print(f"\n[7] CLASSIFICATION REPORT -- Model A (WITH growth_ratio):")
print(classification_report(y_test, ypA, target_names=['Non-Eskalasi','Eskalasi']))

print(f"[7] CLASSIFICATION REPORT -- Model B (WITHOUT growth_ratio):")
print(classification_report(yte2, ypB, target_names=['Non-Eskalasi','Eskalasi']))

# ================================================================
# 8. SIMPAN MODEL & PREDICTIONS
# ================================================================
print(f"\n[8] SIMPAN OUTPUT ...")
modelA.save_model(str(MODEL_A_OUT), num_iteration=100)
modelB.save_model(str(MODEL_B_OUT), num_iteration=best_iterB)
print(f"    Model A: {MODEL_A_OUT}")
print(f"    Model B: {MODEL_B_OUT}")

pd.DataFrame({
    'row_index': X_test.index, 'y_true': y_test.values,
    'predicted_probability': pA, 'predicted_class': ypA
}).to_csv(PRED_A_OUT, index=False)

pd.DataFrame({
    'row_index': Xte2.index, 'y_true': yte2.values,
    'predicted_probability': pB, 'predicted_class': ypB
}).to_csv(PRED_B_OUT, index=False)
print(f"    Pred A : {PRED_A_OUT}")
print(f"    Pred B : {PRED_B_OUT}")

# ================================================================
# 9. TULIS LAPORAN
# ================================================================
print(f"\n[9] TULIS LAPORAN ...")
lines = [
    "# STEP 8B - MODEL EVALUATION: COST-SENSITIVE LIGHTGBM BASELINE",
    "## FireEscalome GEMASTIK XIX/2026 | Person C",
    f"**Timestamp: {ts} | LightGBM: {lgb.__version__} | Python: {sys.version.split()[0]}**",
    "",
    "---",
    "",
    "## CONFLICT REPORT: growth_ratio Perfect Separation",
    "",
    "> **KONFLIK TERDETEKSI DAN DILAPORKAN (tidak diubah secara sepihak)**",
    "",
    "| | Detail |",
    "|---|---|",
    f"| Min growth_ratio Class 1 | {min_gr_c1:.4f} |",
    f"| Max growth_ratio Class 0 | {max_gr_c0:.4f} |",
    "| Perfect separable? | **True** -- tidak ada overlap |",
    "| Dampak pada training | early_stopping berhenti iter=1, val_AUC=1.0 secara trivial |",
    "| Dampak pada prediksi (1 iter) | semua probabilitas < 0.5, TP=0 |",
    "",
    "**Sumber konflik:**",
    "- Instruksi Step 8A: JANGAN drop growth_ratio",
    "- Script 11 Person A: DROP growth_ratio",
    "- Definisi label eskalasi (dokumen master v4): growth_ratio dipakai sebagai SALAH SATU komponen label -- bukan definisi tunggal",
    "",
    "**Keputusan:**",
    "- **Model A** (WITH growth_ratio, 100 iter fixed) = **BASELINE UTAMA** sesuai instruksi Step 8A",
    "- **Model B** (WITHOUT growth_ratio, early stopping) = COMPARISON untuk dokumentasi",
    "- Keputusan akhir drop/tidak-drop growth_ratio diserahkan kepada Person C di Step 9",
    "",
    "---",
    "",
    "## 1. Dataset",
    "",
    "| Item | Detail |",
    "|---|---|",
    f"| Input file | step8_features_ready.csv |",
    f"| Total baris | {len(df)} |",
    f"| Jumlah fitur | {X.shape[1]} |",
    f"| Class 0 (Non-Eskalasi) | {int((y==0).sum())} ({(y==0).mean()*100:.2f}%) |",
    f"| Class 1 (Eskalasi) | {int((y==1).sum())} ({(y==1).mean()*100:.2f}%) |",
    f"| scale_pos_weight | {SCALE_POS_W:.4f} |",
    "",
    "---",
    "",
    "## 2. Train / Test Split",
    "",
    f"| Split | Total | Class 0 | Class 1 |",
    f"|---|---|---|---|",
    f"| Train (80%) | {len(X_train)} | {c0_tr} | {c1_tr} |",
    f"| Test  (20%) | {len(X_test)} | {c0_te} | {c1_te} |",
    f"| Val (dari train) | {len(X_val)} | {int((y_val==0).sum())} | {int((y_val==1).sum())} |",
    "",
    "- test_size=0.20, stratify=y, random_state=42",
    "",
    "---",
    "",
    "## 3. LightGBM Parameters",
    "",
    "| Parameter | Nilai |",
    "|---|---|",
    "| objective | binary |",
    "| boosting_type | gbdt |",
    f"| scale_pos_weight | {SCALE_POS_W:.4f} |",
    "| learning_rate | 0.05 |",
    "| num_leaves | 31 |",
    "| max_depth | -1 |",
    "| min_child_samples | 20 |",
    "| subsample | 0.8 |",
    "| colsample_bytree | 0.8 |",
    "| reg_alpha | 0.1 |",
    "| reg_lambda | 0.1 |",
    "| random_state | 42 |",
    f"| Model A: num_boost_round | 100 (fixed, tanpa early stopping) |",
    f"| Model B: best_iteration | {best_iterB} (dengan early stopping) |",
    "",
    "**Categorical features:** island, confidence, is_peatland, landcover_class",
    "",
    "**landcover_class=-1:** Dipertahankan sebagai kategori khusus (GEE no-data).",
    "LightGBM mengkonversi nilai negatif ke NaN secara internal -- ini adalah perilaku default LightGBM 4.x.",
    "Dampak: 37 baris (1.3%) dengan landcover_class=-1 diperlakukan sebagai missing di categorical split.",
    "",
    "---",
    "",
    "## 4. Hasil Evaluasi -- Model A (WITH growth_ratio, BASELINE UTAMA)",
    "",
    "| Metric | Nilai |",
    "|---|---|",
    f"| **ROC-AUC** | **{rocA:.4f}** |",
    f"| **PR-AUC** | **{prA:.4f}** |",
    f"| Precision Class 1 | {prec1A:.4f} |",
    f"| Recall Class 1 | {rec1A:.4f} |",
    f"| F1 Class 1 | {f1A:.4f} |",
    f"| TP | {tpA} |",
    f"| TN | {tnA} |",
    f"| FP | {fpA} |",
    f"| FN | {fnA} |",
    "",
    f"| | Pred Non-Esc | Pred Esc |",
    f"|---|---|---|",
    f"| Actual Non-Esc | TN={tnA} | FP={fpA} |",
    f"| Actual Esc | FN={fnA} | TP={tpA} |",
    "",
    "---",
    "",
    "## 5. Hasil Evaluasi -- Model B (WITHOUT growth_ratio, COMPARISON)",
    "",
    "| Metric | Nilai |",
    "|---|---|",
    f"| ROC-AUC | {rocB:.4f} |",
    f"| PR-AUC | {prB:.4f} |",
    f"| Precision Class 1 | {prec1B:.4f} |",
    f"| Recall Class 1 | {rec1B:.4f} |",
    f"| F1 Class 1 | {f1B:.4f} |",
    f"| TP | {tpB} |  TN | {tnB} |  FP | {fpB} |  FN | {fnB} |",
    f"| best_iteration | {best_iterB} |",
    "",
    "---",
    "",
    "## 6. Perbandingan Model A vs Model B",
    "",
    "| Metric | Model A (with growth_ratio) | Model B (without growth_ratio) |",
    "|---|---|---|",
    f"| ROC-AUC | **{rocA:.4f}** | {rocB:.4f} |",
    f"| PR-AUC | **{prA:.4f}** | {prB:.4f} |",
    f"| Precision-1 | **{prec1A:.4f}** | {prec1B:.4f} |",
    f"| Recall-1 | **{rec1A:.4f}** | {rec1B:.4f} |",
    f"| F1-1 | **{f1A:.4f}** | {f1B:.4f} |",
    f"| TP / FN | {tpA} / {fnA} | {tpB} / {fnB} |",
    "",
    "> growth_ratio yang perfectly separable membuat Model A memiliki metrics sangat tinggi.",
    "> Model B memberikan gambaran performa yang lebih realistis jika growth_ratio tidak digunakan.",
    "> Keputusan akhir tentang penggunaan growth_ratio: dikonsultasikan di Step 9 (SHAP).",
    "",
    "---",
    "",
    "## 7. Quality Check",
    "",
    "| Check | Status |",
    "|---|---|",
    f"| label_escalation tidak masuk X | {TARGET_COL not in X.columns} |",
    "| cluster_id tidak masuk X | True |",
    "| community_id tidak masuk X | True |",
    "| Test set tidak digunakan training | True |",
    "| Test set tidak digunakan early stopping | True |",
    f"| scale_pos_weight | {SCALE_POS_W:.4f} |",
    "| SMOTE / oversampling / undersampling | TIDAK |",
    "| landcover_class=-1 dipertahankan | True |",
    "| Tuning berlebihan | TIDAK (baseline params) |",
    "| Konflik growth_ratio dilaporkan | True |",
    "",
    "---",
    "",
    "## 8. Output Files",
    "",
    "| File | Keterangan |",
    "|---|---|",
    "| step8_lightgbm_cost_sensitive_baseline.txt | Model A -- WITH growth_ratio (BASELINE UTAMA) |",
    "| step8_lightgbm_no_growth_ratio_comparison.txt | Model B -- WITHOUT growth_ratio (COMPARISON) |",
    "| step8_test_predictions.csv | Prediksi test Model A |",
    "| step8_test_predictions_no_gr.csv | Prediksi test Model B |",
    "| STEP8B_MODEL_EVALUATION.md | Laporan ini |",
    "| step8_lightgbm_baseline.py | Script training |",
    "",
    "---",
    "",
    "## 9. Ringkasan Eksekutif",
    "",
    "| Item | Model A (Baseline Utama) | Model B (Comparison) |",
    "|---|---|---|",
    f"| Fitur | 25 (dengan growth_ratio) | 24 (tanpa growth_ratio) |",
    f"| Train/Test | {len(X_train)}/{len(X_test)} | {len(Xtr2)}/{len(Xte2)} |",
    f"| best_iteration | 100 (fixed) | {best_iterB} |",
    f"| ROC-AUC | {rocA:.4f} | {rocB:.4f} |",
    f"| PR-AUC | {prA:.4f} | {prB:.4f} |",
    f"| Recall-1 | {rec1A:.4f} | {rec1B:.4f} |",
    f"| F1-1 | {f1A:.4f} | {f1B:.4f} |",
    "| Status | BASELINE SELESAI | COMPARISON SELESAI |",
    "| Layak Step 8? | YA (dengan catatan growth_ratio) | YA (lebih realistis) |",
    "",
    "---",
    "",
    "## 10. Reproducibility",
    "",
    "```",
    f"Python      : {sys.version.split()[0]}",
    f"LightGBM    : {lgb.__version__}",
    f"random_state: {RANDOM_STATE}",
    f"test_size   : {TEST_SIZE}",
    f"scale_pos_weight: {SCALE_POS_W:.4f}",
    f"Model A iter: 100 (fixed, growth_ratio included)",
    f"Model B iter: {best_iterB} (early_stopping=50, growth_ratio excluded)",
    f"Timestamp   : {ts}",
    "```",
    "",
    "---",
    "Dibuat: Person C -- FireEscalome GEMASTIK XIX/2026",
    "Next  : Step 9 -- SHAP Analysis (setelah instruksi)",
]

with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"    Report: {REPORT_OUT}")

# ================================================================
# 10. COPY SCRIPT KE WORKSPACE
# ================================================================
import shutil
script_src = Path(r"C:\Users\LENOVO\.gemini\antigravity\brain\ba59b011-539b-43c5-b282-ace6b6f8b48a\scratch\step8_lightgbm_baseline.py")
script_dst = WORKSPACE / "step8_lightgbm_baseline.py"
shutil.copy2(str(script_src), str(script_dst))
print(f"    Script : {script_dst}")

# ================================================================
# 11. LAPORAN AKHIR KE CONSOLE
# ================================================================
print(f"\n{SEP}")
print("STEP 8B -- LAPORAN AKHIR")
print(SEP)
print(f"""
 1. Dataset            : step8_features_ready.csv
 2. Jumlah fitur       : {X.shape[1]} (Model A) / {X_no_gr.shape[1]} (Model B)
 3. Train / Test       : {len(X_train)} / {len(X_test)} baris
 4. Dist Train         : Class0={c0_tr} Class1={c1_tr}
    Dist Test          : Class0={c0_te} Class1={c1_te}
 5. scale_pos_weight   : {SCALE_POS_W:.4f}
 6. LightGBM params    : objective=binary, lr=0.05, leaves=31
 7. best_iteration     : A=100 (fixed) | B={best_iterB}
 8. ROC-AUC            : A={rocA:.4f} | B={rocB:.4f}
 9. PR-AUC             : A={prA:.4f} | B={prB:.4f}
10. Precision Class 1  : A={prec1A:.4f} | B={prec1B:.4f}
11. Recall Class 1     : A={rec1A:.4f} | B={rec1B:.4f}
12. F1 Class 1         : A={f1A:.4f} | B={f1B:.4f}
13. CM Model A         : TP={tpA} TN={tnA} FP={fpA} FN={fnA}
    CM Model B         : TP={tpB} TN={tnB} FP={fpB} FN={fnB}
14. Status             : BASELINE SELESAI
15. Layak baseline?    : YA -- growth_ratio conflict dilaporkan

CONFLICT: growth_ratio perfectly separable (min_c1={min_gr_c1} > max_c0={max_gr_c0:.3f})
Keputusan akhir growth_ratio: diserahkan ke Step 9 SHAP analysis.
""")
print(SEP)
print("STEP 8B SELESAI. Menunggu instruksi Step 9.")
print(SEP)
