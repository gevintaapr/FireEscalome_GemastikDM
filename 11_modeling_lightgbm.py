import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from config import PROCESSED_DIR

def main():
    print("============================================================")
    print("11_MODELING_LIGHTGBM.PY")
    print("Tahap 4: Baseline Classification Model untuk Person B")
    print("============================================================\n")

    input_file = PROCESSED_DIR / "tabular_master_final.csv"
    if not os.path.exists(input_file):
        print(f"[ERROR] Data {input_file} tidak ditemukan!")
        return

    print("1. Memuat Dataset...")
    df = pd.read_csv(input_file)
    
    # Define Target and Features
    target = 'label_escalation'
    drop_cols = ['cluster_id', 'latitude', 'longitude', 'island', 'acq_date', 'latitude_weather', 'longitude_weather', 'region']
    
    # Hapus kolom yang tidak relevan dengan pemodelan (ID, lokasi mutlak)
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop + [target])
    y = df[target]
    
    print(f"   -> Jumlah fitur (X): {X.shape[1]}")
    print(f"   -> Distribusi Kelas Target (Y): \n{y.value_counts()}")

    print("\n2. Membagi Data (Train 80% / Test 20%)...")
    # Stratify penting agar persentase api eskalasi (kelas 1) sama di train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Hitung rasio imbalance untuk scale_pos_weight
    # ratio = jumlah negatif / jumlah positif
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_weight = num_neg / num_pos if num_pos > 0 else 1
    
    print(f"   -> Rasio Imbalance Kelas 0/1: {scale_weight:.2f}")

    print("\n3. Melatih Model LightGBM...")
    model = lgb.LGBMClassifier(
        random_state=42,
        scale_pos_weight=scale_weight,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        verbose=-1 # menyembunyikan log lgbm yang bising
    )
    
    model.fit(X_train, y_train)
    
    print("\n4. Evaluasi Model pada Data Uji (Testing)...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # 5. Visualisasi
    print("\n5. Menyimpan Grafik Visualisasi (Plot)...")
    vis_dir = PROCESSED_DIR / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # A. Confusion Matrix
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Eskalasi', 'Eskalasi'], yticklabels=['Non-Eskalasi', 'Eskalasi'])
    plt.title('Confusion Matrix - LightGBM Baseline')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    cm_path = vis_dir / 'confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    
    # B. Feature Importance
    plt.figure(figsize=(10, 8))
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance, hue='Feature', palette='viridis', legend=False)
    plt.title('Feature Importance (LightGBM)')
    plt.tight_layout()
    fi_path = vis_dir / 'feature_importance.png'
    plt.savefig(fi_path)
    plt.close()
    
    print(f"   -> Visualisasi disimpan di folder: {vis_dir.name}")
    print("\nSELESAI! Baseline model siap digunakan atau dieksplorasi lebih jauh oleh Person B.")

if __name__ == "__main__":
    main()
