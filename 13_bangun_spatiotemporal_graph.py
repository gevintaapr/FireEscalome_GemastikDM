"""
13_bangun_spatiotemporal_graph.py
Fase 3 (INTI): Konstruksi Spatio-Temporal Graph G = (V, E)

Step 7 dari pipeline Person B:
  - Node = setiap cluster kebakaran dari cluster_master_features.csv
  - Edge = dua cluster terhubung jika jarak ≤ 5km DAN selisih waktu 0 < Δt ≤ 3 hari
  - Wij  = gabungan 4 komponen bobot edge:
      1. Spatial Decay       : exp(-α × d_km)
      2. Wind Alignment      : cos(θ) dimana θ = sudut antara arah angin vs bearing i→j
      3. Slope Direction      : kontribusi kemiringan lereng node tujuan
      4. Peatland Continuity  : bonus koneksi jika kedua node di lahan gambut

Output:
  - data_processed/spatiotemporal_graph.pkl  (graf NetworkX lengkap)
  - data_processed/edge_statistics.csv       (tabel breakdown semua edge)

Jalankan: python 13_bangun_spatiotemporal_graph.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pickle
import math
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

# ==============================================================
# KONFIGURASI
# ==============================================================
try:
    from config import PROCESSED_DIR
except ImportError:
    from pathlib import Path
    PROCESSED_DIR = Path("G:/My Drive/gemastik/karhutla/data_processed")

# Input: file hasil merge milik Person B (BUKAN file asli Gevi)
PROCESSED_B_DIR = Path(str(PROCESSED_DIR).replace("data_processed", "data_processed_B"))
INPUT_FILE = PROCESSED_B_DIR / "cluster_master_PersonB.csv"

# Output: simpan ke folder Person B sendiri
OUTPUT_GRAPH = PROCESSED_B_DIR / "spatiotemporal_graph.pkl"
OUTPUT_EDGE_CSV = PROCESSED_B_DIR / "edge_statistics.csv"

# --- Parameter Graf ---
MAX_SPATIAL_DISTANCE_KM = 5.0     # Jarak maksimum antar-node untuk membentuk edge
MAX_TEMPORAL_DISTANCE_DAYS = 3    # Jendela waktu maksimum (hari)
EARTH_RADIUS_KM = 6371.0         # Radius bumi untuk haversine

# --- Parameter Bobot Edge (α, β, γ, δ) ---
# Bobot relatif tiap komponen terhadap Wij total (harus sum = 1.0)
ALPHA_SPATIAL = 0.25   # Spatial Decay
BETA_WIND = 0.35       # Wind Alignment (faktor terpenting untuk fire spread)
GAMMA_SLOPE = 0.15     # Slope Direction
DELTA_PEAT = 0.25      # Peatland Continuity

# --- Parameter decay ---
SPATIAL_DECAY_RATE = 0.5   # α dalam exp(-α × d_km), makin besar = makin cepat melemah

# Nama kolom cuaca angin (akan dicek saat runtime)
WIND_SPEED_CANDIDATES = [
    "windspeed_10m_max", "windspeed_10m_max_weather",
    "wind_speed_10m_max", "windspeed"
]
WIND_DIR_CANDIDATES = [
    "winddirection_10m_dominant", "winddirection_10m_dominant_weather",
    "wind_direction_10m_dominant", "winddirection"
]
# Kolom wind_alignment_score pre-kalkulasi (dari pipeline ERA5 Gevi)
# Digunakan langsung sebagai nilai cos(theta) jika winddirection tidak ada
WIND_ALIGNMENT_PRECALC = "wind_alignment_score"


# ==============================================================
# UTILITAS MATEMATIKA
# ==============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Hitung jarak haversine antara dua titik koordinat dalam km."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def bearing_degrees(lat1, lon1, lat2, lon2):
    """
    Hitung bearing (arah kompas) dari titik i ke titik j dalam derajat [0, 360).
    0° = utara, 90° = timur, 180° = selatan, 270° = barat.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360


def angular_difference(angle1_deg, angle2_deg):
    """Hitung selisih terkecil antara dua sudut (derajat), hasil [0, 180]."""
    diff = abs(angle1_deg - angle2_deg) % 360
    return diff if diff <= 180 else 360 - diff


# ==============================================================
# BAGIAN A: DEFINISI NODE
# ==============================================================

def build_nodes(G, df):
    """Tambahkan semua cluster sebagai node ke graf dengan atribut lengkap."""
    print("\n[A] Membangun Node...")

    for _, row in df.iterrows():
        node_id = int(row["cluster_id"])
        attrs = {
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "year": int(row["year"]),
            "start_day_of_year": int(row["start_day_of_year"]),
            "label_escalation": int(row.get("label_escalation", 0)),
            "growth_ratio": float(row.get("growth_ratio", 0)),
            "is_peatland": int(row.get("is_peatland", 0)),
            "total_hotspots": int(row.get("total_hotspots", 0)),
        }

        # Kolom opsional — tambahkan kalau ada
        for col in ["slope", "elevation", "landcover_class",
                     "island", "confidence"]:
            if col in df.columns and pd.notna(row.get(col)):
                attrs[col] = row[col]

        G.add_node(node_id, **attrs)

    print(f"    Total node: {G.number_of_nodes():,}")
    return G


# ==============================================================
# BAGIAN B: PEMBENTUKAN EDGE (Spatial + Temporal Filter)
# ==============================================================

def find_candidate_edges(df):
    """
    Temukan pasangan node yang memenuhi syarat jarak ≤ MAX_SPATIAL_DISTANCE_KM
    dan selisih waktu 0 < Δt ≤ MAX_TEMPORAL_DISTANCE_DAYS.

    Menggunakan KDTree untuk pre-filtering spasial (menghindari O(N²) penuh).
    """
    print("\n[B] Mencari pasangan edge kandidat (spatial + temporal filter)...")

    # Pre-filter spatial menggunakan KDTree pada koordinat derajat
    # Konversi jarak km ke perkiraan derajat (1° ≈ 111km di ekuator)
    approx_deg = MAX_SPATIAL_DISTANCE_KM / 111.0 * 1.5  # buffer 50% untuk safety

    coords = df[["latitude", "longitude"]].values
    tree = cKDTree(coords)

    # Cari semua pasangan dalam radius derajat
    pairs = tree.query_pairs(r=approx_deg)
    print(f"    Pasangan dalam radius spatial awal: {len(pairs):,}")

    # Filter ketat: haversine + temporal
    valid_edges = []
    for i, j in pairs:
        row_i = df.iloc[i]
        row_j = df.iloc[j]

        # Harus tahun yang sama (kebakaran di tahun berbeda bukan satu event chain)
        if int(row_i["year"]) != int(row_j["year"]):
            continue

        # Hitung jarak haversine sebenarnya
        d_km = haversine_km(
            row_i["latitude"], row_i["longitude"],
            row_j["latitude"], row_j["longitude"]
        )
        if d_km > MAX_SPATIAL_DISTANCE_KM:
            continue

        # Hitung selisih waktu (hari)
        dt = abs(int(row_i["start_day_of_year"]) - int(row_j["start_day_of_year"]))
        if dt < 1 or dt > MAX_TEMPORAL_DISTANCE_DAYS:
            continue

        # Tentukan arah: node yang muncul lebih dulu = source
        if row_i["start_day_of_year"] <= row_j["start_day_of_year"]:
            source_idx, target_idx = i, j
        else:
            source_idx, target_idx = j, i

        valid_edges.append((source_idx, target_idx, d_km, dt))

    print(f"    Edge valid setelah filter haversine + temporal: {len(valid_edges):,}")
    return valid_edges


# ==============================================================
# BAGIAN C: KALKULASI BOBOT EDGE Wij (4 Komponen)
# ==============================================================

def compute_edge_weights(G, df, valid_edges, wind_speed_col, wind_dir_col):
    """
    Untuk setiap edge yang valid, hitung Wij dari 4 komponen:
      1. Spatial Decay:      exp(-α × d_km)
      2. Wind Alignment:     max(0, cos(θ)) dimana θ = |arah_angin - bearing_i→j|
      3. Slope Direction:    sigmoid normalisasi dari slope node target
      4. Peatland Continuity: 1.0 jika keduanya gambut, 0.5 jika salah satu, 0.0 jika tidak
    """
    print("\n[C] Menghitung bobot edge Wij (4 komponen)...")

    edge_records = []
    edges_added = 0

    for source_idx, target_idx, d_km, dt in valid_edges:
        src = df.iloc[source_idx]
        tgt = df.iloc[target_idx]

        src_id = int(src["cluster_id"])
        tgt_id = int(tgt["cluster_id"])

        # --- Komponen 1: Spatial Decay ---
        spatial_decay = math.exp(-SPATIAL_DECAY_RATE * d_km)

        # --- Komponen 2: Wind Alignment (cos θ dari ERA5) ---
        # PRIORITAS: gunakan wind_alignment_score pre-kalkulasi Gevi dari ERA5 (u10, v10)
        # Ini lebih akurat daripada menghitung ulang dari winddirection_dominant
        wind_align_precalc = src.get(WIND_ALIGNMENT_PRECALC, np.nan) if WIND_ALIGNMENT_PRECALC else np.nan

        if pd.notna(wind_align_precalc):
            # Gunakan langsung: ini sudah nilai cos(θ) dari pipeline ERA5 Gevi
            # Normalisasi ke [0,1] karena nilai asli bisa negatif jika angin berlawanan
            wind_alignment = max(0.0, float(wind_align_precalc))
        else:
            # Fallback: hitung dari winddirection_dominant jika tersedia
            brng = bearing_degrees(
                src["latitude"], src["longitude"],
                tgt["latitude"], tgt["longitude"]
            )
            wind_dir = src.get(wind_dir_col, np.nan) if wind_dir_col else np.nan
            wind_spd = src.get(wind_speed_col, 0) if wind_speed_col else 0

            if pd.notna(wind_dir) and pd.notna(wind_spd) and wind_spd > 0:
                theta = angular_difference(wind_dir, brng)
                wind_alignment = max(0.0, math.cos(math.radians(theta)))
                wind_factor = min(wind_spd / 20.0, 1.0)
                wind_alignment *= wind_factor
            else:
                wind_alignment = 0.5  # netral jika tidak ada data sama sekali

        # --- Komponen 3: Slope Direction ---
        slope_target = tgt.get("slope", 0) if "slope" in df.columns else 0
        if pd.isna(slope_target):
            slope_target = 0
        # Normalisasi slope: sigmoid-like, 0°→0, 15°→0.5, 45°→0.93
        slope_direction = 1.0 - math.exp(-0.05 * slope_target)

        # --- Komponen 4: Peatland Continuity ---
        peat_src = int(src.get("is_peatland", 0))
        peat_tgt = int(tgt.get("is_peatland", 0))
        if peat_src == 1 and peat_tgt == 1:
            peatland_continuity = 1.0   # Keduanya gambut — koneksi bawah tanah kuat
        elif peat_src == 1 or peat_tgt == 1:
            peatland_continuity = 0.5   # Salah satu gambut
        else:
            peatland_continuity = 0.0   # Tidak ada gambut

        # --- Hitung Wij Total (weighted sum) ---
        wij = (
            ALPHA_SPATIAL * spatial_decay
            + BETA_WIND * wind_alignment
            + GAMMA_SLOPE * slope_direction
            + DELTA_PEAT * peatland_continuity
        )

        # Tentukan faktor dominan
        components = {
            "spatial_decay": spatial_decay,
            "wind_alignment": wind_alignment,
            "slope_direction": slope_direction,
            "peatland_continuity": peatland_continuity,
        }
        dominant_factor = max(components, key=components.get)

        # Tambahkan edge ke graf
        G.add_edge(
            src_id, tgt_id,
            weight=wij,
            distance_km=d_km,
            delta_days=dt,
            spatial_decay=spatial_decay,
            wind_alignment=wind_alignment,
            slope_direction=slope_direction,
            peatland_continuity=peatland_continuity,
            dominant_factor=dominant_factor,
        )
        edges_added += 1

        # Simpan record untuk CSV
        edge_records.append({
            "source_id": src_id,
            "target_id": tgt_id,
            "distance_km": round(d_km, 3),
            "delta_days": dt,
            "weight_total": round(wij, 4),
            "spatial_decay": round(spatial_decay, 4),
            "wind_alignment": round(wind_alignment, 4),
            "slope_direction": round(slope_direction, 4),
            "peatland_continuity": round(peatland_continuity, 4),
            "dominant_factor": dominant_factor,
            "source_label": int(src.get("label_escalation", 0)),
            "target_label": int(tgt.get("label_escalation", 0)),
        })

    print(f"    Total edge ditambahkan ke graf: {edges_added:,}")
    return G, pd.DataFrame(edge_records)


# ==============================================================
# BAGIAN D: DEKOMPOSISI & SIMPAN
# ==============================================================

def save_outputs(G, df_edges):
    """Simpan graf dan statistik edge ke folder Person B."""
    print("\n[D] Menyimpan output...")

    os.makedirs(PROCESSED_B_DIR, exist_ok=True)

    # Simpan graf
    with open(OUTPUT_GRAPH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    Graf tersimpan: {OUTPUT_GRAPH}")

    # Simpan edge statistics
    if len(df_edges) > 0:
        df_edges.to_csv(OUTPUT_EDGE_CSV, index=False)
        print(f"    Edge statistics tersimpan: {OUTPUT_EDGE_CSV}")

    return


def print_summary(G, df_edges):
    """Cetak ringkasan graf yang dibangun."""
    print("\n" + "=" * 65)
    print("RINGKASAN GRAF SPATIO-TEMPORAL")
    print("=" * 65)

    print(f"\n  Node (cluster): {G.number_of_nodes():,}")
    print(f"  Edge (koneksi): {G.number_of_edges():,}")

    if G.number_of_nodes() > 0:
        # Statistik node
        escalated = sum(1 for _, d in G.nodes(data=True) if d.get("label_escalation") == 1)
        peat_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("is_peatland") == 1)
        isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)

        print(f"\n  Node eskalasi (label=1): {escalated:,}")
        print(f"  Node di gambut         : {peat_nodes:,}")
        print(f"  Node terisolasi (0 edge): {isolated:,}")

    if len(df_edges) > 0:
        print(f"\n  Statistik Bobot Edge (Wij):")
        print(f"    Mean   : {df_edges['weight_total'].mean():.4f}")
        print(f"    Median : {df_edges['weight_total'].median():.4f}")
        print(f"    Min    : {df_edges['weight_total'].min():.4f}")
        print(f"    Max    : {df_edges['weight_total'].max():.4f}")

        print(f"\n  Distribusi Faktor Dominan per Edge:")
        dom_counts = df_edges["dominant_factor"].value_counts()
        for factor, count in dom_counts.items():
            pct = count / len(df_edges) * 100
            print(f"    {factor:25s}: {count:,} ({pct:.1f}%)")

        print(f"\n  Statistik Jarak Edge:")
        print(f"    Mean   : {df_edges['distance_km'].mean():.2f} km")
        print(f"    Max    : {df_edges['distance_km'].max():.2f} km")

        # Edge yang menghubungkan node eskalasi
        esc_edges = df_edges[
            (df_edges["source_label"] == 1) | (df_edges["target_label"] == 1)
        ]
        print(f"\n  Edge terhubung ke node eskalasi: {len(esc_edges):,} ({len(esc_edges)/len(df_edges)*100:.1f}%)")
    else:
        print("\n  [WARNING] Tidak ada edge yang terbentuk!")
        print("  Kemungkinan penyebab:")
        print("    - Cluster terlalu jarang (jarak > 5km semua)")
        print("    - Cluster tidak ada yang muncul dalam 3 hari berurutan")
        print("  Coba perbesar MAX_SPATIAL_DISTANCE_KM atau MAX_TEMPORAL_DISTANCE_DAYS")


def detect_wind_columns(df):
    """Deteksi kolom angin yang tersedia di dataframe."""
    wind_speed_col = None
    wind_dir_col = None
    has_precalc = WIND_ALIGNMENT_PRECALC in df.columns

    for c in WIND_SPEED_CANDIDATES:
        if c in df.columns:
            wind_speed_col = c
            break

    for c in WIND_DIR_CANDIDATES:
        if c in df.columns:
            wind_dir_col = c
            break

    return wind_speed_col, wind_dir_col


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 65)
    print("13_BANGUN_SPATIOTEMPORAL_GRAPH.PY")
    print("Step 7: Konstruksi Spatio-Temporal Graph G = (V, E)")
    print("=" * 65)

    # --- Load data ---
    if not os.path.exists(INPUT_FILE):
        print(f"\n[ERROR] File tidak ditemukan: {INPUT_FILE}")
        print("Pastikan 09_label_eskalasi.py sudah dijalankan.")
        return

    print(f"\nMemuat: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Berhasil dimuat: {len(df):,} cluster")

    # --- Deteksi kolom angin ---
    wind_speed_col, wind_dir_col = detect_wind_columns(df)
    has_precalc = WIND_ALIGNMENT_PRECALC in df.columns
    print(f"\nKolom angin terdeteksi:")
    print(f"  wind_alignment_score (ERA5 precalc): {'ADA - digunakan sebagai cos(theta)' if has_precalc else 'tidak ada'}")
    print(f"  Windspeed    : {wind_speed_col or 'tidak ada'}")
    print(f"  Winddirection: {wind_dir_col or 'tidak ada (fallback ke precalc/default)'}")

    # Cek kolom slope
    has_slope = "slope" in df.columns
    print(f"  Slope DEM    : {'ADA' if has_slope else 'tidak ada (default 0)'}")

    # --- Bangun graf ---
    G = nx.DiGraph()  # Directed graph (arah = urutan waktu)

    # Bagian A: Node
    G = build_nodes(G, df)

    # Bagian B: Edge kandidat
    valid_edges = find_candidate_edges(df)

    if len(valid_edges) == 0:
        print("\n[WARNING] Tidak ada edge yang terbentuk! Simpan graf tanpa edge...")
        save_outputs(G, pd.DataFrame())
        print_summary(G, pd.DataFrame())
        return

    # Bagian C: Kalkulasi Wij
    G, df_edges = compute_edge_weights(G, df, valid_edges, wind_speed_col, wind_dir_col)

    # Bagian D: Simpan
    save_outputs(G, df_edges)

    # Ringkasan
    print_summary(G, df_edges)

    print("\n" + "=" * 65)
    print("SELESAI — Lanjut ke: python 14_community_detection_archetype.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
