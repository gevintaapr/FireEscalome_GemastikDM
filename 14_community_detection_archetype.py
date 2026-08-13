"""
14_community_detection_archetype.py
Fase 4: Community Detection & Archetype Classification Archetype dari Struktur Graf

Menjalankan Louvain Algorithm pada Spatio-Temporal Graph untuk menemukan
kluster alami dari struktur koneksi antar-node — sebagai VALIDASI SILANG
terhadap clustering fitur (yang dilakukan lewat SHAP Interaction Values).

Untuk setiap komunitas yang ditemukan, analisis faktor edge dominan:
- Komunitas dengan edge dominan "wind_alignment" → kandidat Wind-driven archetype
- Komunitas dengan edge dominan "peatland_continuity" → kandidat Peat-driven archetype
- dst.

Output:
  - Kolom 'community_id' dan 'dominant_edge_factor' ditambahkan ke cluster_master_features.csv
  - data_processed/community_analysis.csv (statistik per komunitas)

Jalankan: python 14_community_detection_archetype.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx

try:
    from community import community_louvain  # pip install python-louvain
except ImportError:
    community_louvain = None

# ==============================================================
# KONFIGURASI
# ==============================================================
try:
    from config import PROCESSED_DIR
except ImportError:
    from pathlib import Path
    PROCESSED_DIR = Path("G:/My Drive/gemastik/karhutla/data_processed")

from pathlib import Path
PROCESSED_B_DIR = Path(str(PROCESSED_DIR).replace("data_processed", "data_processed_B"))

GRAPH_FILE = PROCESSED_B_DIR / "spatiotemporal_graph.pkl"
CLUSTER_FILE = PROCESSED_B_DIR / "cluster_master_PersonB.csv"
EDGE_FILE = PROCESSED_B_DIR / "edge_statistics.csv"
OUTPUT_COMMUNITY_CSV = PROCESSED_B_DIR / "community_analysis.csv"

# Louvain resolution parameter (> 1.0 = lebih banyak komunitas, < 1.0 = lebih sedikit)
LOUVAIN_RESOLUTION = 1.0


def load_graph():
    """Load graf dari pickle."""
    print("1. Memuat Spatio-Temporal Graph...")
    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    print(f"   Node: {G.number_of_nodes():,}, Edge: {G.number_of_edges():,}")
    return G


def run_louvain(G):
    """
    Jalankan Louvain community detection.
    Louvain membutuhkan graf undirected, jadi konversi dulu.
    """
    print("\n2. Menjalankan Louvain Community Detection...")

    if community_louvain is None:
        print("   [ERROR] Library python-louvain belum terinstal!")
        print("   Jalankan: pip install python-louvain")
        return None

    # Konversi directed → undirected (ambil weight rata-rata jika ada edge dua arah)
    G_undirected = G.to_undirected()

    # Jalankan Louvain
    partition = community_louvain.best_partition(
        G_undirected,
        weight="weight",
        resolution=LOUVAIN_RESOLUTION,
        random_state=42,
    )

    n_communities = len(set(partition.values()))
    print(f"   Ditemukan {n_communities} komunitas (resolution={LOUVAIN_RESOLUTION})")

    # Distribusi ukuran komunitas
    comm_sizes = pd.Series(partition).value_counts().sort_index()
    print(f"   Distribusi ukuran komunitas:")
    for comm_id, size in comm_sizes.head(10).items():
        print(f"     Komunitas {comm_id}: {size:,} node")
    if len(comm_sizes) > 10:
        print(f"     ... dan {len(comm_sizes) - 10} komunitas lainnya")

    return partition


def analyze_communities(G, partition, df_edges):
    """
    Untuk setiap komunitas, analisis:
    1. Faktor edge dominan apa yang paling banyak?
    2. Berapa % node yang bereskalasi?
    3. Rata-rata growth_ratio?
    """
    print("\n3. Menganalisis karakteristik tiap komunitas...")

    if partition is None or df_edges is None or len(df_edges) == 0:
        print("   [WARNING] Tidak bisa analisis — partition atau edge kosong.")
        return pd.DataFrame()

    # Buat mapping node → community
    node_to_comm = partition

    community_stats = []

    for comm_id in sorted(set(partition.values())):
        # Node dalam komunitas ini
        comm_nodes = [n for n, c in partition.items() if c == comm_id]
        n_nodes = len(comm_nodes)

        if n_nodes == 0:
            continue

        # Statistik node
        escalated = sum(
            1 for n in comm_nodes
            if G.nodes[n].get("label_escalation", 0) == 1
        )
        peat_count = sum(
            1 for n in comm_nodes
            if G.nodes[n].get("is_peatland", 0) == 1
        )
        avg_growth = np.mean([
            G.nodes[n].get("growth_ratio", 0) for n in comm_nodes
        ])

        # Edge dalam komunitas ini (intra-community edges)
        comm_set = set(comm_nodes)
        intra_edges = df_edges[
            (df_edges["source_id"].isin(comm_set)) & (df_edges["target_id"].isin(comm_set))
        ]

        n_intra_edges = len(intra_edges)

        # Faktor dominan pada edge dalam komunitas
        if n_intra_edges > 0:
            dominant_factor_distribution = intra_edges["dominant_factor"].value_counts(normalize=True)
            top_factor = dominant_factor_distribution.index[0]
            top_factor_pct = dominant_factor_distribution.iloc[0] * 100

            # Rata-rata komponen edge
            avg_spatial = intra_edges["spatial_decay"].mean()
            avg_wind = intra_edges["wind_alignment"].mean()
            avg_slope = intra_edges["slope_direction"].mean()
            avg_peat_edge = intra_edges["peatland_continuity"].mean()
        else:
            top_factor = "none"
            top_factor_pct = 0
            avg_spatial = avg_wind = avg_slope = avg_peat_edge = 0

        # Mapping faktor dominan → label archetype sugesti
        archetype_map = {
            "wind_alignment": "Wind-driven",
            "peatland_continuity": "Peat-driven",
            "slope_direction": "Topography-driven",
            "spatial_decay": "Proximity-driven",
            "none": "Unknown",
        }
        suggested_archetype = archetype_map.get(top_factor, "Unknown")

        community_stats.append({
            "community_id": comm_id,
            "n_nodes": n_nodes,
            "n_escalated": escalated,
            "pct_escalated": round(escalated / n_nodes * 100, 1) if n_nodes > 0 else 0,
            "n_peatland_nodes": peat_count,
            "avg_growth_ratio": round(avg_growth, 2),
            "n_intra_edges": n_intra_edges,
            "dominant_factor": top_factor,
            "dominant_factor_pct": round(top_factor_pct, 1),
            "suggested_archetype": suggested_archetype,
            "avg_spatial_decay": round(avg_spatial, 4),
            "avg_wind_alignment": round(avg_wind, 4),
            "avg_slope_direction": round(avg_slope, 4),
            "avg_peatland_continuity": round(avg_peat_edge, 4),
        })

    df_comm = pd.DataFrame(community_stats)
    return df_comm


def merge_community_to_clusters(partition, df_clusters):
    """Tambahkan kolom community_id ke cluster_master_features.csv."""
    print("\n4. Menambahkan kolom community_id ke cluster data...")

    if partition is None:
        df_clusters["community_id"] = -1
        return df_clusters

    df_clusters["community_id"] = df_clusters["cluster_id"].map(partition).fillna(-1).astype(int)

    assigned = (df_clusters["community_id"] >= 0).sum()
    print(f"   {assigned:,} / {len(df_clusters):,} cluster mendapat community_id")

    return df_clusters


def print_archetype_summary(df_comm):
    """Cetak ringkasan archetype yang terdeteksi."""
    print("\n" + "=" * 65)
    print("RINGKASAN ARCHETYPE DARI STRUKTUR GRAF")
    print("=" * 65)

    if len(df_comm) == 0:
        print("  Tidak ada komunitas yang bisa dianalisis.")
        return

    # Hanya komunitas non-trivial (≥ 3 node)
    df_significant = df_comm[df_comm["n_nodes"] >= 3]

    if len(df_significant) == 0:
        print("  Tidak ada komunitas signifikan (≥3 node).")
        return

    print(f"\n  Komunitas signifikan (≥3 node): {len(df_significant)}")
    print()

    for _, row in df_significant.iterrows():
        esc_flag = "🔥" if row["pct_escalated"] > 10 else "  "
        print(f"  {esc_flag} Komunitas {row['community_id']:3d}: "
              f"{row['n_nodes']:4d} node | "
              f"Eskalasi: {row['n_escalated']:3d} ({row['pct_escalated']:.1f}%) | "
              f"Archetype: {row['suggested_archetype']:18s} "
              f"({row['dominant_factor']} {row['dominant_factor_pct']:.0f}%)")

    # Distribusi archetype secara keseluruhan
    print(f"\n  Distribusi archetype (berdasarkan jumlah node):")
    archetype_node_counts = {}
    for _, row in df_significant.iterrows():
        arch = row["suggested_archetype"]
        archetype_node_counts[arch] = archetype_node_counts.get(arch, 0) + row["n_nodes"]

    total_nodes = sum(archetype_node_counts.values())
    for arch, count in sorted(archetype_node_counts.items(), key=lambda x: -x[1]):
        print(f"    {arch:20s}: {count:,} node ({count/total_nodes*100:.1f}%)")


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 65)
    print("14_COMMUNITY_DETECTION_ARCHETYPE.PY")
    print("Fase 4: Louvain Community Detection & Analisis Archetype")
    print("=" * 65)

    # --- Validasi file ---
    if not os.path.exists(GRAPH_FILE):
        print(f"\n[ERROR] Graf tidak ditemukan: {GRAPH_FILE}")
        print("Pastikan 13_bangun_spatiotemporal_graph.py sudah dijalankan.")
        return

    # --- Load ---
    G = load_graph()

    df_edges = pd.DataFrame()
    if os.path.exists(EDGE_FILE):
        df_edges = pd.read_csv(EDGE_FILE)
        print(f"   Edge statistics dimuat: {len(df_edges):,} edge")

    df_clusters = pd.read_csv(CLUSTER_FILE)

    # --- Louvain ---
    partition = run_louvain(G)

    # --- Analisis komunitas ---
    df_comm = analyze_communities(G, partition, df_edges)

    # --- Simpan hasil ---
    if len(df_comm) > 0:
        df_comm.to_csv(OUTPUT_COMMUNITY_CSV, index=False)
        print(f"\n   Community analysis tersimpan: {OUTPUT_COMMUNITY_CSV}")

    # --- Merge ke cluster data ---
    df_clusters = merge_community_to_clusters(partition, df_clusters)
    df_clusters.to_csv(CLUSTER_FILE, index=False)
    print(f"   cluster_master_features.csv diperbarui dengan kolom community_id")

    # --- Ringkasan archetype ---
    print_archetype_summary(df_comm)

    print("\n" + "=" * 65)
    print("SELESAI - Lanjut ke: python 15_graph_explainability_lite.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
