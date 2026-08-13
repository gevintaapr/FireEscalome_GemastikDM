"""
15_graph_explainability_lite.py
Fase 5 (Step 9): Graph Explainability Analysis(Graph-Lite Approach)

Step 9 (bagian graf) — Menjawab pertanyaan:
  "Faktor apa yang paling menentukan dua node (cluster kebakaran) terhubung?"

Karena bobot edge Wij sudah didekomposisi per komponen sejak awal (di script 11),
explainability TIDAK membutuhkan model black box (GNN/GNNExplainer).
Cukup analisis statistik dari breakdown edge yang sudah ada.

Analisis yang dilakukan:
  1. Distribusi faktor dominan secara global
  2. Perbandingan faktor pada edge eskalasi vs non-eskalasi
  3. Analisis per archetype/komunitas
  4. Top-K edge terkuat untuk setiap faktor
  5. Visualisasi graf (matplotlib)

Output:
  - data_processed/graph_explainability_report.csv
  - data_processed/graph_visualization.png
  - Laporan teks lengkap di console

Jalankan: python 15_graph_explainability_lite.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
EDGE_FILE = PROCESSED_B_DIR / "edge_statistics.csv"
COMMUNITY_FILE = PROCESSED_B_DIR / "community_analysis.csv"
CLUSTER_FILE = PROCESSED_B_DIR / "cluster_master_PersonB.csv"

OUTPUT_REPORT        = PROCESSED_B_DIR / "graph_explainability_report.csv"
OUTPUT_VIZ           = PROCESSED_B_DIR / "graph_visualization.png"          # dark mode, garis melengkung
OUTPUT_VIZ_STRAIGHT  = PROCESSED_B_DIR / "graph_visualization_straight.png" # dark mode, garis lurus
OUTPUT_VIZ_NODES     = PROCESSED_B_DIR / "graph_visualization_nodes_only.png" # hanya node, tanpa edge
OUTPUT_VIZ_ZOOM      = PROCESSED_B_DIR / "graph_visualization_zoom.png"      # zoom kluster terpadat
OUTPUT_FACTOR_CHART  = PROCESSED_B_DIR / "graph_factor_comparison.png"

# Komponen edge
FACTOR_COLS = ["spatial_decay", "wind_alignment", "slope_direction", "peatland_continuity"]
FACTOR_LABELS = {
    "spatial_decay": "Spatial Decay (Kedekatan Jarak)",
    "wind_alignment": "Wind Alignment (Arah Angin)",
    "slope_direction": "Slope Direction (Kemiringan Lereng)",
    "peatland_continuity": "Peatland Continuity (Gambut)",
}

# Warna per faktor — dipilih kontras terhadap merah/abu node
FACTOR_COLORS = {
    "spatial_decay":       "#2ECC71",   # Hijau  — penularan karena jarak dekat
    "wind_alignment":      "#3498DB",   # Biru   — penularan karena angin
    "slope_direction":     "#F39C12",   # Oranye — penularan karena lereng
    "peatland_continuity": "#8E44AD",   # Ungu   — penularan karena gambut
}


# ==============================================================
# ANALISIS 1: Distribusi Faktor Dominan Global
# ==============================================================

def analyze_global_distribution(df_edges):
    """Distribusi faktor dominan di semua edge."""
    print("\n" + "=" * 65)
    print("[ANALISIS 1] DISTRIBUSI FAKTOR DOMINAN — GLOBAL")
    print("=" * 65)

    if len(df_edges) == 0:
        print("  Tidak ada edge untuk dianalisis.")
        return

    dom = df_edges["dominant_factor"].value_counts()
    dom_pct = df_edges["dominant_factor"].value_counts(normalize=True) * 100

    print(f"\n  Total edge: {len(df_edges):,}\n")
    for factor in FACTOR_COLS:
        count = dom.get(factor, 0)
        pct = dom_pct.get(factor, 0)
        label = FACTOR_LABELS.get(factor, factor)
        bar = "█" * int(pct / 2)
        print(f"  {label:40s}: {count:6,} ({pct:5.1f}%) {bar}")

    # Rata-rata nilai komponen (bukan cuma dominan)
    print(f"\n  Rata-rata nilai komponen (semua edge):")
    for factor in FACTOR_COLS:
        mean_val = df_edges[factor].mean()
        label = FACTOR_LABELS.get(factor, factor)
        print(f"    {label:40s}: {mean_val:.4f}")


# ==============================================================
# ANALISIS 2: Eskalasi vs Non-Eskalasi
# ==============================================================

def analyze_escalation_comparison(df_edges):
    """
    Bandingkan profil edge yang terhubung ke node eskalasi vs tidak.
    Ini adalah analisis paling penting untuk Graph Explainability.
    """
    print("\n" + "=" * 65)
    print("[ANALISIS 2] PERBANDINGAN FAKTOR: ESKALASI vs NON-ESKALASI")
    print("=" * 65)

    if len(df_edges) == 0:
        print("  Tidak ada edge untuk dianalisis.")
        return pd.DataFrame()

    # Edge yang menghubungkan minimal 1 node eskalasi
    esc_mask = (df_edges["source_label"] == 1) | (df_edges["target_label"] == 1)
    df_esc = df_edges[esc_mask]
    df_non = df_edges[~esc_mask]

    print(f"\n  Edge terhubung eskalasi    : {len(df_esc):,}")
    print(f"  Edge non-eskalasi         : {len(df_non):,}")

    if len(df_esc) == 0:
        print("  [WARNING] Tidak ada edge eskalasi — analisis tidak bisa dilanjutkan.")
        return pd.DataFrame()

    print(f"\n  {'Komponen':<35s} {'Eskalasi':>10s} {'Non-Esc.':>10s} {'Δ (diff)':>10s} {'Insight':>15s}")
    print("  " + "-" * 80)

    comparison_data = []

    for factor in FACTOR_COLS:
        mean_esc = df_esc[factor].mean()
        mean_non = df_non[factor].mean() if len(df_non) > 0 else 0
        diff = mean_esc - mean_non

        # Insight: faktor mana yang lebih tinggi di edge eskalasi?
        if abs(diff) < 0.01:
            insight = "≈ Sama"
        elif diff > 0:
            insight = "↑ Eskalasi"
        else:
            insight = "↓ Eskalasi"

        label = FACTOR_LABELS.get(factor, factor)
        print(f"  {label:<35s} {mean_esc:>10.4f} {mean_non:>10.4f} {diff:>+10.4f} {insight:>15s}")

        comparison_data.append({
            "factor": factor,
            "mean_escalation": round(mean_esc, 4),
            "mean_non_escalation": round(mean_non, 4),
            "difference": round(diff, 4),
            "insight": insight,
        })

    # Faktor dominan pada edge eskalasi
    print(f"\n  Faktor DOMINAN pada edge eskalasi:")
    esc_dom = df_esc["dominant_factor"].value_counts(normalize=True) * 100
    for factor, pct in esc_dom.items():
        label = FACTOR_LABELS.get(factor, factor)
        print(f"    {label:40s}: {pct:.1f}%")

    print(f"\n  Faktor DOMINAN pada edge non-eskalasi:")
    if len(df_non) > 0:
        non_dom = df_non["dominant_factor"].value_counts(normalize=True) * 100
        for factor, pct in non_dom.items():
            label = FACTOR_LABELS.get(factor, factor)
            print(f"    {label:40s}: {pct:.1f}%")

    return pd.DataFrame(comparison_data)


# ==============================================================
# ANALISIS 3: Per Komunitas / Archetype
# ==============================================================

def analyze_per_community(df_edges, df_comm):
    """Analisis faktor dominan per komunitas (dari Louvain)."""
    print("\n" + "=" * 65)
    print("[ANALISIS 3] PROFIL FAKTOR PER KOMUNITAS (ARCHETYPE)")
    print("=" * 65)

    if df_comm is None or len(df_comm) == 0:
        print("  Community analysis belum tersedia. Jalankan 14_community_detection_archetype.py dulu.")
        return

    # Hanya komunitas signifikan
    df_sig = df_comm[df_comm["n_nodes"] >= 3].copy()
    if len(df_sig) == 0:
        print("  Tidak ada komunitas signifikan.")
        return

    print(f"\n  {'Komunitas':>10s} {'Node':>6s} {'Esc%':>6s} {'Archetype':>20s} "
          f"{'Spatial':>8s} {'Wind':>8s} {'Slope':>8s} {'Peat':>8s}")
    print("  " + "-" * 82)

    for _, row in df_sig.iterrows():
        print(f"  {row['community_id']:>10d} {row['n_nodes']:>6d} "
              f"{row['pct_escalated']:>5.1f}% "
              f"{row['suggested_archetype']:>20s} "
              f"{row['avg_spatial_decay']:>8.4f} "
              f"{row['avg_wind_alignment']:>8.4f} "
              f"{row['avg_slope_direction']:>8.4f} "
              f"{row['avg_peatland_continuity']:>8.4f}")


# ==============================================================
# ANALISIS 4: Top-K Edge Terkuat per Faktor
# ==============================================================

def analyze_top_edges(df_edges, top_k=5):
    """Tampilkan edge terkuat untuk setiap komponen faktor."""
    print("\n" + "=" * 65)
    print(f"[ANALISIS 4] TOP-{top_k} EDGE TERKUAT PER FAKTOR")
    print("=" * 65)

    if len(df_edges) == 0:
        print("  Tidak ada edge.")
        return

    for factor in FACTOR_COLS:
        label = FACTOR_LABELS.get(factor, factor)
        print(f"\n  --- {label} ---")

        top = df_edges.nlargest(top_k, factor)
        for _, row in top.iterrows():
            esc_flag = "🔥" if (row["source_label"] == 1 or row["target_label"] == 1) else "  "
            print(f"  {esc_flag} {row['source_id']:.0f} → {row['target_id']:.0f} | "
                  f"{factor}={row[factor]:.4f} | "
                  f"Wij={row['weight_total']:.4f} | "
                  f"d={row['distance_km']:.1f}km Δt={row['delta_days']:.0f}d")


# ==============================================================
# VISUALISASI 5: Grafik Perbandingan Faktor
# ==============================================================

def plot_factor_comparison(df_edges):
    """Buat bar chart perbandingan faktor eskalasi vs non-eskalasi."""
    print("\n[VISUALISASI] Membuat grafik perbandingan faktor...")

    esc_mask = (df_edges["source_label"] == 1) | (df_edges["target_label"] == 1)
    df_esc = df_edges[esc_mask]
    df_non = df_edges[~esc_mask]

    if len(df_esc) == 0:
        print("  Tidak ada edge eskalasi untuk divisualisasikan.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Chart 1: Bar chart rata-rata komponen ---
    ax1 = axes[0]
    x = np.arange(len(FACTOR_COLS))
    width = 0.35

    esc_means = [df_esc[f].mean() for f in FACTOR_COLS]
    non_means = [df_non[f].mean() for f in FACTOR_COLS] if len(df_non) > 0 else [0] * len(FACTOR_COLS)

    bars1 = ax1.bar(x - width/2, esc_means, width, label="Edge Eskalasi",
                     color="#FF6B6B", alpha=0.85, edgecolor="white")
    bars2 = ax1.bar(x + width/2, non_means, width, label="Edge Non-Eskalasi",
                     color="#4ECDC4", alpha=0.85, edgecolor="white")

    short_labels = ["Spatial\nDecay", "Wind\nAlignment", "Slope\nDirection", "Peatland\nContinuity"]
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=10)
    ax1.set_ylabel("Rata-rata Nilai Komponen", fontsize=11)
    ax1.set_title("Perbandingan Faktor Edge:\nEskalasi vs Non-Eskalasi", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Tambahkan nilai di atas bar
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax1.annotate(f"{height:.3f}",
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=8)

    # --- Chart 2: Pie chart faktor dominan pada edge eskalasi ---
    ax2 = axes[1]
    esc_dom = df_esc["dominant_factor"].value_counts()
    colors = [FACTOR_COLORS.get(f, "#999999") for f in esc_dom.index]
    wedges, texts, autotexts = ax2.pie(
        esc_dom.values, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.85,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )

    # Legend
    legend_labels = [FACTOR_LABELS.get(f, f).split(" (")[0] for f in esc_dom.index]
    ax2.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax2.set_title("Faktor Dominan pada\nEdge Eskalasi", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_FACTOR_CHART, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Grafik tersimpan: {OUTPUT_FACTOR_CHART}")
    plt.close()


# ==============================================================
# VISUALISASI 6: Gambar Graf (Node + Edge)
# ==============================================================

def plot_graph_visualization(G, output_path=None, curved=True, show_edges=True):
    """Visualisasi graf — edge diwarnai berdasarkan faktor dominan penularan api.
    
    Args:
        G           : NetworkX graph
        output_path : Path file output (default = OUTPUT_VIZ)
        curved      : True = garis melengkung, False = garis lurus
        show_edges  : True = tampilkan edge berwarna, False = hanya node saja
    """
    if output_path is None:
        output_path = OUTPUT_VIZ
    print("\n[VISUALISASI] Membuat peta graf spatio-temporal...")

    if G.number_of_edges() == 0:
        print("  Tidak ada edge — skip visualisasi graf.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(18, 13))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    # Pisahkan node yang punya edge (connected) vs terisolasi
    connected_nodes = set()
    for u, v in G.edges():
        connected_nodes.add(u)
        connected_nodes.add(v)

    pos = {}
    for node, data in G.nodes(data=True):
        lon = data.get("longitude", 0)
        lat = data.get("latitude", 0)
        pos[node] = (lon, lat)

    # --- Gambar node terisolasi dulu (background, sangat kecil & transparan) ---
    isolated_nodes = [n for n in G.nodes() if n not in connected_nodes]
    isolated_colors = []
    for node in isolated_nodes:
        data = G.nodes[node]
        if data.get("label_escalation", 0) == 1:
            isolated_colors.append("#FF4444")
        else:
            isolated_colors.append("#555577")

    if isolated_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=isolated_nodes, ax=ax,
            node_color=isolated_colors, node_size=6,
            alpha=0.25, edgecolors="none"
        )

    # --- Gambar edge dengan warna berdasarkan faktor dominan ---
    if show_edges:
        rad = 0.15 if curved else 0.0
        factor_edges = {f: [] for f in FACTOR_COLORS}
        for u, v, data in G.edges(data=True):
            factor = data.get("dominant_factor", "spatial_decay")
            if factor in factor_edges:
                factor_edges[factor].append((u, v))

        for factor, edges_list in factor_edges.items():
            if not edges_list:
                continue
            color = FACTOR_COLORS[factor]
            nx.draw_networkx_edges(
                G, pos, edgelist=edges_list, ax=ax,
                edge_color=color, width=2.0, alpha=0.85,
                arrows=True, arrowsize=12,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                min_source_margin=6, min_target_margin=6
            )

    # --- Gambar node yang terhubung (foreground, lebih besar & jelas) ---
    connected_esc    = [n for n in connected_nodes if G.nodes[n].get("label_escalation", 0) == 1]
    connected_nonesc = [n for n in connected_nodes if G.nodes[n].get("label_escalation", 0) == 0]

    if connected_nonesc:
        nx.draw_networkx_nodes(
            G, pos, nodelist=connected_nonesc, ax=ax,
            node_color="#AAAACC", node_size=60,
            alpha=0.9, edgecolors="white", linewidths=0.8
        )
    if connected_esc:
        nx.draw_networkx_nodes(
            G, pos, nodelist=connected_esc, ax=ax,
            node_color="#FF2222", node_size=130,
            alpha=1.0, edgecolors="white", linewidths=1.5
        )

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor="#FF2222", edgecolor="white", linewidth=1,
                       label="Node Eskalasi (api besar)"),
        mpatches.Patch(facecolor="#AAAACC", edgecolor="white", linewidth=0.8,
                       label="Node Non-Eskalasi (api kecil)"),
        mpatches.Patch(facecolor="#555577", alpha=0.4, label="Node Terisolasi (tidak terhubung)"),
    ]
    for factor, color in FACTOR_COLORS.items():
        short_name = FACTOR_LABELS.get(factor, factor).split(" (")[0]
        legend_elements.append(
            mpatches.Patch(facecolor=color, label=f"Edge: {short_name}")
        )

    leg = ax.legend(
        handles=legend_elements, loc="upper left", fontsize=10,
        framealpha=0.85, edgecolor="#AAAACC", facecolor="#1A1A2E",
        labelcolor="white", title="Keterangan", title_fontsize=11
    )
    leg.get_title().set_color("white")

    ax.set_xlabel("Longitude", fontsize=12, color="white")
    ax.set_ylabel("Latitude", fontsize=12, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    total_esc_conn = len(connected_esc)
    if show_edges:
        style_info = "Garis Lurus" if not curved else "Garis Melengkung"
        title_str = (
            f"Spatio-Temporal Fire Propagation Graph — FireEscalome\n"
            f"{G.number_of_edges()} Edge Penularan Api  |  Warna Edge = Faktor Penyebab  |  "
            f"{total_esc_conn} Node Eskalasi Terhubung  [{style_info}]"
        )
    else:
        title_str = (
            f"Spatio-Temporal Fire Propagation Graph — FireEscalome\n"
            f"Distribusi Node  |  Merah = Eskalasi  |  {total_esc_conn} Node Eskalasi Terhubung"
        )
    ax.set_title(title_str, fontsize=13, fontweight="bold", color="white", pad=15)
    ax.grid(True, alpha=0.1, color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#1A1A2E")
    print(f"  Graf tersimpan: {output_path}")
    plt.close()


# ==============================================================
# VISUALISASI 7: Zoom-In Kluster Terpadat
# ==============================================================

def plot_graph_zoom(G, output_path=None, padding_deg=0.15):
    """Zoom-in ke area kluster yang paling padat eskalasinya.
    
    Strategi:
    - Cari area 1° x 1° yang paling banyak mengandung node eskalasi terhubung.
    - Gambar ulang dengan node besar, edge tebal, dan panah yang jelas.
    """
    if output_path is None:
        output_path = OUTPUT_VIZ_ZOOM

    print("\n[VISUALISASI ZOOM] Mencari kluster terpadat...")

    if G.number_of_edges() == 0:
        print("  Tidak ada edge — skip zoom.")
        return

    # --- Kumpulkan koordinat semua node terhubung ---
    connected_nodes = set()
    for u, v in G.edges():
        connected_nodes.add(u)
        connected_nodes.add(v)

    pos = {}
    for node, data in G.nodes(data=True):
        lon = data.get("longitude", 0)
        lat = data.get("latitude", 0)
        pos[node] = (lon, lat)

    # --- Strategi: cari area dengan konsentrasi EDGE tertinggi ---
    # Hitung midpoint setiap edge
    edge_midpoints = []
    for u, v, d in G.edges(data=True):
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        edge_midpoints.append((mx, my, u, v, d))

    # Grid search: kotak 0.5°x0.5° dengan edge terbanyak
    step = 0.25
    all_lons_e = [ep[0] for ep in edge_midpoints]
    all_lats_e = [ep[1] for ep in edge_midpoints]

    best_cx, best_cy = all_lons_e[0], all_lats_e[0]
    best_count = 0

    lon_range = [round(min(all_lons_e) + i * step, 4)
                 for i in range(int((max(all_lons_e) - min(all_lons_e)) / step) + 2)]
    lat_range = [round(min(all_lats_e) + i * step, 4)
                 for i in range(int((max(all_lats_e) - min(all_lats_e)) / step) + 2)]

    for lon_c in lon_range:
        for lat_c in lat_range:
            count = sum(
                1 for ep in edge_midpoints
                if abs(ep[0] - lon_c) <= 0.5 and abs(ep[1] - lat_c) <= 0.5
            )
            if count > best_count:
                best_count = count
                best_cx, best_cy = lon_c, lat_c

    cx, cy = best_cx, best_cy
    print(f"  Pusat kluster (lon, lat): ({cx:.4f}, {cy:.4f}), {best_count} edge di sekitarnya")

    x_min, x_max = cx - padding_deg, cx + padding_deg
    y_min, y_max = cy - padding_deg, cy + padding_deg

    # Filter node dan edge dalam area zoom
    nodes_in_zoom = [
        n for n in connected_nodes
        if x_min <= pos[n][0] <= x_max and y_min <= pos[n][1] <= y_max
    ]
    edges_in_zoom = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if u in nodes_in_zoom and v in nodes_in_zoom
    ]

    print(f"  Area zoom: lon [{x_min:.2f}, {x_max:.2f}], lat [{y_min:.2f}, {y_max:.2f}]")
    print(f"  Node dalam zoom: {len(nodes_in_zoom)}, Edge dalam zoom: {len(edges_in_zoom)}")

    # Perlebar sedikit jika edge kurang dari 5
    while len(edges_in_zoom) < 5 and padding_deg < 1.0:
        padding_deg += 0.05
        x_min, x_max = cx - padding_deg, cx + padding_deg
        y_min, y_max = cy - padding_deg, cy + padding_deg
        nodes_in_zoom = [
            n for n in connected_nodes
            if x_min <= pos[n][0] <= x_max and y_min <= pos[n][1] <= y_max
        ]
        edges_in_zoom = [
            (u, v, d) for u, v, d in G.edges(data=True)
            if u in nodes_in_zoom and v in nodes_in_zoom
        ]
    print(f"  Final: padding={padding_deg:.2f}°, {len(nodes_in_zoom)} node, {len(edges_in_zoom)} edge")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    # Gambar edge per faktor — garis LURUS, tebal, panah besar
    factor_edge_map = {f: [] for f in FACTOR_COLORS}
    for u, v, data in edges_in_zoom:
        factor = data.get("dominant_factor", "spatial_decay")
        if factor in factor_edge_map:
            factor_edge_map[factor].append((u, v))

    for factor, elist in factor_edge_map.items():
        if not elist:
            continue
        color = FACTOR_COLORS[factor]
        nx.draw_networkx_edges(
            G, pos, edgelist=elist, ax=ax,
            edge_color=color, width=3.5, alpha=0.92,
            arrows=True, arrowsize=25,
            connectionstyle="arc3,rad=0.0",   # garis LURUS
            arrowstyle="-|>",
            min_source_margin=10, min_target_margin=10
        )

    # Gambar node di dalam zoom
    zoom_esc    = [n for n in nodes_in_zoom if G.nodes[n].get("label_escalation", 0) == 1]
    zoom_nonesc = [n for n in nodes_in_zoom if G.nodes[n].get("label_escalation", 0) == 0]

    if zoom_nonesc:
        nx.draw_networkx_nodes(
            G, pos, nodelist=zoom_nonesc, ax=ax,
            node_color="#AAAACC", node_size=250,
            alpha=0.95, edgecolors="white", linewidths=1.5
        )
    if zoom_esc:
        nx.draw_networkx_nodes(
            G, pos, nodelist=zoom_esc, ax=ax,
            node_color="#FF2222", node_size=500,
            alpha=1.0, edgecolors="white", linewidths=2.5
        )

    # Label node (hanya node eskalasi di zoom)
    labels = {n: str(n) for n in zoom_esc}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8, font_color="white", font_weight="bold"
    )

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor="#FF2222", edgecolor="white", linewidth=1.5,
                       label="Node Eskalasi (api besar)"),
        mpatches.Patch(facecolor="#AAAACC", edgecolor="white", linewidth=1,
                       label="Node Non-Eskalasi (api kecil)"),
    ]
    for factor, color in FACTOR_COLORS.items():
        short_name = FACTOR_LABELS.get(factor, factor).split(" (")[0]
        legend_elements.append(
            mpatches.Patch(facecolor=color, label=f"Edge: {short_name}")
        )

    leg = ax.legend(
        handles=legend_elements, loc="upper left", fontsize=11,
        framealpha=0.9, edgecolor="#AAAACC", facecolor="#1A1A2E",
        labelcolor="white", title="Keterangan", title_fontsize=12
    )
    leg.get_title().set_color("white")

    # Zoom axis
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Longitude", fontsize=13, color="white")
    ax.set_ylabel("Latitude", fontsize=13, color="white")
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    # --- Deteksi nama lokasi berdasarkan koordinat ---
    def get_location_name(lon, lat):
        """Mengembalikan nama pulau/provinsi berdasarkan koordinat kasar."""
        if 108 <= lon <= 119 and -4 <= lat <= 7:
            if lon < 112:
                return "Kalimantan Barat"
            elif lon < 115 and lat > 0:
                return "Kalimantan Utara/Timur"
            elif lon < 115 and lat <= 0:
                return "Kalimantan Tengah"
            elif lon >= 115 and lat < -1:
                return "Kalimantan Selatan"
            else:
                return "Kalimantan Timur"
        elif 95 <= lon <= 106 and -6 <= lat <= 6:
            if lat > 2:
                return "Aceh / Sumatra Utara"
            elif lat > 0:
                return "Riau / Sumatra Barat"
            elif lat > -2:
                return "Jambi / Riau"
            else:
                return "Sumatra Selatan / Lampung"
        elif 119 <= lon <= 127 and -4 <= lat <= 2:
            return "Sulawesi"
        elif lon > 130:
            return "Papua"
        else:
            return "Indonesia"

    loc_name = get_location_name(cx, cy)

    ax.set_title(
        f"ZOOM: Kluster Terpadat — {loc_name}\n"
        f"Koordinat Pusat: Lon {cx:.3f}\u00b0  Lat {cy:.3f}\u00b0  |  "
        f"{len(edges_in_zoom)} Edge  |  {len(zoom_esc)} Node Eskalasi\n"
        f"Warna Garis = Faktor Penyebab Penularan Api",
        fontsize=12, fontweight="bold", color="white", pad=15
    )
    ax.grid(True, alpha=0.15, color="white", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#1A1A2E")
    print(f"  Zoom graf tersimpan: {output_path}")
    plt.close()


# ==============================================================
# SIMPAN LAPORAN RINGKASAN
# ==============================================================

def save_report(df_comparison, df_edges):
    """Simpan ringkasan explainability ke CSV."""
    if df_comparison is not None and len(df_comparison) > 0:
        df_comparison.to_csv(OUTPUT_REPORT, index=False)
        print(f"\n  Laporan explainability tersimpan: {OUTPUT_REPORT}")

    return


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 65)
    print("15_GRAPH_EXPLAINABILITY_LITE.PY")
    print("Step 9 (bagian graf): Graph Explainability — Analisis Faktor Edge")
    print("=" * 65)

    # --- Validasi file ---
    if not os.path.exists(EDGE_FILE):
        print(f"\n[ERROR] Edge statistics tidak ditemukan: {EDGE_FILE}")
        print("Pastikan 13_bangun_spatiotemporal_graph.py sudah dijalankan.")
        return

    if not os.path.exists(GRAPH_FILE):
        print(f"\n[ERROR] Graf tidak ditemukan: {GRAPH_FILE}")
        return

    # --- Load data ---
    print("\nMemuat data...")
    df_edges = pd.read_csv(EDGE_FILE)
    print(f"  Edge statistics: {len(df_edges):,} edge")

    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    print(f"  Graf: {G.number_of_nodes():,} node, {G.number_of_edges():,} edge")

    df_comm = None
    if os.path.exists(COMMUNITY_FILE):
        df_comm = pd.read_csv(COMMUNITY_FILE)
        print(f"  Community analysis: {len(df_comm):,} komunitas")

    # --- Jalankan semua analisis ---
    analyze_global_distribution(df_edges)
    df_comparison = analyze_escalation_comparison(df_edges)
    analyze_per_community(df_edges, df_comm)
    analyze_top_edges(df_edges, top_k=5)

    # --- Visualisasi ---
    plot_factor_comparison(df_edges)

    print("\n[VISUALISASI 6a] Peta graf — garis melengkung (sudah ada)...")
    plot_graph_visualization(G, output_path=OUTPUT_VIZ, curved=True, show_edges=True)

    print("\n[VISUALISASI 6b] Peta graf — garis LURUS...")
    plot_graph_visualization(G, output_path=OUTPUT_VIZ_STRAIGHT, curved=False, show_edges=True)

    print("\n[VISUALISASI 6c] Peta node saja (tanpa garis edge)...")
    plot_graph_visualization(G, output_path=OUTPUT_VIZ_NODES, curved=False, show_edges=False)

    print("\n[VISUALISASI 7] Zoom-in kluster terpadat (garis lurus + panah jelas)...")
    plot_graph_zoom(G, output_path=OUTPUT_VIZ_ZOOM)

    # --- Simpan report ---
    save_report(df_comparison, df_edges)

    # --- Verdict akhir ---
    print("\n" + "=" * 65)
    print("KESIMPULAN GRAPH EXPLAINABILITY")
    print("=" * 65)

    if len(df_edges) > 0:
        esc_mask = (df_edges["source_label"] == 1) | (df_edges["target_label"] == 1)
        df_esc = df_edges[esc_mask]

        if len(df_esc) > 0:
            top_factor = df_esc["dominant_factor"].value_counts().index[0]
            top_pct = df_esc["dominant_factor"].value_counts(normalize=True).iloc[0] * 100
            label = FACTOR_LABELS.get(top_factor, top_factor)

            print(f"\n  🔥 TEMUAN UTAMA:")
            print(f"     Pada edge yang terhubung ke cluster eskalasi,")
            print(f"     faktor PALING DOMINAN adalah:")
            print(f"       \"{label}\" ({top_pct:.1f}% dari seluruh edge eskalasi)")
            print(f"\n     Ini berarti mekanisme fire propagation di wilayah studi")
            print(f"     paling banyak dipengaruhi oleh faktor {top_factor.replace('_', ' ')}.")

    print("\n" + "=" * 65)
    print("SELESAI — Semua output Person B telah dihasilkan!")
    print("=" * 65)
    print(f"\n  Output files:")
    print(f"    - spatiotemporal_graph.pkl (Graf lengkap)")
    print(f"    - edge_statistics.csv (Breakdown tiap edge)")
    print(f"    - community_analysis.csv (Analisis komunitas)")
    print(f"    4. {OUTPUT_REPORT.name:40s} (Laporan explainability)")
    print(f"    5. {OUTPUT_VIZ.name:40s} (Peta graf)")
    print(f"    6. {OUTPUT_FACTOR_CHART.name:40s} (Chart perbandingan faktor)")


if __name__ == "__main__":
    main()
