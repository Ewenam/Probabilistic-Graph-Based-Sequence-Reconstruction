#!/usr/bin/env python3
"""
Advanced visualization tools for Spectral-GMM wastewater clustering.

Inputs (from your pipeline):
  - spectral_embedding.npy          (Y: spectral embedding)
  - spectral_gmm_assignments.csv    (taxon, community)
  - graph_taxa_index.tsv            (taxon names)
  - graph_adjacency.csv             (adjacency matrix A)
  - rho.npy                         (Spearman correlation matrix)        [optional, best-effort auto find]
  - sbm_block_P.csv                 (SBM block matrix)                   [optional, best-effort auto find]

Main entry-points:
  - run_advanced_visualization(embedding_path, assignments_path, taxa_path, adjacency_path,
                               rho_path=None, sbm_block_path=None, outdir="visualizations")
  - auto_visualize(project_out_dir)   # automatically finds all the above under project_out_dir

This module produces:
  - spectral_embedding.png
  - umap_embedding.png
  - spectral_vs_umap.png
  - network_graph.png
  - cluster_sizes.png
  - degree_distribution.png
  - sorted_correlation_heatmap.png
  - taxa_per_cluster.tsv
  - taxa_enrichment_summary.tsv
  - louvain_vs_gmm_confusion_counts.png
  - louvain_gmm_ARI.txt
  - silhouette_scores.png
  - sbm_block_heatmap.png            (if sbm_block_P.csv found)
  - pathogens_embedding.png
  - interactive_network.html         (if pyvis is installed)
  - spectral_animation.gif           (simple eigenvector build-up animation)
  - wiley_circular_biological.png    (Wiley-style circular degree plot with biological labels)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
import umap

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def auto_find_file(base: str, name_substrings):
    """
    Recursively search for a file containing ANY of the given substrings in its name.
    Returns the first match or None.
    """
    name_substrings = list(name_substrings)
    for root, dirs, files in os.walk(base):
        for f in files:
            for pat in name_substrings:
                if pat in f:
                    return os.path.join(root, f)
    return None


# ---------------------------------------------------------------------
# Biological signature dictionaries + community naming
# ---------------------------------------------------------------------

GUT_SET = {
    "Bacteroides","Phocaeicola","Parabacteroides","Prevotella","Segatella",
    "Faecalibacterium","Agathobacter","Roseburia","Blautia","Bifidobacterium",
    "Eubacterium","Anaerostipes","Ruminococcus","Butyricicoccus","Coprococcus",
    "Subdoligranulum","Enterococcus","Escherichia","Klebsiella","Citrobacter",
    "Enterobacter","Morganella","Proteus","Serratia","Bilophila","Fusobacterium",
    "Veillonella","Eggerthella","Clostridium","Lactobacillus","Lactococcus",
    "Turicibacter","Akkermansia","Collinsella","Odoribacter","Barnesiella",
    "Alistipes","Cetobacterium"
}

PATHOGEN_SET = {
    "Salmonella","Shigella","Campylobacter","Listeria","Vibrio","Yersinia",
    "Acinetobacter","Pseudomonas","Stenotrophomonas","Staphylococcus",
    "Streptococcus","Klebsiella","Enterobacter","Citrobacter","Raoultella",
    "Morganella","Proteus","Serratia","Aeromonas","Aliarcobacter","Arcobacter",
    "Neisseria","Haemophilus","Bordetella","Mycobacterium"
}

ENVIRONMENTAL_SET = {
    "Comamonas","Variovorax","Delftia","Cupriavidus","Flavobacterium",
    "Sphingomonas","Sphingopyxis","Paracoccus","Pseudonocardia","Rhodococcus",
    "Rhizobium","Agrobacterium","Acidovorax","Azoarcus","Azospirillum",
    "Brevundimonas","Hydrogenophaga","Janthinobacterium","Methylobacterium",
    "Nitrospira","Nitrosomonas","Nitrosococcus","Phenylobacterium","Gordonia"
}

AQUATIC_SET = {
    "Aeromonas","Aliarcobacter","Arcobacter","Vibrio","Flavobacterium",
    "Shewanella","Plesiomonas","Photobacterium","Psychrobacter","Edwardsiella"
}

BIOFILM_SET = {
    "Thiothrix","Zoogloea","Comamonas","Delftia","Paracoccus","Sphingomonas",
    "Sphingopyxis","Gordonia","Trichococcus","Mycolicibacterium","Nocardia",
    "Brevibacterium","Nitrospira","Acidovorax"
}

SLUDGE_SET = {
    "Methanosaeta","Methanosarcina","Methanobacterium","Clostridium",
    "Bacteroides","Firmicutes","Chloroflexi","Synergistes","Desulfovibrio",
    "Syntrophomonas"
}

ORAL_SET = {
    "Streptococcus","Haemophilus","Actinomyces","Veillonella","Prevotella",
    "Neisseria","Porphyromonas","Fusobacterium","Capnocytophaga",
    "Leptotrichia","Gemella","Granulicatella","Rothia"
}

SKIN_SET = {
    "Staphylococcus","Corynebacterium","Cutibacterium","Micrococcus",
    "Dermacoccus","Kocuria","Propionibacterium","Brevibacterium"
}

HOSPITAL_SET = {
    "Acinetobacter","Pseudomonas","Stenotrophomonas","Klebsiella",
    "Enterobacter","Morganella","Proteus","Serratia","Citrobacter",
    "Elizabethkingia","Burkholderia","Achromobacter"
}

N_CYCLE_SET = {
    "Nitrosomonas","Nitrosospira","Nitrospira","Nitrobacter","Azoarcus",
    "Paracoccus","Alcaligenes","Thauera","Comamonas","Acidovorax",
    "Thiomicrospira","Sulfurimonas","Desulfovibrio","Beggiatoa"
}

FECAL_SET = {
    "Escherichia","Enterococcus","Bacteroides","Clostridium",
    "Parabacteroides","Phocaeicola","Prevotella","Akkermansia"
}

VIRAL_SET = {
    "Myovirus","Siphovirus","Podovirus","Vequintavirus","Septimatrevirus",
    "Carjivirus","T4virus","Lambda-like","CrAssphage","Levivirus"
}


def infer_community_label(genus_list):
    roots = [g.split()[0] for g in genus_list]

    scores = {
        "Pathogen-Enriched": sum(r in PATHOGEN_SET for r in roots),
        "Human Gut–Enriched": sum(r in GUT_SET for r in roots),
        "Fecal Indicator Community": sum(r in FECAL_SET for r in roots),
        "Hospital-Associated": sum(r in HOSPITAL_SET for r in roots),
        "Aquatic Pathogens": sum(r in AQUATIC_SET for r in roots),
        "Wastewater Biofilm": sum(r in BIOFILM_SET for r in roots),
        "Environmental": sum(r in ENVIRONMENTAL_SET for r in roots),
        "Oral/Nasal Microbiome": sum(r in ORAL_SET for r in roots),
        "Skin Microbiome": sum(r in SKIN_SET for r in roots),
        "Anaerobic Sludge": sum(r in SLUDGE_SET for r in roots),
        "Nitrogen/Sulfur Cycle": sum(r in N_CYCLE_SET for r in roots),
        "Viral/Phage-Enriched": sum(r in VIRAL_SET for r in roots)
    }

    # Priority: pathogens
    if scores["Pathogen-Enriched"] >= 2:
        return "Pathogen-Enriched"

    label, max_score = max(scores.items(), key=lambda x: x[1])
    return label if max_score > 0 else "Mixed Community"


# ---------------------------------------------------------------------
# Core plots helpers
# ---------------------------------------------------------------------

def _select_2d_from_embedding(Y: np.ndarray) -> np.ndarray:
    """
    Pick two highest-variance dimensions from embedding Y (n x d).
    Returns n x 2 matrix.
    """
    if Y.shape[1] <= 2:
        return Y
    var = np.var(Y, axis=0)
    idx = np.argsort(var)[-2:]
    return Y[:, idx]

def _create_legend_handles(labels: np.ndarray, cmap_name: str = "tab10"):
    """Create legend handles for cluster labels."""
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap(cmap_name)
    legend_handles = []
    for i, label in enumerate(unique_labels):
        color = cmap(i % 10)
        legend_handles.append(
            Patch(facecolor=color, edgecolor='black',
                  label=f'Community {label+1}', alpha=0.8)
        )
    return legend_handles


# ---------------------------------------------------------------------
# Spectral / UMAP / Network
# ---------------------------------------------------------------------

def plot_spectral_embedding(Y: np.ndarray, labels: np.ndarray, outpath: str):
    Y2 = _select_2d_from_embedding(Y)
    plt.figure(figsize=(8, 6))
    plt.scatter(Y2[:, 0], Y2[:, 1], c=labels, cmap="tab10", s=80, alpha=0.8)
    plt.title("Spectral Embedding (GMM Clusters)")
    plt.xlabel("Spectral Dimension 1")
    plt.ylabel("Spectral Dimension 2")
    legend_handles = _create_legend_handles(labels)
    plt.legend(handles=legend_handles, loc='best', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")


def plot_umap_embedding(Y: np.ndarray, labels: np.ndarray, outpath: str) -> np.ndarray:
    reducer = umap.UMAP(random_state=0, n_neighbors=20, min_dist=0.1)
    Z = reducer.fit_transform(Y)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=80, alpha=0.8)
    plt.title("UMAP Embedding (GMM Clusters)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    legend_handles = _create_legend_handles(labels)
    plt.legend(handles=legend_handles, loc='best', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")
    return Z


def plot_spectral_vs_umap(Y: np.ndarray, Z: np.ndarray, labels: np.ndarray, outpath: str):
    Y2 = _select_2d_from_embedding(Y)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(Y2[:, 0], Y2[:, 1], c=labels, cmap="tab10", s=80, alpha=0.8)
    ax[0].set_title("Spectral Embedding")
    ax[0].set_xlabel("Dim 1")
    ax[0].set_ylabel("Dim 2")

    ax[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=80, alpha=0.8)
    ax[1].set_title("UMAP Embedding")
    ax[1].set_xlabel("UMAP-1")
    ax[1].set_ylabel("UMAP-2")

    legend_handles = _create_legend_handles(labels)
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.02),
               ncol=min(10, len(legend_handles)),
               frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")


def plot_network_graph(A, labels, taxa=None, outpath="network.png", min_degree=3):
    """
    Network graph that uses the SAME taxon sorting as the circular bar plot:
    Sort by (cluster ascending, degree descending).
    """

    A = np.asarray(A)
    labels = np.asarray(labels)
    if taxa is None:
        taxa = np.arange(len(labels))

    degrees = A.sum(axis=1)

    df = pd.DataFrame({
        "taxa": taxa,
        "labels": labels,
        "deg": degrees
    })

    df = df.sort_values(["labels", "deg"], ascending=[True, False]).reset_index(drop=True)

    sorted_idx = df.index.values
    labels_sorted = df["labels"].to_numpy()
    A_sorted = A[sorted_idx][:, sorted_idx]

    unique_clusters = np.unique(labels_sorted)
    cmap = plt.get_cmap("tab10")
    cluster_to_color = {c: cmap(i % 10) for i, c in enumerate(unique_clusters)}

    G = nx.from_numpy_array(A_sorted)

    degrees_sorted = dict(G.degree())
    keep_nodes = [n for n, deg in degrees_sorted.items() if deg >= min_degree]
    H = G.subgraph(keep_nodes).copy()

    pos = nx.spring_layout(H, seed=42, k=0.35)

    node_colors = [cluster_to_color[labels_sorted[n]] for n in H.nodes]

    plt.figure(figsize=(10, 10))
    nx.draw(
        H, pos,
        node_color=node_colors,
        node_size=80,
        edge_color="gray",
        width=0.3,
        alpha=0.9
    )
    plt.title(f"Network Graph (Degree ≥ {min_degree})")
    plt.axis("off")

    legend_handles = []
    for cluster_id in unique_clusters:
        legend_handles.append(
            Patch(facecolor=cluster_to_color[cluster_id], edgecolor='black',
                  label=f'Community {cluster_id+1}', alpha=0.8)
        )

    plt.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(1.15, 1), frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {outpath}")


# ---------------------------------------------------------------------
# Wiley-style circular plot with biological labels (Option A)
# ---------------------------------------------------------------------

def wiley_circular_biological(taxa, labels, adjacency, outpath="wiley_circular_biological.png"):
    """
    Wiley-style circular degree plot with:
      - bars colored by cluster
      - numeric community IDs
      - biological community labels via infer_community_label()
      - top-2 taxa bullet list for each community
    """

    taxa = np.array(taxa)
    labels = np.array(labels)
    A = np.asarray(adjacency)
    degrees = A.sum(axis=1)

    df = pd.DataFrame({
        "genus": taxa,
        "cluster": labels,
        "deg": degrees
    })

    df = df.sort_values(["cluster", "deg"], ascending=[True, False]).reset_index(drop=True)

    n = len(df)
    unique_clusters = sorted(df["cluster"].unique())

    df["angle"] = np.linspace(0, 2*np.pi, n, endpoint=False)
    max_deg = df["deg"].max()
    df["height"] = 0.65 * (df["deg"] / max_deg)

    cmap = plt.get_cmap("tab10")
    cluster_colors = {c: cmap(c % 10) for c in unique_clusters}

    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)

    bar_bottom = 0.25
    arc_bottom = 0.15
    arc_top = 0.80
    bar_width = 2*np.pi / n * 0.9
    delta = (2*np.pi / n) * 0.5

    # Cluster arcs
    for c in unique_clusters:
        sub = df[df["cluster"] == c]
        if sub.empty:
            continue
        start = sub["angle"].min() - delta
        end   = sub["angle"].max() + delta
        width = end - start
        center = start + width/2

        ax.bar(
            center,
            arc_top - arc_bottom,
            width=width,
            bottom=arc_bottom,
            color=cluster_colors[c],
            alpha=0.25,
            edgecolor="none",
            zorder=0
        )

    # Bars
    for _, row in df.iterrows():
        ax.bar(
            row["angle"],
            row["height"],
            width=bar_width,
            bottom=bar_bottom,
            color=cluster_colors[row["cluster"]],
            edgecolor="none",
            alpha=0.90,
            zorder=2
        )

    # Genus labels on top of bars (simple version)
    for _, row in df.iterrows():
        angle   = row["angle"]
        genus   = row["genus"]
        h       = row["height"]
        
        # Position label EXACTLY on top of the bar
        bar_top = bar_bottom + h
        r_label = bar_top  # No offset - directly on top
        
        # Convert angle to degrees
        ang_deg = np.degrees(angle)
        
        # Simpler approach: Just check if we're on left or right side
        # With theta_offset=pi/2 and theta_direction=-1:
        # - angle=0 is at top (12 o'clock)
        # - angle increases clockwise
        
        # Get the actual display angle after transformations
        display_angle = (450 - ang_deg) % 360  # Adjust for polar settings
        
        # Determine if we're on left or right side
        if 0 <= display_angle < 180:
            # Right side of circle
            rotation = display_angle
            va = "bottom"  # Text sits on top of bar
        else:
            # Left side of circle
            rotation = display_angle - 180
            va = "top"  # Text hangs from top of bar
        
        # Simple horizontal alignment
        if 0 <= display_angle < 180:
            ha = "center"
        else:
            ha = "center"
        
        # For better readability, we can slightly adjust position
        # Add small outward shift for labels on very short bars
        if h < 0.1:
            # For very short bars, shift label outward slightly
            outward_shift = 0.015
            r_label = bar_top + outward_shift
        
        # Set font properties
        fontsize = 8
        if len(genus) > 12:  # Longer genus names
            fontsize = 7
        
        ax.text(
            angle,
            r_label,
            genus,
            fontsize=fontsize,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va=va,
            color="black",
            fontweight="medium",
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.1", 
                     facecolor="white", 
                     edgecolor="none",
                     alpha=0.7)  # Optional white background for readability
        )

        # ------------------------------------------------------
    # COMMUNITY LABELS + TOP TWO TAXA (bulleted list)
    # ------------------------------------------------------
    for c in unique_clusters:
        sub = df[df["cluster"] == c]

        # midpoint angle of cluster arc
        theta_mid = 0.5 * (sub["angle"].min() + sub["angle"].max())

        # Determine top two genera by degree
        top_two = list(sub.sort_values("deg", ascending=False)["genus"].head(2))
        
        # Get ALL genera in this cluster for biological inference
        all_genera = sub["genus"].tolist()
        
        # Use infer_community_label() for biological classification
        bio_label = infer_community_label(all_genera)
        
        # Main label using biological classification
        main_label = f"{bio_label} Community"

        # Bullet points of top two taxa
        bullets = "\n".join([f"• {g}" for g in top_two])

        # Combine with line break
        full_label = f"{main_label}\n{bullets}"

        # Position OUTSIDE the circle
        ax.text(
            theta_mid,
            1.25,
            full_label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=0
        )

        # Inner ring cluster number
        ax.text(
            theta_mid,
            0.20,
            c + 1,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")


# ---------------------------------------------------------------------
# Cluster size + degree distribution
# ---------------------------------------------------------------------

def plot_cluster_sizes(labels: np.ndarray, outpath: str):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=unique, y=counts, palette="tab10")
    plt.xlabel("Cluster")
    plt.ylabel("Number of taxa")
    plt.title("Cluster Size Distribution (GMM)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")


def plot_degree_distribution(A: np.ndarray, outpath: str):
    degrees = A.sum(axis=1)
    plt.figure(figsize=(6, 4))
    sns.histplot(degrees, bins=20, kde=False)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution of Co-occurrence Graph")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")


# ---------------------------------------------------------------------
# Correlation heatmap and taxa table
# ---------------------------------------------------------------------

def plot_sorted_correlation_heatmap(rho: np.ndarray, labels: np.ndarray, outpath: str):
    """
    Reorder correlation matrix by GMM cluster and plot heatmap.
    """
    idx = np.argsort(labels)
    rho_sorted = rho[idx][:, idx]

    plt.figure(figsize=(10, 8))
    sns.heatmap(rho_sorted, cmap="vlag", center=0, xticklabels=False, yticklabels=False)
    plt.title("Correlation Heatmap (sorted by GMM cluster)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")


def export_taxa_per_cluster(taxa, labels, outpath: str):
    df = pd.DataFrame({"taxon": taxa, "cluster": labels})
    df.sort_values(["cluster", "taxon"]).to_csv(outpath, sep="\t", index=False)
    print(f"[Saved] {outpath}")


# ---------------------------------------------------------------------
# Simple biological “enrichment” (phage/virus/bacteria/other)
# ---------------------------------------------------------------------

def categorize_taxon(name: str) -> str:
    n = name.lower()
    if "phage" in n or "bacteriophage" in n:
        return "phage"
    if "virus" in n or "cov" in n or "sars" in n:
        return "virus"
    # crude bacterial cue
    if "bacter" in n or "clostridium" in n or "enterobacter" in n:
        return "bacteria"
    return "other"


def export_taxa_enrichment_summary(taxa, labels, outpath: str):
    cats = [categorize_taxon(t) for t in taxa]
    df = pd.DataFrame({"taxon": taxa, "cluster": labels, "category": cats})
    summaries = []

    for c in np.unique(labels):
        sub = df[df["cluster"] == c]
        total = len(sub)
        counts = sub["category"].value_counts()
        for cat, cnt in counts.items():
            summaries.append({
                "cluster": c,
                "category": cat,
                "count": cnt,
                "fraction": cnt / total if total > 0 else 0.0
            })

    out = pd.DataFrame(summaries).sort_values(["cluster", "category"])
    out.to_csv(outpath, sep="\t", index=False)
    print(f"[Saved] {outpath}")


# ---------------------------------------------------------------------
# Louvain vs GMM comparison
# ---------------------------------------------------------------------

def compare_louvain_gmm(A: np.ndarray, labels_gmm: np.ndarray, outdir: str):
    try:
        import community  # python-louvain
    except ImportError:
        print("[WARN] python-louvain not installed; skipping Louvain comparison.")
        return

    G = nx.from_numpy_array(A)
    lv_dict = community.best_partition(G)
    labels_lv = np.array(list(lv_dict.values()))

    cm = pd.crosstab(labels_lv, labels_gmm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=True,
        fmt="d",
        cbar=True,
        linewidths=0.5,
        linecolor="white"
    )

    plt.xlabel("GMM Cluster", fontsize=12)
    plt.ylabel("Louvain Cluster", fontsize=12)
    plt.title("Louvain vs GMM (Confusion Matrix with Counts)", fontsize=14)

    fname = os.path.join(outdir, "louvain_vs_gmm_confusion_counts.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[Saved] {fname}")

    ari = adjusted_rand_score(labels_lv, labels_gmm)
    ari_file = os.path.join(outdir, "louvain_gmm_ARI.txt")
    with open(ari_file, "w") as f:
        f.write(f"Adjusted Rand Index (Louvain vs GMM): {ari:.6f}\n")

    print(f"[INFO] ARI(Louvain, GMM) = {ari:.4f}")


# ---------------------------------------------------------------------
# Silhouette analysis
# ---------------------------------------------------------------------

def plot_silhouette(Y: np.ndarray, labels: np.ndarray, outpath: str):
    """
    Silhouette scores using the embedding Y (all dimensions).
    """
    if len(np.unique(labels)) < 2:
        print("[WARN] Only one cluster; skipping silhouette analysis.")
        return

    s_scores = silhouette_samples(Y, labels)
    s_avg = silhouette_score(Y, labels)

    plt.figure(figsize=(6, 4))
    sns.histplot(s_scores, bins=20, kde=False)
    plt.axvline(s_avg, color="red", linestyle="--", label=f"mean = {s_avg:.2f}")
    plt.xlabel("Silhouette score")
    plt.ylabel("Number of taxa")
    plt.title("Silhouette distribution (GMM clusters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")
    print(f"[INFO] Mean silhouette score = {s_avg:.3f}")


# ---------------------------------------------------------------------
# SBM block probability heatmap (if available)
# ---------------------------------------------------------------------

def plot_sbm_block_heatmap(P: np.ndarray, outpath: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(P, cmap="viridis", annot=False)
    plt.title("SBM Block Probability Matrix P")
    plt.xlabel("Block j")
    plt.ylabel("Block i")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")


# ---------------------------------------------------------------------
# Pathogen-focused embedding plot
# ---------------------------------------------------------------------

def plot_pathogen_embedding(Y: np.ndarray, labels: np.ndarray, taxa, outpath: str):
    Y2 = _select_2d_from_embedding(Y)
    mask = np.array([categorize_taxon(t) in ("virus", "phage") for t in taxa])
    if mask.sum() == 0:
        print("[WARN] No taxa categorized as virus/phage; skipping pathogen plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(Y2[~mask, 0], Y2[~mask, 1], c="lightgray", s=40, alpha=0.3, label="other")
    plt.scatter(Y2[mask, 0], Y2[mask, 1], c=labels[mask], cmap="tab10", s=80, label="virus/phage")
    plt.xlabel("Spectral Dimension 1")
    plt.ylabel("Spectral Dimension 2")
    plt.title("Pathogen-focused view (viruses/phages highlighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")


# ---------------------------------------------------------------------
# Interactive network (PyVis)
# ---------------------------------------------------------------------

def export_interactive_network(A: np.ndarray, labels: np.ndarray, taxa, outpath: str):
    try:
        from pyvis.network import Network
        import matplotlib
    except ImportError:
        print("[WARN] pyvis or matplotlib not installed; skipping interactive network export.")
        return

    G = nx.from_numpy_array(A)
    net = Network(notebook=False, height="800px", width="100%", bgcolor="#ffffff", font_color="black")

    cmap = plt.get_cmap("tab10")
    for i in range(A.shape[0]):
        color = matplotlib.colors.to_hex(cmap(labels[i] % 20))
        net.add_node(i, label=taxa[i], color=color)

    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[0]):
            if A[i, j] == 1:
                net.add_edge(i, j)

    net.show(outpath)
    print(f"[Saved] interactive network HTML: {outpath}")


# ---------------------------------------------------------------------
# Simple spectral “animation” GIF (optional)
# ---------------------------------------------------------------------

def export_spectral_animation(Y: np.ndarray, labels: np.ndarray, outpath: str, frames: int = 10):
    """
    Create a simple animation where we gradually include more spectral dimensions
    and always project down to 2D using the first two selected dims. This is more
    of a visual toy than a strict 'relaxation', but good for talks.
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("[WARN] matplotlib.animation not available; skipping spectral animation.")
        return

    Y2 = _select_2d_from_embedding(Y)
    fig, ax = plt.subplots(figsize=(6, 5))

    scat = ax.scatter(Y2[:, 0], Y2[:, 1], c=labels, cmap="tab10", s=80)
    ax.set_xlabel("Spectral Dimension 1")
    ax.set_ylabel("Spectral Dimension 2")
    ax.set_title("Spectral Embedding (animation)")

    def update(frame):
        alpha = (frame + 1) / frames
        scat.set_alpha(alpha)
        ax.set_title(f"Spectral Embedding (frame {frame+1}/{frames})")
        return scat,

    anim = FuncAnimation(fig, update, frames=frames, interval=400, blit=True)
    writer = PillowWriter(fps=2)
    anim.save(outpath, writer=writer)
    plt.close()
    print(f"[Saved] {outpath}")


# ---------------------------------------------------------------------
# Master function (now with proper taxa–label alignment + circular plot)
# ---------------------------------------------------------------------

def run_advanced_visualization(embedding_path: str,
                               assignments_path: str,
                               taxa_path: str,
                               adjacency_path: str,
                               rho_path: str = None,
                               sbm_block_path: str = None,
                               outdir: str = "visualizations"):
    """
    Main driver: generate all plots and tables into outdir.

    IMPORTANT: We now ALIGN community labels to taxon names by merging:
       graph_taxa_index.tsv  (taxa_path, 1 column: taxon)
       spectral_gmm_assignments.csv (assignments_path, has columns: taxon, community)

    This fixes mismatches between circular plot vs embeddings vs network.
    """
    outdir = ensure_dir(outdir)

    print("[INFO] Loading core files...")
    Y = np.load(embedding_path)
    assign_df = pd.read_csv(assignments_path)          # expects 'taxon', 'community'
    taxa_df = pd.read_csv(taxa_path, header=None, names=["taxon"])
    A = np.loadtxt(adjacency_path, delimiter=",")

    # Align taxa and labels via merge, preserving taxa_df order (graph indexing)
    df = taxa_df.merge(assign_df, on="taxon", how="left")
    if df["community"].isnull().any():
        print("[WARN] Some taxa have missing community labels after merge:")
        print(df[df["community"].isnull()])

    taxa = df["taxon"].tolist()
    labels = df["community"].astype(int).values

    # Core visualizations
    plot_spectral_embedding(Y, labels, os.path.join(outdir, "spectral_embedding.png"))
    Z = plot_umap_embedding(Y, labels, os.path.join(outdir, "umap_embedding.png"))
    plot_spectral_vs_umap(Y, Z, labels, os.path.join(outdir, "spectral_vs_umap.png"))
    plot_network_graph(A, labels, taxa, os.path.join(outdir, "network_graph.png"), min_degree=3)

    # Wiley-style circular biological plot
    wiley_circular_biological(taxa, labels, A, os.path.join(outdir, "wiley_circular_biological.png"))

    # Cluster + graph stats
    plot_cluster_sizes(labels, os.path.join(outdir, "cluster_sizes.png"))
    plot_degree_distribution(A, os.path.join(outdir, "degree_distribution.png"))

    # Correlation heatmap
    if rho_path is not None and os.path.exists(rho_path):
        rho = np.load(rho_path)
        plot_sorted_correlation_heatmap(rho, labels, os.path.join(outdir, "sorted_correlation_heatmap.png"))
    else:
        print("[WARN] rho.npy not found; skipping correlation heatmap.")

    # Taxa tables + simple enrichment
    export_taxa_per_cluster(taxa, labels, os.path.join(outdir, "taxa_per_cluster.tsv"))
    export_taxa_enrichment_summary(taxa, labels, os.path.join(outdir, "taxa_enrichment_summary.tsv"))

    # Silhouette analysis
    plot_silhouette(Y, labels, os.path.join(outdir, "silhouette_scores.png"))

    # Louvain vs GMM
    compare_louvain_gmm(A, labels, outdir)

    # SBM block matrix (if available)
    if sbm_block_path is not None and os.path.exists(sbm_block_path):
        P = pd.read_csv(sbm_block_path, index_col=0).values
        plot_sbm_block_heatmap(P, os.path.join(outdir, "sbm_block_heatmap.png"))
    else:
        print("[INFO] SBM block matrix not provided; skipping sbm_block_heatmap.")

    # Pathogen-focused embedding
    plot_pathogen_embedding(Y, labels, taxa, os.path.join(outdir, "pathogens_embedding.png"))

    # Optional animation
    export_spectral_animation(Y, labels, os.path.join(outdir, "spectral_animation.gif"))

    print("[INFO] Advanced visualization complete.")
    print(f"[INFO] Outputs in: {os.path.abspath(outdir)}")


# ---------------------------------------------------------------------
# Auto-visualize wrapper
# ---------------------------------------------------------------------

def auto_visualize(project_out_dir: str, outdir: str = "visualizations"):
    """
    Automatically find:
      - spectral_embedding.npy
      - spectral_gmm_assignments.csv
      - graph_taxa_index.tsv
      - graph_adjacency.csv
      - rho.npy (optional)
      - sbm_block_P.csv (optional)
    under project_out_dir, then run full visualization.
    """
    base = os.path.abspath(project_out_dir)
    print(f"[INFO] Auto-searching under: {base}")

    embedding = auto_find_file(base, ["spectral_embedding.npy"])
    assignments = auto_find_file(base, ["spectral_gmm_assignments.csv"])
    taxa = auto_find_file(base, ["graph_taxa_index.tsv"])
    adjacency = auto_find_file(base, ["graph_adjacency.csv"])
    rho = auto_find_file(base, ["rho.npy"])
    sbm_block = auto_find_file(base, ["sbm_block_P.csv", "sbm_block_P"])

    missing = []
    if embedding is None: missing.append("spectral_embedding.npy")
    if assignments is None: missing.append("spectral_gmm_assignments.csv")
    if taxa is None: missing.append("graph_taxa_index.tsv")
    if adjacency is None: missing.append("graph_adjacency.csv")

    if missing:
        raise FileNotFoundError(f"[ERROR] Could not find required files: {missing}")

    print(f"[INFO] Using embedding:  {embedding}")
    print(f"[INFO] Using assignments: {assignments}")
    print(f"[INFO] Using taxa index: {taxa}")
    print(f"[INFO] Using adjacency:  {adjacency}")
    print(f"[INFO] Using rho:        {rho}")
    print(f"[INFO] Using SBM block:  {sbm_block}")

    run_advanced_visualization(
        embedding_path=embedding,
        assignments_path=assignments,
        taxa_path=taxa,
        adjacency_path=adjacency,
        rho_path=rho,
        sbm_block_path=sbm_block,
        outdir=outdir,
    )
