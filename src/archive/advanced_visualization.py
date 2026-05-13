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
  - louvain_vs_gmm_confusion.png
  - louvain_gmm_ARI.txt
  - silhouette_scores.png
  - sbm_block_heatmap.png            (if sbm_block_P.csv found)
  - pathogens_embedding.png
  - interactive_network.html         (if pyvis is installed)
  - spectral_animation.gif           (simple eigenvector build-up animation)
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


# # ---------------------------------------------------------------------
# # Core plots
# # ---------------------------------------------------------------------

# def _select_2d_from_embedding(Y: np.ndarray) -> np.ndarray:
#     """
#     Pick two highest-variance dimensions from embedding Y (n x d).
#     Returns n x 2 matrix.
#     """
#     if Y.shape[1] <= 2:
#         return Y
#     var = np.var(Y, axis=0)
#     idx = np.argsort(var)[-2:]
#     return Y[:, idx]


# def plot_spectral_embedding(Y: np.ndarray, labels: np.ndarray, outpath: str):
#     Y2 = _select_2d_from_embedding(Y)
#     plt.figure(figsize=(8, 6))
#     plt.scatter(Y2[:, 0], Y2[:, 1], c=labels, cmap="tab10", s=80)
#     plt.title("Spectral Embedding (GMM Clusters)")
#     plt.xlabel("Spectral Dimension 1")
#     plt.ylabel("Spectral Dimension 2")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=300)
#     plt.close()
#     print(f"[Saved] {outpath}")


# def plot_umap_embedding(Y: np.ndarray, labels: np.ndarray, outpath: str) -> np.ndarray:
#     reducer = umap.UMAP(random_state=0, n_neighbors=20, min_dist=0.1)
#     Z = reducer.fit_transform(Y)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=80)
#     plt.title("UMAP Embedding (GMM Clusters)")
#     plt.xlabel("UMAP-1")
#     plt.ylabel("UMAP-2")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=300)
#     plt.close()
#     print(f"[Saved] {outpath}")
#     return Z


# def plot_spectral_vs_umap(Y: np.ndarray, Z: np.ndarray, labels: np.ndarray, outpath: str):
#     Y2 = _select_2d_from_embedding(Y)
#     fig, ax = plt.subplots(1, 2, figsize=(14, 6))

#     ax[0].scatter(Y2[:, 0], Y2[:, 1], c=labels, cmap="tab10", s=80)
#     ax[0].set_title("Spectral Embedding")
#     ax[0].set_xlabel("Dim 1")
#     ax[0].set_ylabel("Dim 2")

#     ax[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=80)
#     ax[1].set_title("UMAP Embedding")
#     ax[1].set_xlabel("UMAP-1")
#     ax[1].set_ylabel("UMAP-2")

#     plt.tight_layout()
#     plt.legend()
#     plt.savefig(outpath, dpi=300)
#     plt.close()
#     print(f"[Saved] {outpath}")


# # def plot_network_graph(A: np.ndarray, labels: np.ndarray, outpath: str, min_degree: int = 3):
# #     """
# #     Force-directed layout of the adjacency graph, showing only nodes with degree >= min_degree.
# #     Colors = GMM communities.
# #     """
# #     G = nx.from_numpy_array(A)
# #     degrees = dict(G.degree())
# #     keep_nodes = [n for n, deg in degrees.items() if deg >= min_degree]
# #     H = G.subgraph(keep_nodes).copy()

# #     pos = nx.spring_layout(H, seed=42, k=0.35)
# #     node_colors = [labels[n] for n in H.nodes]

# #     plt.figure(figsize=(10, 10))
# #     nx.draw(
# #         H, pos,
# #         node_color=node_colors,
# #         node_size=80,
# #         cmap="tab10",
# #         edge_color="gray",
# #         width=0.3,
# #         alpha=0.9
# #     )
# #     plt.title(f"Network Graph (Degree ≥ {min_degree})")
# #     plt.axis("off")
# #     plt.tight_layout()
# #     plt.savefig(outpath, dpi=300)
# #     plt.close()
# #     print(f"[Saved] {outpath}")

# def plot_network_graph(A, labels, taxa=None, outpath="network.png", min_degree=3):
#     """
#     Network graph that uses the SAME taxon sorting as the circular bar plot:
#     Sort by (cluster ascending, degree descending).
#     """

#     # Convert data
#     A = np.asarray(A)
#     labels = np.asarray(labels)
#     if taxa is None:
#         taxa = np.arange(len(labels))

#     # ---- Compute degree ----
#     degrees = A.sum(axis=1)

#     # ---- Build DataFrame identical to circular plot ----
#     df = pd.DataFrame({
#         "taxa": taxa,
#         "labels": labels,
#         "deg": degrees
#     })

#     # ---- SORT EXACTLY AS IN THE CIRCULAR PLOT ----
#     df = df.sort_values(["labels", "deg"], ascending=[True, False]).reset_index(drop=True)

#     sorted_idx = df.index.values
#     labels_sorted = df["labels"].to_numpy()
#     taxa_sorted = df["taxa"].to_numpy()
#     A_sorted = A[sorted_idx][:, sorted_idx]

#     # ---- FIXED COLOR MAP based on unique cluster IDs ----
#     unique_clusters = np.unique(labels_sorted)
#     cmap = plt.get_cmap("tab10")
#     cluster_to_color = {c: cmap(i % 10) for i, c in enumerate(unique_clusters)}

#     # ---- Build Graph ----
#     G = nx.from_numpy_array(A_sorted)

#     # ---- Keep only nodes with degree >= min_degree ----
#     degrees_sorted = dict(G.degree())
#     keep_nodes = [n for n, deg in degrees_sorted.items() if deg >= min_degree]
#     H = G.subgraph(keep_nodes).copy()

#     # ---- Layout ----
#     pos = nx.spring_layout(H, seed=42, k=0.35)

#     # ---- Colors (using FULL sorted mapping, NOT re-indexed subset) ----
#     node_colors = [cluster_to_color[labels_sorted[n]] for n in H.nodes]

#     # ---- Draw ----
#     plt.figure(figsize=(10, 10))
#     nx.draw(
#         H, pos,
#         node_color=node_colors,
#         node_size=80,
#         edge_color="gray",
#         width=0.3,
#         alpha=0.9
#     )
#     plt.title(f"Network Graph (Degree ≥ {min_degree})")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=300)
#     plt.close()
#     print(f"[Saved] {outpath}")




# ---------------------------------------------------------------------
# Core plots
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
    
    # Create legend handles
    legend_handles = []
    for i, label in enumerate(unique_labels):
        color = cmap(i % 10)
        legend_handles.append(
            Patch(facecolor=color, edgecolor='black', 
                  label=f'Community {label+1}', alpha=0.8)
        )
    
    return legend_handles

def plot_spectral_embedding(Y: np.ndarray, labels: np.ndarray, outpath: str):
    Y2 = _select_2d_from_embedding(Y)
    plt.figure(figsize=(8, 6))
    
    # Create scatter plot
    scatter = plt.scatter(Y2[:, 0], Y2[:, 1], c=labels, cmap="tab10", s=80, alpha=0.8)
    
    plt.title("Spectral Embedding (GMM Clusters)")
    plt.xlabel("Spectral Dimension 1")
    plt.ylabel("Spectral Dimension 2")
    
    # Add legend
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
    
    # Add legend
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

    # Spectral embedding plot
    sc1 = ax[0].scatter(Y2[:, 0], Y2[:, 1], c=labels, cmap="tab10", s=80, alpha=0.8)
    ax[0].set_title("Spectral Embedding")
    ax[0].set_xlabel("Dim 1")
    ax[0].set_ylabel("Dim 2")
    
    # UMAP plot
    sc2 = ax[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=80, alpha=0.8)
    ax[1].set_title("UMAP Embedding")
    ax[1].set_xlabel("UMAP-1")
    ax[1].set_ylabel("UMAP-2")
    
    # Create a single legend for both plots
    legend_handles = _create_legend_handles(labels)
    fig.legend(handles=legend_handles, loc='upper center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(10, len(legend_handles)), 
               frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Make room for bottom legend
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")

def plot_network_graph(A, labels, taxa=None, outpath="network.png", min_degree=3):
    """
    Network graph that uses the SAME taxon sorting as the circular bar plot:
    Sort by (cluster ascending, degree descending).
    """

    # Convert data
    A = np.asarray(A)
    labels = np.asarray(labels)
    if taxa is None:
        taxa = np.arange(len(labels))

    # ---- Compute degree ----
    degrees = A.sum(axis=1)

    # ---- Build DataFrame identical to circular plot ----
    df = pd.DataFrame({
        "taxa": taxa,
        "labels": labels,
        "deg": degrees
    })

    # ---- SORT EXACTLY AS IN THE CIRCULAR PLOT ----
    df = df.sort_values(["labels", "deg"], ascending=[True, False]).reset_index(drop=True)

    sorted_idx = df.index.values
    labels_sorted = df["labels"].to_numpy()
    taxa_sorted = df["taxa"].to_numpy()
    A_sorted = A[sorted_idx][:, sorted_idx]

    # ---- FIXED COLOR MAP based on unique cluster IDs ----
    unique_clusters = np.unique(labels_sorted)
    cmap = plt.get_cmap("tab10")
    cluster_to_color = {c: cmap(i % 10) for i, c in enumerate(unique_clusters)}

    # ---- Build Graph ----
    G = nx.from_numpy_array(A_sorted)

    # ---- Keep only nodes with degree >= min_degree ----
    degrees_sorted = dict(G.degree())
    keep_nodes = [n for n, deg in degrees_sorted.items() if deg >= min_degree]
    H = G.subgraph(keep_nodes).copy()

    # ---- Layout ----
    pos = nx.spring_layout(H, seed=42, k=0.35)

    # ---- Colors (using FULL sorted mapping, NOT re-indexed subset) ----
    node_colors = [cluster_to_color[labels_sorted[n]] for n in H.nodes]

    # ---- Draw ----
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
    
    # Add legend for network graph
    legend_handles = []
    for cluster_id in unique_clusters:
        legend_handles.append(
            Patch(facecolor=cluster_to_color[cluster_id], edgecolor='black',
                  label=f'Community {cluster_id}', alpha=0.8)
        )
    
    plt.legend(handles=legend_handles, loc='upper right', 
               bbox_to_anchor=(1.15, 1), frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
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
# Simple biological “enrichment”
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

# def compare_louvain_gmm(A: np.ndarray, labels_gmm: np.ndarray, outdir: str):
#     try:
#         import community  # python-louvain
#     except ImportError:
#         print("[WARN] python-louvain not installed; skipping Louvain comparison.")
#         return

#     G = nx.from_numpy_array(A)
#     lv_dict = community.best_partition(G)
#     labels_lv = np.array(list(lv_dict.values()))

#     # confusion matrix
#     cm = pd.crosstab(labels_lv, labels_gmm)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, cmap="viridis", cbar=True, xticklabels=True, yticklabels=True)
#     plt.xlabel("GMM cluster")
#     plt.ylabel("Louvain cluster")
#     plt.title("Louvain vs GMM (confusion matrix)")
#     fname = os.path.join(outdir, "louvain_vs_gmm_confusion.png")
#     plt.tight_layout()
#     plt.savefig(fname, dpi=300)
#     plt.close()
#     print(f"[Saved] {fname}")

#     ari = adjusted_rand_score(labels_lv, labels_gmm)
#     with open(os.path.join(outdir, "louvain_gmm_ARI.txt"), "w") as f:
#         f.write(f"Adjusted Rand Index (Louvain vs GMM): {ari:.6f}\n")
#     print(f"[INFO] ARI(Louvain, GMM) = {ari:.4f}")


def compare_louvain_gmm(A: np.ndarray, labels_gmm: np.ndarray, outdir: str):
    try:
        import community  # python-louvain
    except ImportError:
        print("[WARN] python-louvain not installed; skipping Louvain comparison.")
        return

    # --- Run Louvain ---
    G = nx.from_numpy_array(A)
    lv_dict = community.best_partition(G)
    labels_lv = np.array(list(lv_dict.values()))

    # --- Confusion Matrix ---
    cm = pd.crosstab(labels_lv, labels_gmm)

    # --- Plot Confusion Matrix with Counts ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        cmap="Blues",        # <-- light colormap
        annot=True,          # <-- write counts inside the boxes
        fmt="d",             # <-- integer format for counts
        cbar=True,
        linewidths=0.5,
        linecolor="white"
    )

    plt.xlabel("GMM Cluster", fontsize=12)
    plt.ylabel("Louvain Cluster", fontsize=12)
    plt.title("Louvain vs GMM (Confusion Matrix with Counts)", fontsize=14)

    # --- Save Figure ---
    fname = os.path.join(outdir, "louvain_vs_gmm_confusion_counts.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[Saved] {fname}")

    # --- Adjusted Rand Index ---
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
    except ImportError:
        print("[WARN] pyvis not installed; skipping interactive network export.")
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
        # just vary alpha / size to give sense of transition
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
# Master function
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
    """
    outdir = ensure_dir(outdir)

    print("[INFO] Loading core files...")
    Y = np.load(embedding_path)
    labels = pd.read_csv(assignments_path)["community"].values
    taxa = pd.read_csv(taxa_path, header=None)[0].tolist()
    A = np.loadtxt(adjacency_path, delimiter=",")

    # Core visualizations
    plot_spectral_embedding(Y, labels, os.path.join(outdir, "spectral_embedding.png"))
    Z = plot_umap_embedding(Y, labels, os.path.join(outdir, "umap_embedding.png"))
    plot_spectral_vs_umap(Y, Z, labels, os.path.join(outdir, "spectral_vs_umap.png"))
    plot_network_graph(A, labels, os.path.join(outdir, "network_graph.png"), min_degree=3)

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

    # Optional animation (can be commented out if you don't want GIF)
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
