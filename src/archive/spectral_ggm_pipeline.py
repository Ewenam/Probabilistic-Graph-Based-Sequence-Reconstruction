# #!/usr/bin/env python3
# """
# Spectral + Gaussian Mixture Model community detection for wastewater taxa.
# Inputs:
#   - rho.npy: correlation matrix from CLR + Spearman
#   - graph_taxa_index.tsv: list of taxa (one per row)

# Outputs:
#   - spectral_gmm_assignments.csv (hard labels)
#   - spectral_gmm_soft_assignments.csv (soft probabilities)
  
  
# Usage:
#   python spectral_ggm_pipeline.py --rho path/to/rho.npy \
#                                   --taxa path/to/graph_taxa_index.tsv \
#                                   --outdir path/to/output_dir \
#                                   --embed_dim 10 \
#                                   --Kmax 10
# """

# import argparse
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from scipy.sparse.linalg import eigsh
# from sklearn.mixture import GaussianMixture


# def load_inputs(rho_path: Path, taxa_path: Path):
#     rho = np.load(rho_path)
#     taxa = pd.read_csv(taxa_path, header=None)[0].tolist()

#     if rho.shape[0] != len(taxa):
#         raise ValueError(
#             f"Mismatch: rho has {rho.shape[0]} taxa but index file has {len(taxa)} names"
#         )

#     return rho, taxa


# def build_similarity(rho):
#     """
#     Convert correlation matrix to positive similarity.
#     Negative correlations are set to 0.
#     """
#     W = np.maximum(rho, 0.0)
#     np.fill_diagonal(W, 0.0)
#     return W


# def spectral_embedding(W, m=10):
#     """
#     Compute normalized Laplacian eigenvectors for embedding.
#     """
#     n = W.shape[0]
#     d = W.sum(axis=1)

#     # Avoid division by zero
#     d_safe = np.where(d > 0, d, 1.0)
#     D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))

#     I = np.eye(n)
#     L = I - D_inv_sqrt @ W @ D_inv_sqrt

#     # Compute the lowest-m eigenvectors
#     vals, vecs = eigsh(L, k=m, which="SM")

#     # Normalize rows
#     Y = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
#     return Y


# def fit_gmm(Y, Ks=range(2, 10)):
#     best_bic = np.inf
#     best_model = None
#     best_K = None

#     for K in Ks:
#         gmm = GaussianMixture(
#             n_components=K,
#             covariance_type="full",
#             random_state=0
#         )
#         gmm.fit(Y)
#         bic = gmm.bic(Y)

#         print(f"[INFO] K={K}, BIC={bic:.2f}")

#         if bic < best_bic:
#             best_bic = bic
#             best_model = gmm
#             best_K = K

#     print(f"\n[RESULT] Best K according to BIC: {best_K}")
#     return best_model, best_K


# def main():
#     p = argparse.ArgumentParser(description="Spectral + GMM community detector")
#     p.add_argument("--rho", required=True, help="Path to rho.npy")
#     p.add_argument("--taxa", required=True, help="Path to graph_taxa_index.tsv")
#     p.add_argument("--outdir", required=True, help="Output directory")
#     p.add_argument("--embed_dim", type=int, default=10, help="Spectral embedding dimension")
#     p.add_argument("--Kmax", type=int, default=10, help="Maximum K for GMM")
#     args = p.parse_args()

#     outdir = Path(args.outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     # 1. Load rho + taxa
#     rho, taxa = load_inputs(Path(args.rho), Path(args.taxa))

#     # 2. Similarity matrix
#     W = build_similarity(rho)

#     # 3. Spectral embedding
#     print("[INFO] Computing spectral embedding...")
#     Y = spectral_embedding(W, m=args.embed_dim)

#     # 4. Fit GMM with BIC model selection
#     print("[INFO] Fitting Gaussian Mixture Models...")
#     Ks = range(2, args.Kmax + 1)
#     gmm, best_K = fit_gmm(Y, Ks)

#     # 5. Predict communities
#     labels = gmm.predict(Y)
#     probs = gmm.predict_proba(Y)

#     # 6. Save hard assignments
#     hard_df = pd.DataFrame({
#         "taxon": taxa,
#         "community": labels
#     })
#     hard_df.to_csv(outdir / "spectral_gmm_assignments.csv", index=False)
#     print(f"[INFO] Saved hard labels to {outdir/'spectral_gmm_assignments.csv'}")

#     # 7. Save soft probabilities
#     soft_df = pd.DataFrame(probs, columns=[f"k{k}" for k in range(best_K)])
#     soft_df.insert(0, "taxon", taxa)
#     soft_df.to_csv(outdir / "spectral_gmm_soft_assignments.csv", index=False)
#     print(f"[INFO] Saved soft labels to {outdir/'spectral_gmm_soft_assignments.csv'}")

#     print("\n[FINISHED] Spectral–GMM community inference complete.")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Spectral + Gaussian Mixture Model community detection for wastewater taxa.
This version FIXES the spectral embedding collapse by:
  - Automatically selecting meaningful eigenvectors (non-flat)
  - Ignoring eigenvectors with ~zero variance (EV1, EV2)
  - Saving the correct spectral embedding for visualization

Outputs:
  - spectral_embedding.npy
  - spectral_gmm_assignments.csv
  - spectral_gmm_soft_assignments.csv
  - spectral_gmm_selected_eigenvectors.txt
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse.linalg import eigsh
from sklearn.mixture import GaussianMixture


# ------------------------------------------------------------
# Load inputs
# ------------------------------------------------------------
def load_inputs(rho_path: Path, taxa_path: Path):
    rho = np.load(rho_path)
    taxa = pd.read_csv(taxa_path, header=None)[0].tolist()

    if rho.shape[0] != len(taxa):
        raise ValueError(
            f"Mismatch: rho has {rho.shape[0]} taxa but index file has {len(taxa)} names."
        )
    return rho, taxa


# ------------------------------------------------------------
# Build similarity (positive correlation only)
# ------------------------------------------------------------
def build_similarity(rho):
    W = np.maximum(rho, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


# ------------------------------------------------------------
# FIXED SPECTRAL EMBEDDING
# (Auto-selects meaningful eigenvectors)
# ------------------------------------------------------------
def stable_spectral_embedding(W, max_dims=10):
    n = W.shape[0]
    d = W.sum(axis=1)

    # Avoid zero division
    d_safe = np.where(d > 0, d, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))

    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    # Compute bottom eigenvectors of L
    vals, vecs = eigsh(L, k=max_dims, which="SM")

    # Sort eigenvectors by eigenvalue
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Compute variance per eigenvector
    variances = np.var(vecs, axis=0)

    # Select eigenvectors with meaningful variance
    usable = [i for i, v in enumerate(variances) if v > 1e-4]

    if len(usable) < 2:
        raise RuntimeError(
            "Not enough meaningful eigenvectors found. Graph has extremely weak structure."
        )

    selected = usable[:min(len(usable), max_dims)]
    Y = vecs[:, selected]

    print("\n[INFO] Eigenvector variances:", variances)
    print("[INFO] Selected eigenvectors:", selected)
    print("[INFO] Spectral embedding shape:", Y.shape)

    return Y, selected


# ------------------------------------------------------------
# Fit GMM with model selection
# ------------------------------------------------------------
def fit_gmm(Y, Ks=range(2, 10)):
    best_bic = np.inf
    best_model = None
    best_K = None

    for K in Ks:
        gmm = GaussianMixture(
            n_components=K, covariance_type="full", random_state=0
        )
        gmm.fit(Y)
        bic = gmm.bic(Y)
        print(f"[INFO] K={K}, BIC={bic:.3f}")

        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_K = K

    print(f"\n[RESULT] Best K according to BIC: {best_K}")
    return best_model, best_K


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Spectral + GMM community detector (stable eigenvectors)")
    p.add_argument("--rho", required=True, help="Path to rho.npy")
    p.add_argument("--taxa", required=True, help="Path to graph_taxa_index.tsv")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--embed_dim", type=int, default=10, help="Max spectral embedding dimension")
    p.add_argument("--Kmax", type=int, default=10, help="Max K for GMM")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load rho + taxa
    rho, taxa = load_inputs(Path(args.rho), Path(args.taxa))

    # 2. Similarity matrix (positive corr only)
    W = build_similarity(rho)

    # 3. Stable spectral embedding (auto eigenvector selection)
    Y, selected_eigs = stable_spectral_embedding(W, max_dims=args.embed_dim)

    # Save embedding + eigenvector info
    np.save(outdir / "spectral_embedding.npy", Y)
    with open(outdir / "spectral_gmm_selected_eigenvectors.txt", "w") as f:
        f.write("Selected eigenvector indices:\n")
        f.write(", ".join(map(str, selected_eigs)) + "\n")

    # 4. GMM with BIC model selection
    print("[INFO] Fitting Gaussian Mixture Models...")
    Ks = range(2, args.Kmax + 1)
    gmm, best_K = fit_gmm(Y, Ks)

    # 5. Predict communities
    labels = gmm.predict(Y)
    probs = gmm.predict_proba(Y)

    # 6. Save hard assignments
    hard_df = pd.DataFrame({"taxon": taxa, "community": labels})
    hard_df.to_csv(outdir / "spectral_gmm_assignments.csv", index=False)

    # 7. Save soft probabilities
    soft_df = pd.DataFrame(probs, columns=[f"k{k}" for k in range(best_K)])
    soft_df.insert(0, "taxon", taxa)
    soft_df.to_csv(outdir / "spectral_gmm_soft_assignments.csv", index=False)

    print("\n[FINISHED] Spectral–GMM inference complete.")
    print(f"Embedding saved to {outdir/'spectral_embedding.npy'}")


if __name__ == "__main__":
    main()
