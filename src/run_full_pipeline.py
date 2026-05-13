#!/usr/bin/env python3
"""
End-to-end wastewater pipeline:
FASTA -> Kraken2/Bracken -> samples×taxa matrix -> CLR -> associations -> graph
-> SBM (EM) -> outputs

Now includes:
- Exact SBM(n, p, W) simulator (sample_sbm)
- Optional --simulate_sbm mode to override the empirical graph with a synthetic SBM

Usage:
python  src/run_full_pipeline.py     
        --input_dir "/home/users/razumah1/Desktop/AAB Project/data/fastas"     
        --kraken_db "/home/users/razumah1"     
        --outdir "/home/users/razumah1/Desktop/AAB Project/results/project_out/pipeline_results_genus_fixed"     
        --rank species     --threads 16     
        --confidence 0.1     --min_prevalence 0.2     
        --min_total 20     --tau 0.6     
        --Kmin 2     --Kmax 12
"""

import argparse, sys, shutil, subprocess, math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# ------------------------------------------------------------
# EXACT SBM(n, p, W) SIMULATOR — matches Bernoulli SBM definition
# ------------------------------------------------------------
def sample_sbm(n, p, W, seed=None):
    """
    Sample a Stochastic Block Model graph exactly as defined in the Bernoulli SBM:

    - Community labels X_v ~ Categorical(p)
    - Edges A_uv ~ Bernoulli(W_{X_u, X_v}) independently for u < v

    Args:
        n  : number of vertices
        p  : 1D array of length K, community prior probabilities (sum to 1)
        W  : (K,K) symmetric matrix of connection probabilities in [0,1]
        seed : random seed (int or None)

    Returns:
        X : (n,) community labels in {0,...,K-1}
        A : (n,n) adjacency matrix (undirected, no self-loops)
    """
    rng = np.random.default_rng(seed)
    K = len(p)

    # Step 1: draw community assignments
    X = rng.choice(K, size=n, p=p)

    # Step 2: generate adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for u in range(n):
        for v in range(u + 1, n):
            prob = W[X[u], X[v]]
            A[u, v] = rng.random() < prob
            A[v, u] = A[u, v]

    return X, A

# ---------------------------
# Helpers: system + IO
# ---------------------------
def have_bin(name: str) -> bool:
    return shutil.which(name) is not None

def run_cmd(cmd):
    subprocess.run(cmd, check=True)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Kraken/Bracken parsing
# ---------------------------
RANK_CODE = {
    "species": "S",
    "genus": "G",
    "family": "F",
    "order": "O",
    "class": "C",
    "phylum": "P",
    "kingdom": "K",
    "domain": "D"
}

def parse_kraken_report(report_path: Path, target_rank_code: str) -> pd.Series:
    """
    Parse a Kraken2 report and extract read counts for a specific rank (e.g., genus, species).
    Returns a pandas Series with taxon names as index and read counts as values.
    """
    taxa_counts = {}
    with report_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Kraken2 report format:
            # percent, reads_clade, reads_taxon, rank_code, taxid, name
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            rank_code = parts[3].strip()
            if rank_code == target_rank_code:
                try:
                    reads_taxon = int(parts[2])
                except ValueError:
                    continue

                taxon_name = parts[5].strip()
                if taxon_name and not taxon_name.lower().startswith("unclassified"):
                    taxa_counts[taxon_name] = taxa_counts.get(taxon_name, 0) + reads_taxon

    if not taxa_counts:
        print(f"[WARN] No taxa found at rank {target_rank_code} in {report_path.name}")
    else:
        print(f"[INFO] Parsed {len(taxa_counts)} taxa at rank {target_rank_code} from {report_path.name}")
    return pd.Series(taxa_counts, dtype="int64")


def classify_folder_to_matrix(input_dir: Path, patterns, kraken_db: Path,
                              outdir: Path, rank: str, threads: int,
                              confidence: float, use_bracken: bool, bracken_db: Path,
                              keep_intermediate: bool) -> pd.DataFrame:
    if not have_bin("kraken2"):
        print("[ERROR] kraken2 not found in PATH.", file=sys.stderr)
        sys.exit(1)
    if use_bracken and not have_bin("bracken"):
        print("[WARN] bracken not found; proceeding without Bracken.", file=sys.stderr)
        use_bracken = False

    reports_dir = outdir / "reports"
    bracken_dir = outdir / "bracken"
    safe_mkdir(outdir)
    safe_mkdir(reports_dir)
    if use_bracken:
        safe_mkdir(bracken_dir)

    # find FASTA files
    fasta_files = []
    for pat in patterns:
        fasta_files.extend(sorted(input_dir.glob(pat)))
    fasta_files = sorted(set(fasta_files))
    if not fasta_files:
        print("[ERROR] No FASTA files matched patterns.", file=sys.stderr)
        sys.exit(1)

    target_rank_code = RANK_CODE[rank]
    per_sample = {}

    for fa in tqdm(fasta_files, desc="Classifying samples"):
        sample = fa.stem
        report = reports_dir / f"{sample}.kreport"
        kr_out = reports_dir / f"{sample}.kraken.out"
        cmd = [
            "kraken2",
            "--db", str(kraken_db),
            "--threads", str(threads),
            "--report", str(report),
            "--output", str(kr_out),
            "--confidence", str(confidence),
            str(fa)
        ]
        run_cmd(cmd)

        if use_bracken:
            # Bracken - refine counts (expects matching DB)
            level = {"species": "S", "genus": "G", "family": "F"}.get(rank, "S")
            br_out = bracken_dir / f"{sample}.bracken"
            cmdb = [
                "bracken",
                "-d", str(bracken_db if bracken_db else kraken_db),
                "-i", str(report),
                "-o", str(br_out),
                "-r", "100",
                "-l", level
            ]
            run_cmd(cmdb)
            # parse bracken TSV
            dfb = pd.read_csv(br_out, sep="\t")
            s = pd.Series(dfb["new_est_reads"].values,
                          index=dfb["name"].values,
                          dtype="float64")
            s = s[s > 0].astype("int64")
            per_sample[sample] = s
            if not keep_intermediate:
                try:
                    kr_out.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            s = parse_kraken_report(report, target_rank_code)
            per_sample[sample] = s
            if not keep_intermediate:
                try:
                    kr_out.unlink(missing_ok=True)
                    report.unlink(missing_ok=True)
                except Exception:
                    pass

    df = pd.DataFrame(per_sample).fillna(0).astype("int64")
    df.index.name = f"{rank}_taxon"
    return df

# ---------------------------
# Preprocess → CLR + associations + graph
# ---------------------------
def clr_transform(X: np.ndarray, pseudo=1.0):
    X = X.astype(float) + pseudo
    gm = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
    return np.log(X / gm)

def bh_fdr(pvals: np.ndarray, alpha=0.05):
    """Benjamini-Hochberg mask of significant entries for upper triangular."""
    m = pvals.size
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    thresh = (ranks / m) * alpha
    return pvals <= thresh

def build_graph_from_counts(counts_df: pd.DataFrame,
                            min_prevalence=0.1,
                            min_total=50,
                            fdr_alpha=0.05,
                            tau=0.45):
    """
    Build an association graph from taxa × samples count matrix.
    Returns adjacency matrix (A), taxa list, rho, and pval matrices.
    """

    df = counts_df.copy()

    # If rows are taxa (more rows than columns = samples), transpose so columns = taxa
    if df.shape[0] > df.shape[1]:
        df = df.T  # now samples × taxa

    # --- Filter taxa ---
    present = (df > 0).sum(axis=0) / df.shape[0] >= min_prevalence
    enough = df.sum(axis=0) >= min_total
    keep = present & enough
    df = df.loc[:, keep]
    taxa = df.columns.to_list()

    if len(taxa) < 3:
        raise ValueError(f"Too few taxa after filtering ({len(taxa)} remain).")

    # --- Normalize via CLR ---
    Xclr = clr_transform(df.values)  # samples × taxa

    # --- Compute correlation ---
    rho, pval = spearmanr(Xclr, axis=0)
    n = len(taxa)
    rho = np.array(rho)[:n, :n]
    pval = np.array(pval)[:n, :n]

    # --- Significance and thresholding ---
    iu = np.triu_indices(n, 1)
    # If you want BH-FDR, uncomment this:
    # sig_mask = bh_fdr(pval[iu], alpha=fdr_alpha)
    sig_mask = np.ones_like(pval[iu], dtype=bool)
    tau_mask = np.abs(rho[iu]) >= tau
    sel = sig_mask & tau_mask

    A = np.zeros((n, n), dtype=np.int8)
    A[iu[0][sel], iu[1][sel]] = 1
    A = A + A.T
    np.fill_diagonal(A, 0)

    return A, taxa, rho, pval

# ---------------------------
# SBM (Variational EM on graph)
# ---------------------------
def _safe_log(x, eps=1e-12):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x)

def fit_sbm_em(A, K, max_iter=200, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    q = rng.dirichlet(np.ones(K), size=n)  # (n,K)
    pi = q.mean(axis=0)
    denom = q.T @ q
    num = q.T @ A @ q
    P = np.clip((num + num.T) / np.maximum(denom + denom.T, 1e-12),
                1e-6, 1 - 1e-6)

    ll_hist = []
    for it in range(max_iter):
        log_pi = _safe_log(pi)
        logP = _safe_log(P)
        log1mP = _safe_log(1 - P)

        # Precompute messages
        m_edge = np.stack([q @ logP[k] for k in range(K)], axis=1)    # (n,K)
        m_none = np.stack([q @ log1mP[k] for k in range(K)], axis=1)  # (n,K)

        s = A @ m_edge + (1 - A) @ m_none  # (n,K)

        q_new = s + log_pi  # broadcast (K,)
        q_new -= q_new.max(axis=1, keepdims=True)
        q_new = np.exp(q_new)
        q_new /= q_new.sum(axis=1, keepdims=True)
        q = q_new

        pi = q.mean(axis=0)
        denom = q.T @ q
        num = q.T @ A @ q
        P = np.clip((num + num.T) / np.maximum(denom + denom.T, 1e-12),
                    1e-6, 1 - 1e-6)

        # LL via expected Bernoulli with Q = q P q^T (upper triangle)
        Q = q @ P @ q.T
        iu = np.triu_indices(n, 1)
        ll = np.sum(A[iu] * _safe_log(Q[iu]) +
                    (1 - A[iu]) * _safe_log(1 - Q[iu]))
        ll_hist.append(ll)
        if it > 0 and abs(ll_hist[-1] - ll_hist[-2]) < tol * (1 + abs(ll_hist[-2])):
            break
    return q, pi, P, ll_hist

def bic_from_ll(ll, n, K):
    m = n * (n - 1) // 2
    d = (K - 1) + K * (K + 1) // 2
    return -2 * ll + d * math.log(m)

def icl_from_bic(bic, q):
    ent = -np.sum(q * np.log(np.clip(q, 1e-12, 1)), axis=1).sum()
    return bic + 2 * ent

def fit_many_K(A, Kmin=2, Kmax=12, seeds=(0, 1, 2), max_iter=200):
    best = None
    all_runs = []
    n = A.shape[0]
    for K in range(Kmin, Kmax + 1):
        bestK = None
        for s in seeds:
            q, pi, P, ll_hist = fit_sbm_em(A, K, seed=s, max_iter=max_iter)
            ll = ll_hist[-1]
            bic = bic_from_ll(ll, n, K)
            icl = icl_from_bic(bic, q)
            cand = dict(K=K, seed=s, ll=ll, bic=bic, icl=icl,
                        q=q, pi=pi, P=P, ll_hist=ll_hist)
            if (bestK is None) or (cand['icl'] < bestK['icl']):
                bestK = cand
        all_runs.append(bestK)
        if (best is None) or (bestK['icl'] < best['icl']):
            best = bestK
    return best, all_runs

# ---------------------------
# Plotting
# ---------------------------
def plot_block_matrix(P: np.ndarray, outpath: Path):
    plt.figure()
    plt.imshow(P, aspect='equal')
    plt.title("SBM Block Probabilities (P)")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_degree_hist(A: np.ndarray, outpath: Path):
    deg = A.sum(axis=1)
    plt.figure()
    plt.hist(deg, bins=30)
    plt.title("Graph degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    
    
# ---------------------------
# SBM Wiley-style plot
# ---------------------------

def plot_sbm_wiley(A, communities, labels, outpath: Path):
    """
    Recreates the Wiley block matrix SVG-style figure using:
    - Sorted adjacency matrix
    - Red dashed block boundaries
    - Community color sidebar
    - Black/white adjacency heatmap
    """

    # Sort nodes by community
    idx = np.argsort(communities)
    A_sorted = A[idx][:, idx]
    comm_sorted = communities[idx]
    labels_sorted = [labels[i] for i in idx]

    # Identify block boundaries
    unique = np.unique(comm_sorted)
    K = len(unique)
    block_sizes = [np.sum(comm_sorted == c) for c in unique]
    boundaries = np.cumsum(block_sizes)

    # Black/white cmap
    bw = ListedColormap(["white", "black"])

    plt.figure(figsize=(14, 14))
    ax = sns.heatmap(
        A_sorted,
        cmap=bw,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
        linewidth=0,
        linecolor=None
    )

    # Add dashed block boundaries
    for b in boundaries[:-1]:
        ax.axhline(b, color='red', linestyle='--', linewidth=0.7)
        ax.axvline(b, color='red', linestyle='--', linewidth=0.7)

    # Community color palette (7 colors like paper)
    sidebar_colors = {
        i: c for i, c in zip(unique, [
            "#ca2c92", "#0099e6", "#00cc44",
            "#33ccff", "#ffbf00", "#66cc66",
            "#cc3300"
        ])
    }

    # Sidebar showing community membership
    sidebar_ax = plt.gcf().add_axes([0.06, 0.1, 0.015, 0.8])
    sidebar_ax.imshow(
        comm_sorted.reshape(-1, 1),
        cmap=ListedColormap([sidebar_colors[c] for c in unique])
    )
    sidebar_ax.set_xticks([])
    sidebar_ax.set_yticks([])

    # Add legend
    for c in unique:
        plt.plot([], [], color=sidebar_colors[c],
                 label=f"Community {c}", linewidth=6)

    plt.legend(
        title="Communities",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=11
    )

    plt.suptitle("SBM Block Structure", fontsize=18, y=0.93)
    plt.tight_layout(rect=[0.08, 0.05, 0.95, 0.9])
    plt.savefig(outpath, dpi=220)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="End-to-end wastewater SBM pipeline from FASTA.")
    p.add_argument("--input_dir", required=True, help="Folder with FASTA files (one sample per file).")
    p.add_argument("--glob", default="*.fa,*.fasta,*.fna", help="Comma-separated FASTA patterns.")
    p.add_argument("--kraken_db", required=True, help="Kraken2 database path.")
    p.add_argument("--outdir", default="pipeline_out", help="Output folder.")

    p.add_argument("--rank", default="species", choices=list(RANK_CODE.keys()))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--confidence", type=float, default=0.1)

    p.add_argument("--use_bracken", action="store_true")
    p.add_argument("--bracken_db", default="", help="If using Bracken, provide its DB if different from Kraken DB.")
    p.add_argument("--keep_intermediate", action="store_true")

    # Filters + graph
    p.add_argument("--min_prevalence", type=float, default=0.1,
                   help="Fraction of samples where taxon must appear.")
    p.add_argument("--min_total", type=int, default=50,
                   help="Minimum total reads per taxon.")
    p.add_argument("--fdr_alpha", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=0.45,
                   help="Abs(rho) threshold for edges.")

    # SBM fitting
    p.add_argument("--Kmin", type=int, default=2)
    p.add_argument("--Kmax", type=int, default=12)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max_iter", type=int, default=200)

    # SBM simulation (new)
    p.add_argument(
        "--simulate_sbm",
        action="store_true",
        help="If set, replace empirical graph with a synthetic SBM(n,p,W) graph."
    )
    p.add_argument(
        "--sim_K",
        type=int,
        default=3,
        help="Number of communities for synthetic SBM when --simulate_sbm is used."
    )

    # Viral filter
    p.add_argument(
        "--viral_only",
        action="store_true",
        help="Retain only viral taxa (names matching virus/coronavirus/SARS/CoV/COVID). "
             "Use with --rank species; genus-level names rarely match."
    )

    args = p.parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    patterns = [x.strip() for x in args.glob.split(",")]
    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))

    # 1) FASTA -> taxa×samples (or samples×taxa after transpose below)
    print("[1/5] Classifying FASTA with Kraken2" + (" + Bracken" if args.use_bracken else ""))
    counts_taxa_as_rows = classify_folder_to_matrix(
        input_dir=input_dir,
        patterns=patterns,
        kraken_db=Path(args.kraken_db),
        outdir=outdir / "taxonomy",
        rank=args.rank,
        threads=args.threads,
        confidence=args.confidence,
        use_bracken=args.use_bracken,
        bracken_db=Path(args.bracken_db) if args.bracken_db else Path(args.kraken_db),
        keep_intermediate=args.keep_intermediate
    )
    counts_taxa_as_rows.to_csv(outdir / "matrix_counts_taxa_rows.tsv", sep="\t")

    # samples × taxa for convenience / sanity checks
    counts = counts_taxa_as_rows.T
    counts.to_csv(outdir / "matrix_counts_samples_rows.tsv", sep="\t")
    rel = (counts.T / counts.sum(axis=1).replace(0, 1)).T * 100.0
    rel.to_csv(outdir / "matrix_relative_abundance_samples_rows.tsv", sep="\t")

    def filter_viral_taxa(df: pd.DataFrame) -> pd.DataFrame:
        viral_mask = df.index.str.contains(
            "virus|coronavirus|SARS|CoV|COVID",
            case=False, regex=True
        )
        df_viral = df.loc[viral_mask].copy()
        if df_viral.empty:
            print("[WARN] No viral taxa found — check Kraken DB or rank setting.")
        else:
            print(f"[INFO] Retained {df_viral.shape[0]} viral taxa (of {df.shape[0]} total).")
        return df_viral

    if args.viral_only:
        counts_taxa_as_rows = filter_viral_taxa(counts_taxa_as_rows)

    # 2) Build graph from counts
    print("[2/5] Building association graph (CLR + Spearman + threshold)")
    A, taxa, rho, pval = build_graph_from_counts(
        counts_df=counts_taxa_as_rows,   # taxa × samples
        min_prevalence=args.min_prevalence,
        min_total=args.min_total,
        fdr_alpha=args.fdr_alpha,
        tau=args.tau
    )
    print(f"[INFO] Number of taxa (nodes) in graph: {len(taxa)}")

    # ------------------------------------------------------------
    # OPTIONAL: Replace empirical graph with synthetic SBM
    # ------------------------------------------------------------
    if args.simulate_sbm:
        print("[INFO] Simulating SBM graph instead of using empirical CLR graph.")
        n = len(taxa)
        K_sim = args.sim_K

        # Equal priors by default
        p_vec = np.full(K_sim, 1.0 / K_sim)

        # Simple block structure: high within, low between
        within = 0.8
        between = 0.1
        W_sim = np.full((K_sim, K_sim), between)
        np.fill_diagonal(W_sim, within)

        X_true, A_sbm = sample_sbm(n=n, p=p_vec, W=W_sim, seed=42)
        A = A_sbm

        # Save ground-truth SBM parameters for evaluation
        pd.Series(X_true, name="true_community").to_csv(
            outdir / "sbm_true_labels.csv", index=False
        )
        np.savetxt(outdir / "sbm_true_W.csv", W_sim, delimiter=",")
        print("[INFO] SBM(n,p,W) simulation complete. Graph replaced.")

    # Save graph + extra info
    np.savetxt(outdir / "graph_adjacency.csv", A, fmt="%d", delimiter=",")
    pd.Series(taxa).to_csv(outdir / "graph_taxa_index.tsv",
                           sep="\t", header=False, index=False)
    np.save(outdir / "rho.npy", rho)
    np.save(outdir / "pval.npy", pval)
    plot_degree_hist(A, outdir / "degree_hist.png")

    # 3) Fit SBM across K
    print("[3/5] Fitting SBM across K")
    best, all_runs = fit_many_K(
        A,
        Kmin=args.Kmin,
        Kmax=args.Kmax,
        seeds=seeds,
        max_iter=args.max_iter
    )
    Kbest = best["K"]
    q = best["q"]
    P = best["P"]
    pi = best["pi"]
    assign = q.argmax(axis=1)

    # 4) Save SBM outputs
    print("[4/5] Saving SBM outputs")
    pd.DataFrame(q, columns=[f"k{k}" for k in range(Kbest)]).to_csv(
        outdir / "sbm_q_soft_assignments.csv", index=False
    )
    pd.DataFrame({"taxon": taxa, "community": assign}).to_csv(
        outdir / "sbm_assignments.csv", index=False
    )
    pd.DataFrame(
        P,
        columns=[f"k{k}" for k in range(Kbest)],
        index=[f"k{k}" for k in range(Kbest)]
    ).to_csv(outdir / "sbm_block_P.csv")
    pd.Series(pi, index=[f"k{k}" for k in range(Kbest)]).to_csv(
        outdir / "sbm_pi.csv"
    )
    with (outdir / "sbm_bestK.txt").open("w") as f:
        f.write(f"Best K (ICL): {Kbest}\n")

    plot_block_matrix(P, outdir / "sbm_block_P.png")

    # NEW: Wiley-style block matrix plot
    communities = assign + 1
    print("[INFO] Creating Wiley-style SBM adjacency block figure...")
    plot_sbm_wiley(
        A=A,
        communities=communities,
        labels=taxa,
        outpath=outdir / "sbm_block_wiley_style.png"
    )
    print("[INFO] Saved sbm_block_wiley_style.png")


    # 5) Final notes
    print("[5/5] Done.")
    print(f"Best K (ICL): {Kbest}")
    print(f"Outputs in: {outdir.resolve()}")
    print("Next: Use sbm_assignments.csv to compare with VQ-VAE clusters or do biological enrichment.")

if __name__ == "__main__":
    main()
