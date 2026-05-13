# Probabilistic Graph-Based Sequence Reconstruction

Wastewater metagenomics pipeline for microbial co-occurrence network analysis using Stochastic Block Models (SBM).

## Overview

This project applies probabilistic community detection to wastewater metagenomic data. Raw paired-end FASTQ reads (9 samples, accessions SRR35556007–SRR35939740) are classified taxonomically with Kraken2, transformed via CLR normalization, and analyzed as a co-occurrence network. Stochastic Block Model (SBM) inference and spectral/UMAP embeddings identify latent microbial community structure.

## Pipeline

```
FASTQ → Trimmomatic (QC) → Kraken2 (taxonomic classification)
      → CLR-normalized abundance matrix
      → Spearman co-occurrence graph (|ρ| ≥ τ threshold)
      → SBM (variational EM, ICL model selection)
      → Spectral + UMAP embedding + GMM clustering
```

## Repository Structure

```
src/
  run_full_pipeline.py          # End-to-end pipeline: FASTA → SBM outputs
  auto_run_wastewater_pipeline.py  # FASTQ → FASTA conversion + pipeline launcher
  advanced_visualization_v2.py  # Spectral/UMAP/network visualizations
  archive/                      # Older script versions

notebooks/
  genus_network_analysis.ipynb  # Main analysis notebook (genus-level)
  preprocessing pipeline.ipynb  # QC and preprocessing steps
  visualizations_genus/         # Figures: genus-level network analysis
  visualizations_species/       # Figures: species-level network analysis

results/
  project_out/
    pipeline_results_genus/           # Baseline (217 taxa, τ=0.45, prev≥0.1)
    experiments/
      pipeline_results_genus_fixed/   # Strict filtering (39 taxa)
      pipeline_results_genus_loose1/  # Loose filtering (334 taxa)
    ggm_genus/                        # Spectral-GMM outputs (genus)
    ggm/                              # Spectral-GMM outputs (species)
    pipeline_results_viral_species/   # Viral-only species subset

qc/
  fastqc_before/   # FastQC reports pre-trimming
  fastqc_after/    # FastQC reports post-trimming
```

> **Data not tracked in git:** raw FASTQs, cleaned FASTQs, FASTA intermediates, Kraken2 reports, and the Kraken2 database (`kraken_db_standard/`) are excluded via `.gitignore`.

## Results Summary

All runs performed at genus level with 9 wastewater samples.

| Run | Nodes | Edges | Density | Best K (ICL) |
|-----|-------|-------|---------|--------------|
| `pipeline_results_genus` (baseline) | 217 | 6,080 | 0.26 | 2 |
| `pipeline_results_genus_fixed` (strict) | 39 | 199 | 0.27 | 2 |
| `pipeline_results_genus_loose1` (loose) | 334 | 19,787 | 0.36 | 2 |

The three runs serve as a parameter sensitivity analysis: `_fixed` uses stricter prevalence/count filters, `_loose1` uses looser thresholds, and the baseline is the primary result.

**Community detection agreement (ARI, Louvain vs GMM):**
- Genus level: 0.43
- Species level: 0.79

The higher species-level ARI suggests finer taxonomic resolution yields more coherent community structure.

**Note on block matrix:** All three SBM runs return near-degenerate block probability matrices (P ≈ 1 for all blocks), indicating that within the filtered network the two inferred communities have near-identical connectivity profiles. This likely reflects the high density of the co-occurrence graph at these thresholds — tightening `--tau` or applying BH-FDR correction (see commented code in `build_graph_from_counts`) would reduce density and may reveal sharper block structure.

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

External tools also required: [Kraken2](https://ccb.jhu.edu/software/kraken2/), [Trimmomatic](http://www.usadellab.org/cms/?page=trimmomatic), [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/), [seqtk](https://github.com/lh3/seqtk).

### Run the full pipeline

```bash
python src/run_full_pipeline.py \
    --input_dir path/to/fastas \
    --kraken_db path/to/kraken_db \
    --outdir results/project_out/my_run \
    --rank genus \
    --threads 16 \
    --confidence 0.1 \
    --min_prevalence 0.2 \
    --min_total 20 \
    --tau 0.6 \
    --Kmin 2 --Kmax 12
```

Add `--viral_only` to restrict the analysis to viral taxa (recommended with `--rank species`).

### Run from raw FASTQ

```bash
python src/auto_run_wastewater_pipeline.py \
    --input_dir path/to/fastqs \
    --kraken_db path/to/kraken_db \
    --outdir results/project_out/my_run
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rank` | `species` | Taxonomic rank (genus/species/family/…) |
| `--tau` | `0.45` | Spearman |ρ| threshold for graph edges |
| `--min_prevalence` | `0.1` | Min fraction of samples a taxon must appear in |
| `--min_total` | `50` | Min total reads across all samples |
| `--Kmin/--Kmax` | `2/12` | Community count search range (ICL selects best K) |
| `--viral_only` | off | Filter to viral taxa before graph construction |
| `--simulate_sbm` | off | Replace empirical graph with synthetic SBM for validation |

## Citation

If you use this pipeline, please cite the relevant tools:
- Wood & Salzberg (2014) — Kraken2
- Bolger et al. (2014) — Trimmomatic
- Andrews (2010) — FastQC
