# Results

Large result files are not tracked in git (see .gitignore).

## Directory Layout

```
project_out/
├── pipeline_results_genus/         ← CANONICAL results (genus-level, final parameters)
├── pipeline_results_viral_species/ ← Viral species-level results
├── ggm/                            ← Gaussian Graphical Model outputs
├── ggm_genus/                      ← GGM at genus level
└── experiments/                    ← Parameter sensitivity runs (not for publication)
    ├── pipeline_results_genus_fixed/
    └── pipeline_results_genus_loose1/
```

## Key Output Files (pipeline_results_genus/)

| File | Description |
|------|-------------|
| `graph_adjacency.csv` | Co-occurrence network adjacency matrix |
| `rho.npy` / `pval.npy` | Correlation matrix and p-values |
| `sbm_assignments.csv` | SBM community assignments per taxon |
| `sbm_bestK.txt` | Selected number of communities |
| `matrix_relative_abundance_samples_rows.tsv` | Relative abundance table |
| `sbm_block_wiley_style.png` | Publication-quality block interaction figure |
