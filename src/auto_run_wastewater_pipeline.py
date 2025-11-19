#!/usr/bin/env python3
"""
Automated Wastewater Pipeline Runner
------------------------------------
Steps:
1. Find all FASTQ files in input_dir.
2. Convert each to FASTA.
3. Run Kraken2 (and Bracken if enabled).
4. Launch the full probabilistic pipeline on the generated FASTA files.
"""

import argparse, subprocess, shutil
from pathlib import Path
from tqdm import tqdm

def run_cmd(cmd, msg=None):
    if msg: print(f"\n🧩 {msg}")
    subprocess.run(cmd, check=True)

def have_bin(name):
    return shutil.which(name) is not None

def convert_fastq_to_fasta(input_dir, out_dir):
    """Convert all .fastq/.fq to .fasta using seqtk (fast)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fastqs = sorted(list(input_dir.glob("*.fastq")) + list(input_dir.glob("*.fq")))
    if not fastqs:
        raise FileNotFoundError("No FASTQ files found.")
    for fq in tqdm(fastqs, desc="Converting FASTQ → FASTA"):
        fa = out_dir / (fq.stem + ".fasta")
        run_cmd(["seqtk", "seq", "-a", str(fq)], msg=f"Converting {fq.name}")
        with open(fa, "w") as fout:
            subprocess.run(["seqtk", "seq", "-a", str(fq)], stdout=fout)
    return out_dir

def classify_fastas(fasta_dir, kraken_db, threads=8, use_bracken=False, bracken_db=None):
    """Run Kraken2 (+ optional Bracken) on all FASTA files"""
    fastas = sorted(list(fasta_dir.glob("*.fasta")))
    if not fastas:
        raise FileNotFoundError("No FASTA files found.")
    out_reports = fasta_dir / "reports"
    out_reports.mkdir(exist_ok=True)

    for fa in tqdm(fastas, desc="Classifying with Kraken2"):
        sample = fa.stem
        report = out_reports / f"{sample}.report"
        out = out_reports / f"{sample}.kraken.out"
        cmd = [
            "kraken2",
            "--db", str(kraken_db),
            "--threads", str(threads),
            "--report", str(report),
            "--output", str(out),
            str(fa)
        ]
        run_cmd(cmd, msg=f"Running Kraken2 on {sample}")

        if use_bracken:
            if not have_bin("bracken"):
                print("[WARN] Bracken not installed — skipping refinement.")
                continue
            level = "S"  # species-level
            br_out = out_reports / f"{sample}.bracken"
            cmd_b = [
                "bracken",
                "-d", str(bracken_db if bracken_db else kraken_db),
                "-i", str(report),
                "-o", str(br_out),
                "-r", "100",
                "-l", level
            ]
            run_cmd(cmd_b, msg=f"Running Bracken on {sample}")

def main():
    ap = argparse.ArgumentParser(description="Automate FASTQ → FASTA → Kraken2 → full pipeline")
    ap.add_argument("--input_dir", required=True, help="Directory containing FASTQ files")
    ap.add_argument("--kraken_db", required=True, help="Path to Kraken2 database")
    ap.add_argument("--outdir", default="auto_pipeline_out", help="Output directory")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--use_bracken", action="store_true")
    ap.add_argument("--bracken_db", default="", help="Path to Bracken DB (if using Bracken)")
    ap.add_argument("--pipeline_script", default="run_full_pipeline.py", help="Path to your main pipeline script")
    ap.add_argument("--rank", default="species", help="Taxonomic rank for analysis (species/genus)")
    ap.add_argument("--confidence", type=float, default=0.1)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    fasta_dir = outdir / "fastas"
    outdir.mkdir(exist_ok=True)

    # Check dependencies
    for tool in ["seqtk", "kraken2"]:
        if not have_bin(tool):
            raise EnvironmentError(f"{tool} not found in PATH. Please install or load the module.")

    # 1. Convert FASTQ → FASTA
    convert_fastq_to_fasta(input_dir, fasta_dir)

    # 2. Run Kraken2 (+ Bracken if set)
    classify_fastas(fasta_dir, Path(args.kraken_db), threads=args.threads,
                    use_bracken=args.use_bracken,
                    bracken_db=Path(args.bracken_db) if args.bracken_db else None)

    # 3. Run full viral-filtered pipeline
    run_cmd([
        "python", args.pipeline_script,
        "--input_dir", str(fasta_dir),
        "--kraken_db", str(args.kraken_db),
        "--outdir", str(outdir / "pipeline_results"),
        "--rank", args.rank,
        "--threads", str(args.threads),
        "--confidence", str(args.confidence),
        "--use_bracken" if args.use_bracken else ""
    ], msg="Launching full probabilistic pipeline")

if __name__ == "__main__":
    main()
