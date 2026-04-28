#!/usr/bin/env python3
"""
Read bench/results/results.csv (median across trials) and produce four
PDF figures into paper/figures/:

  fig_strong_scaling.pdf   speedup vs ranks, with ideal line
  fig_weak_scaling.pdf     parallel efficiency vs ranks
  fig_cpu_vs_gpu.pdf       CPU vs hybrid GPU bar chart per rank count
  fig_breakdown.pdf        stacked time breakdown for one config

If the CSV is the synthetic placeholder, every figure is overlaid with a
PLACEHOLDER watermark so we can't accidentally ship draft data as real.
"""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_csv(path):
    rows = []
    with Path(path).open() as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ("ranks","dims_x","dims_y","global_w","global_h","steps","trial"):
                row[k] = int(row[k])
            for k in ("t_total","t_halo","t_kernel","t_io"):
                row[k] = float(row[k])
            row["synthetic"] = bool(int(row.get("synthetic","0") or 0))
            rows.append(row)
    return rows

def median_by(rows, key, value="t_total"):
    g = defaultdict(list)
    for r in rows: g[r[key]].append(r[value])
    return {k: median(v) for k, v in g.items()}

def watermark(ax, synthetic):
    if not synthetic: return
    ax.text(0.5, 0.5, "PLACEHOLDER", transform=ax.transAxes,
            ha="center", va="center", fontsize=44, alpha=0.18,
            rotation=22, color="red", fontweight="bold")

def fig_strong(rows, outdir, synthetic):
    sub = [r for r in rows if r["label"].startswith("strong_")]
    times = median_by(sub, "ranks")
    ranks = sorted(times)
    if not ranks: return
    base = times[ranks[0]]
    speedup = [base / times[r] for r in ranks]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(ranks, speedup, "o-", label="measured")
    ax.plot(ranks, ranks, "--", color="gray", label="ideal")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("MPI ranks (= GPUs)")
    ax.set_ylabel("Speedup vs single rank")
    ax.set_title("Strong scaling, $32768^2$, 1000 steps")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    watermark(ax, synthetic)
    fig.tight_layout(); fig.savefig(outdir / "fig_strong_scaling.pdf"); plt.close(fig)

def fig_weak(rows, outdir, synthetic):
    sub = [r for r in rows if r["label"].startswith("weak_")]
    times = median_by(sub, "ranks")
    ranks = sorted(times)
    if not ranks: return
    base = times[ranks[0]]
    eff = [100.0 * base / times[r] for r in ranks]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(ranks, eff, "s-")
    ax.axhline(100, ls="--", color="gray", label="ideal 100%")
    ax.set_xscale("log", base=2); ax.set_xlabel("MPI ranks (= GPUs)")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title("Weak scaling, $\\sim 8192^2$ per rank")
    ax.set_ylim(0, 110)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    watermark(ax, synthetic)
    fig.tight_layout(); fig.savefig(outdir / "fig_weak_scaling.pdf"); plt.close(fig)

def fig_cpu_vs_gpu(rows, outdir, synthetic):
    sub_cpu = [r for r in rows if r["label"].startswith("cpu_")]
    sub_gpu = [r for r in rows if r["label"].startswith("gpu_") and r["mode"]=="gpu"]
    cpu_t = median_by(sub_cpu, "ranks")
    gpu_t = median_by(sub_gpu, "ranks")
    ranks = sorted(set(cpu_t) & set(gpu_t))
    if not ranks: return
    import numpy as np
    x = np.arange(len(ranks)); w = 0.4
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar(x - w/2, [cpu_t[r] for r in ranks], w, label="CPU (MPI only)")
    ax.bar(x + w/2, [gpu_t[r] for r in ranks], w, label="GPU (MPI + CUDA)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("Total time (s)")
    ax.set_title("CPU-only vs hybrid GPU, $8192^2$, 500 steps")
    ax.legend(); ax.grid(alpha=0.3, axis="y", which="both")
    watermark(ax, synthetic)
    fig.tight_layout(); fig.savefig(outdir / "fig_cpu_vs_gpu.pdf"); plt.close(fig)

def fig_breakdown(rows, outdir, synthetic):
    sub = [r for r in rows if r["label"].startswith("breakdown_")]
    if not sub: return
    kernels = ["naive", "shared", "overlap"]
    by = {k: [r for r in sub if r["kernel"] == k] for k in kernels}
    halo   = [median([r["t_halo"]   for r in by[k]]) if by[k] else 0 for k in kernels]
    kern   = [median([r["t_kernel"] for r in by[k]]) if by[k] else 0 for k in kernels]
    import numpy as np
    x = np.arange(len(kernels))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar(x, kern,                     label="kernel")
    ax.bar(x, halo, bottom=kern,        label="halo exchange")
    ax.set_xticks(x); ax.set_xticklabels(kernels)
    ax.set_ylabel("Time per 1000 steps (s)")
    ax.set_title("Per-phase breakdown, 16 ranks, $32768^2$")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    watermark(ax, synthetic)
    fig.tight_layout(); fig.savefig(outdir / "fig_breakdown.pdf"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bench/results/results.csv")
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args()

    rows = read_csv(args.csv)
    synthetic = any(r["synthetic"] for r in rows)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    fig_strong(rows, outdir, synthetic)
    fig_weak(rows, outdir, synthetic)
    fig_cpu_vs_gpu(rows, outdir, synthetic)
    fig_breakdown(rows, outdir, synthetic)
    print(f"wrote 4 PDFs to {outdir} (synthetic={synthetic})")

if __name__ == "__main__":
    main()
