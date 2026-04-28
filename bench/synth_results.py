#!/usr/bin/env python3
"""
Generate plausible-but-synthetic benchmark CSVs so the paper can be
laid out end-to-end before real AiMOS data lands. Every row produced
here is tagged ``synthetic=1`` so it's obvious if it ever leaks into
the real results file.

Model:
    kernel time    ~ alpha_k * (lw * lh) / throughput_per_rank,
                     with a small shared-memory speedup factor.
    halo time      ~ alpha_h * (perimeter) / bandwidth + latency,
                     scaled up across nodes for inter-node hops.
    overlap        ~ max(kernel, halo) + epsilon.
    cpu            ~ 60x slower kernel, no GPU.
    io time        ~ checkpoint_bytes / IO_bandwidth_per_rank.

Numbers are loosely calibrated against published V100 stencil throughput
(low hundreds of GCells/s on dense single-byte stencils) and Spectrum
MPI inter-node bandwidth (~20 GB/s per link). They are PLACEHOLDERS.
"""
from __future__ import annotations
import argparse, csv, math, random
from pathlib import Path

GCELLS_S_GPU       = 220.0   # naive V100, 1 GPU, ~32K^2
SHARED_SPEEDUP     = 1.18
CPU_SLOWDOWN       = 60.0
HALO_BW_INTRA_GBPS = 60.0    # NVLink-ish intra-node
HALO_BW_INTER_GBPS = 20.0    # IB inter-node
HALO_LATENCY_S     = 5e-6
IO_BW_PER_RANK_GBPS= 0.8

NOISE = 0.04

def kernel_time(lw, lh, mode, kernel):
    cells = lw * lh
    if mode == "cpu":
        return cells / (GCELLS_S_GPU / CPU_SLOWDOWN * 1e9)
    base = cells / (GCELLS_S_GPU * 1e9)
    if kernel == "shared":
        base /= SHARED_SPEEDUP
    return base

def halo_time(lw, lh, ranks, dims_x, dims_y, nodes):
    perim_bytes = 2 * (lw + lh)  # 1-byte cells, 4 sides, 2 neighbors per axis
    msgs = 4
    intra = nodes == 1
    bw = (HALO_BW_INTRA_GBPS if intra else HALO_BW_INTER_GBPS) * 1e9
    return msgs * (HALO_LATENCY_S + perim_bytes / bw)

def io_time(global_w, global_h, ranks, every, steps):
    if every <= 0: return 0.0
    n_writes = steps // every
    bytes_per_rank = (global_w * global_h) // ranks
    bw = IO_BW_PER_RANK_GBPS * 1e9
    return n_writes * (bytes_per_rank / bw + 0.01)

def jitter(x):
    return x * (1.0 + random.uniform(-NOISE, NOISE))

def emit_row(w, csvw, *, label, ranks, dims_x, dims_y, mode, kernel,
             gw, gh, steps, trial, t_kernel, t_halo, t_io, overlap):
    if overlap:
        t_total = max(t_kernel, t_halo) + 1e-4 + t_io
    else:
        t_total = t_kernel + t_halo + t_io
    csvw.writerow([
        label, ranks, dims_x, dims_y, mode, kernel,
        gw, gh, steps, trial,
        f"{jitter(t_total):.6f}",
        f"{jitter(t_halo):.6f}",
        f"{jitter(t_kernel):.6f}",
        f"{jitter(t_io):.6f}",
        1,
    ])

def dims_for(p):
    """Best 2-factor split, dims_x * dims_y == p, near square."""
    best = (1, p)
    for a in range(1, int(math.sqrt(p)) + 1):
        if p % a == 0: best = (a, p // a)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/results.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label","ranks","dims_x","dims_y","mode","kernel",
                    "global_w","global_h","steps","trial",
                    "t_total","t_halo","t_kernel","t_io","synthetic"])

        # Strong scaling: 32768^2, GPU overlap
        for R in [1, 2, 4, 6, 12, 24, 48]:
            dx, dy = dims_for(R)
            nodes = max(1, (R + 5) // 6)
            lw, lh = 32768 // dx, 32768 // dy
            for tr in (1, 2, 3):
                tk = kernel_time(lw, lh, "gpu", "shared")
                th = halo_time(lw, lh, R, dx, dy, nodes)
                emit_row(0, w, label=f"strong_{R}", ranks=R, dims_x=dx, dims_y=dy,
                         mode="gpu", kernel="overlap", gw=32768, gh=32768,
                         steps=1000, trial=tr, t_kernel=tk*1000, t_halo=th*1000,
                         t_io=0.0, overlap=True)

        # Weak scaling: per-rank ~8192^2, GPU overlap
        for R, GW in [(1, 8192), (4, 16384), (16, 32768)]:
            dx, dy = dims_for(R)
            nodes = max(1, (R + 5) // 6)
            lw, lh = GW // dx, GW // dy
            for tr in (1, 2, 3):
                tk = kernel_time(lw, lh, "gpu", "shared")
                th = halo_time(lw, lh, R, dx, dy, nodes)
                emit_row(0, w, label=f"weak_{R}", ranks=R, dims_x=dx, dims_y=dy,
                         mode="gpu", kernel="overlap", gw=GW, gh=GW,
                         steps=1000, trial=tr, t_kernel=tk*1000, t_halo=th*1000,
                         t_io=0.0, overlap=True)

        # CPU vs GPU at 8192^2
        for R in (1, 4, 16):
            dx, dy = dims_for(R)
            nodes = max(1, (R + 5) // 6)
            lw, lh = 8192 // dx, 8192 // dy
            for mode, kernel, lab in [("cpu","naive","cpu"),("gpu","overlap","gpu")]:
                for tr in (1, 2, 3):
                    tk = kernel_time(lw, lh, mode, kernel if mode=="gpu" else "naive")
                    th = halo_time(lw, lh, R, dx, dy, nodes)
                    emit_row(0, w, label=f"{lab}_{R}", ranks=R, dims_x=dx, dims_y=dy,
                             mode=mode, kernel=kernel, gw=8192, gh=8192,
                             steps=500, trial=tr, t_kernel=tk*500, t_halo=th*500,
                             t_io=0.0, overlap=(mode=="gpu"))

        # Per-phase breakdown (single config, both naive and shared and overlap)
        R = 16; dx, dy = dims_for(R); lw, lh = 32768//dx, 32768//dy
        for kernel, ovr in [("naive", False), ("shared", False), ("overlap", True)]:
            for tr in (1, 2, 3):
                tk = kernel_time(lw, lh, "gpu", kernel if kernel != "overlap" else "shared")
                th = halo_time(lw, lh, R, dx, dy, 3)
                emit_row(0, w, label=f"breakdown_{kernel}", ranks=R, dims_x=dx, dims_y=dy,
                         mode="gpu", kernel=kernel, gw=32768, gh=32768,
                         steps=1000, trial=tr, t_kernel=tk*1000, t_halo=th*1000,
                         t_io=0.0, overlap=ovr)

        # Single-GPU kernel sweep (naive vs shared)
        for W in (256, 1024, 4096, 16384):
            for kernel in ("naive", "shared"):
                for tr in (1, 2, 3):
                    tk = kernel_time(W, W, "gpu", kernel)
                    emit_row(0, w, label=f"ksweep_{kernel}_{W}",
                             ranks=1, dims_x=1, dims_y=1,
                             mode="gpu", kernel=kernel, gw=W, gh=W,
                             steps=500, trial=tr, t_kernel=tk*500, t_halo=0.0,
                             t_io=0.0, overlap=False)

        # MPI-IO checkpoint cost
        R = 6; dx, dy = dims_for(R); lw, lh = 8192//dx, 8192//dy
        for every in (0, 100, 25):
            for tr in (1, 2, 3):
                tk = kernel_time(lw, lh, "gpu", "shared")
                th = halo_time(lw, lh, R, dx, dy, 1)
                tio = io_time(8192, 8192, R, every, 500)
                emit_row(0, w, label=f"io_{every}", ranks=R, dims_x=dx, dims_y=dy,
                         mode="gpu", kernel="overlap", gw=8192, gh=8192,
                         steps=500, trial=tr, t_kernel=tk*500, t_halo=th*500,
                         t_io=tio, overlap=True)

    print(f"wrote {out}")

if __name__ == "__main__":
    main()
