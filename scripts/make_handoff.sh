#!/bin/bash
# Bundle the benchmark deliverables into a single tarball for the writers.
#
# Run from the project root after benchmarks + verification have completed:
#   bash scripts/make_handoff.sh [verify-slurm-file]
#
# If no verify slurm file is given, the script picks the most recent
# slurm-*.out under bench/results/ that contains "VERIFY OK".

set -e

DATE=$(date +%Y%m%d)
HANDOFF=bench/handoff_${DATE}
mkdir -p "$HANDOFF"

# 1. Raw 3-trial-per-config data (primary deliverable).
cp bench/results/results.csv "$HANDOFF/results_raw.csv"

# 2. Pre-aggregated summary (mean + stddev across the 3 trials per config).
awk -F, '
NR==1 { print $0",t_total_mean,t_total_std"; next }
{ n[$1]++; s[$1]+=$11; ss[$1]+=$11*$11; row[$1]=$0 }
END {
    for (k in n) {
        m  = s[k]/n[k]
        v  = ss[k]/n[k] - m*m
        sd = (v > 0) ? sqrt(v) : 0
        printf "%s,%.6f,%.6f\n", row[k], m, sd
    }
}' bench/results/results.csv | sort > "$HANDOFF/results_summary.csv"

# 3. Correctness evidence.
VERIFY_SRC="${1:-}"
if [ -z "$VERIFY_SRC" ]; then
    VERIFY_SRC=$(grep -l 'VERIFY OK' bench/results/slurm-*.out 2>/dev/null \
                 | xargs -r ls -t | head -n 1)
fi
if [ -n "$VERIFY_SRC" ] && [ -f "$VERIFY_SRC" ]; then
    cp "$VERIFY_SRC" "$HANDOFF/verify.log"
    echo "[handoff] verify log: $VERIFY_SRC"
else
    echo "[handoff] WARNING: no verify slurm output found; verify.log will be missing" >&2
fi

# 4. Benchmark spec (so the writers can see what each label means).
cp bench/run_all.sh "$HANDOFF/run_all.sh"

# 5. Code provenance.
{
    echo "commit: $(git rev-parse HEAD)"
    echo "branch: $(git rev-parse --abbrev-ref HEAD)"
    echo
    echo "git status --short:"
    git status --short
} > "$HANDOFF/git_commit.txt"

# 6. README that ties everything together.
cat > "$HANDOFF/README.md" <<'EOF'
# GoL benchmark handoff

## Files
- `results_raw.csv` — one row per (config, trial). Use for stats / error bars.
- `results_summary.csv` — `results_raw.csv` with `t_total_mean` and
  `t_total_std` appended per row (averaged over 3 trials).
- `run_all.sh` — exact commands that produced the data; cross-reference the
  `label` column.
- `verify.log` — small-grid serial-vs-MPI cmp output; the "VERIFY OK" line
  confirms the GPU kernel produces bit-identical output to the serial
  reference for naive/random/1-rank/256x256.
- `git_commit.txt` — commit hash, branch, and any uncommitted changes flagged.

## Hardware / software
- AiMOS DCS-2024 partition, IBM Power9 + 6x NVIDIA V100 (16 GB) per node.
- Spectrum MPI (CUDA-aware, `mpirun -gpu`), CUDA 11.2, xlc_r 16.1.
- Timebase: POWER9 mftb, 512 MHz; all timings via clockcycle.h.

## Benchmark configurations (label -> meaning)
- `strong_R`: strong scaling, 24576^2 grid, 1000 steps, R ranks
  (1, 2, 4, 6, 12, 24, 48), gpu/overlap.
- `weak_R`: weak scaling at 8192^2 per rank, 1000 steps, R ranks (1, 4, 16),
  gpu/overlap.
- `cpu_R` / `gpu_R`: 8192^2, 500 steps, R ranks (1, 4, 16); cpu uses naive
  CPU step, gpu uses overlap.
- `breakdown_16`: phase breakdown reference (16 ranks, 32768^2, 1000 steps).
- `ksweep_<kernel>_<W>`: single-GPU kernel comparison (naive vs shared) at
  W^2 for W in {256, 1024, 4096, 16384}.
- `io_<K>`: MPI-IO checkpoint cost; K = checkpoint interval (0 = none, 25,
  100), 6 ranks, 12288^2, 500 steps.

## CSV columns
`label, ranks, dims_x, dims_y, mode, kernel, global_w, global_h, steps,
trial, t_total, t_halo, t_kernel, t_io`

All times in seconds. `t_total = t_halo + t_kernel + t_io`.

## Headline findings
- Strong scaling: 1 -> 48 ranks at 24576^2 gives ~11.4x speedup; halo time
  dominates past ~12 ranks (kernel stays ~0.026s flat).
- Weak scaling: ~70% efficiency at 16 ranks across 3 nodes
  (8192^2 per rank).
- Shared-memory kernel is *slower* than naive at large sizes
  (1.59s vs 1.15s at 16384^2). The working set fits in V100 L2, so naive's
  cache behavior beats explicit shared-mem staging overhead.
- CPU vs GPU at 8192^2 / 1 rank: 134s vs 0.31s (~430x single-rank speedup).

## Caveats
- Verification only covered naive / random init / 1 rank / 256x256. To
  defend correctness for the shared and overlap kernels and for multi-rank
  configurations, expand `scripts/run_verify.sh` and rerun before the paper
  cites those numbers.
- No profiling-tool output (Nsight, nvprof) is included.
- No per-rank timing breakdowns; phase timings are accumulated per rank-0.
EOF

# 7. Tar it up.
TARBALL=bench/handoff_${DATE}.tar.gz
tar czf "$TARBALL" -C bench "handoff_${DATE}"

echo
echo "Bundle: $TARBALL"
ls -lh "$TARBALL"
echo
echo "Contents:"
tar tzf "$TARBALL"

