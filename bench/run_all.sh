#!/bin/bash
# Master harness: submits all benchmark configurations described in the
# project plan. Each row in the matrices below becomes one sbatch job;
# each job runs 3 trials internally and appends to bench/results/results.csv.
#
# Run from the project root after `make all`:
#   ./bench/run_all.sh
#
# Multi-node jobs can sit in the queue for hours; submit early.

set -e
SCRIPTS=scripts

mkdir -p bench/results

# ---- strong scaling ---------------------------------------------------------
# Same global grid (32768 x 32768), increasing rank count.
for R in 1 2 4 6; do
    sbatch --export=ALL $SCRIPTS/run_single_node.sh $R 32768 32768 1000 gpu overlap strong_${R}
done
for R in 12 24 48; do
    NODES=$(( (R + 5) / 6 ))
    sbatch --export=ALL --nodes=$NODES $SCRIPTS/run_multi_node.sh $R 32768 32768 1000 gpu overlap strong_${R}
done

# ---- weak scaling -----------------------------------------------------------
# Per-rank work fixed at ~8192^2; global grid grows with sqrt(ranks).
sbatch --export=ALL $SCRIPTS/run_single_node.sh  1  8192  8192 1000 gpu overlap weak_1
sbatch --export=ALL $SCRIPTS/run_single_node.sh  4 16384 16384 1000 gpu overlap weak_4
sbatch --export=ALL --nodes=3 $SCRIPTS/run_multi_node.sh 16 32768 32768 1000 gpu overlap weak_16

# ---- CPU vs hybrid (same grid, both modes) ----------------------------------
for R in 1 4; do
    sbatch --export=ALL $SCRIPTS/run_single_node.sh $R 8192 8192 500 cpu naive   cpu_${R}
    sbatch --export=ALL $SCRIPTS/run_single_node.sh $R 8192 8192 500 gpu overlap gpu_${R}
done
sbatch --export=ALL --nodes=3 $SCRIPTS/run_multi_node.sh 16 8192 8192 500 cpu naive   cpu_16
sbatch --export=ALL --nodes=3 $SCRIPTS/run_multi_node.sh 16 8192 8192 500 gpu overlap gpu_16

# ---- per-phase breakdown (16 ranks, 32K grid) -------------------------------
sbatch --export=ALL --nodes=3 $SCRIPTS/run_multi_node.sh 16 32768 32768 1000 gpu overlap breakdown_16

# ---- single-GPU kernel sweep ------------------------------------------------
for W in 256 1024 4096 16384; do
    sbatch --export=ALL $SCRIPTS/run_single_node.sh 1 $W $W 500 gpu naive  ksweep_naive_${W}
    sbatch --export=ALL $SCRIPTS/run_single_node.sh 1 $W $W 500 gpu shared ksweep_shared_${W}
done

# ---- MPI-IO checkpoint cost -------------------------------------------------
# Same job, varying checkpoint frequency.
for K in 0 100 25; do
    sbatch --export=ALL --job-name=gol_io_${K} \
        $SCRIPTS/run_single_node.sh 6 8192 8192 500 gpu overlap io_${K}
done

echo "All jobs submitted. Watch with: squeue -u $USER"
