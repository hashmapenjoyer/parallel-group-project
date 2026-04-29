#!/bin/bash
# Master harness: submits all benchmark configurations described in the
# project plan. Each row in the matrices below becomes one sbatch job;
# each job runs 3 trials internally and appends to bench/results/results.csv.
#
# Run from the project root after `make all`:
#   ./bench/run_all.sh
#
# Multi-node jobs can sit in the queue for hours; submit early.
#
# Resilience: each sbatch is wrapped in a retry loop. If the scheduler
# times out (socket timeout, prolog hang, transient queue issue), we
# retry with exponential backoff up to MAX_RETRIES times. A submit
# log is written to bench/results/submit.log so re-runs can resume
# without double-submitting.

SCRIPTS=scripts
LOGFILE=bench/results/submit.log
MAX_RETRIES=8
RETRY_BASE=5     # seconds; doubled each attempt up to a cap

mkdir -p bench/results
touch "$LOGFILE"

# Returns 0 on success, prints job ID on stdout. Retries on failure.
# Uses a unique key (the full label string) to skip already-submitted
# jobs on a re-run.
submit() {
    local key="$1"; shift
    if grep -qF "KEY=$key " "$LOGFILE"; then
        echo "[skip] $key already submitted" >&2
        return 0
    fi
    local attempt=0 wait=$RETRY_BASE
    while (( attempt < MAX_RETRIES )); do
        local out
        if out=$(sbatch "$@" 2>&1); then
            local jid=$(echo "$out" | awk '/Submitted batch job/ {print $4}')
            echo "KEY=$key JOB=$jid" >> "$LOGFILE"
            echo "[ok]   $key -> job $jid" >&2
            return 0
        fi
        echo "[warn] $key failed (attempt $((attempt+1))/$MAX_RETRIES): $out" >&2
        sleep $wait
        wait=$(( wait * 2 < 120 ? wait * 2 : 120 ))
        attempt=$((attempt + 1))
    done
    echo "[FAIL] $key gave up after $MAX_RETRIES attempts" >&2
    return 1
}

# Grid sizes are chosen so that every rank count in the sweeps below has a
# 2D factorization (a, b) with a|H and b|W. Powers of 2 don't work here
# because rank counts 6/12/24/48 carry a factor of 3, so we use
# 24576 = 24*1024 and 12288 = 6*2048 -- both divisible by 1,2,3,4,6,8,12,24,48.

# ---- strong scaling ---------------------------------------------------------
for R in 1 2 4 6; do
    submit "strong_${R}" --export=ALL \
        $SCRIPTS/run_single_node.sh $R 24576 24576 1000 gpu overlap strong_${R}
done
for R in 12 24 48; do
    NODES=$(( (R + 5) / 6 ))
    submit "strong_${R}" --export=ALL --nodes=$NODES --ntasks=$R \
        $SCRIPTS/run_multi_node.sh $R 24576 24576 1000 gpu overlap strong_${R}
done

# ---- weak scaling -----------------------------------------------------------
# Power-of-two rank counts decompose cleanly across power-of-two grids.
submit "weak_1"  --export=ALL $SCRIPTS/run_single_node.sh  1  8192  8192 1000 gpu overlap weak_1
submit "weak_4"  --export=ALL $SCRIPTS/run_single_node.sh  4 16384 16384 1000 gpu overlap weak_4
submit "weak_16" --export=ALL --nodes=3 --ntasks=16 \
    $SCRIPTS/run_multi_node.sh 16 32768 32768 1000 gpu overlap weak_16

# ---- CPU vs hybrid ----------------------------------------------------------
for R in 1 4; do
    submit "cpu_${R}" --export=ALL $SCRIPTS/run_single_node.sh $R 8192 8192 500 cpu naive   cpu_${R}
    submit "gpu_${R}" --export=ALL $SCRIPTS/run_single_node.sh $R 8192 8192 500 gpu overlap gpu_${R}
done
submit "cpu_16" --export=ALL --nodes=3 --ntasks=16 \
    $SCRIPTS/run_multi_node.sh 16 8192 8192 500 cpu naive   cpu_16
submit "gpu_16" --export=ALL --nodes=3 --ntasks=16 \
    $SCRIPTS/run_multi_node.sh 16 8192 8192 500 gpu overlap gpu_16

# ---- per-phase breakdown (16 ranks, large grid) -----------------------------
submit "breakdown_16" --export=ALL --nodes=3 --ntasks=16 \
    $SCRIPTS/run_multi_node.sh 16 32768 32768 1000 gpu overlap breakdown_16

# ---- single-GPU kernel sweep ------------------------------------------------
# Single-rank, so the 1x1 decomposition is valid for any grid size; powers of
# 2 keep the kernel sweep clean.
for W in 256 1024 4096 16384; do
    submit "ksweep_naive_${W}"  --export=ALL \
        $SCRIPTS/run_single_node.sh 1 $W $W 500 gpu naive  ksweep_naive_${W}
    submit "ksweep_shared_${W}" --export=ALL \
        $SCRIPTS/run_single_node.sh 1 $W $W 500 gpu shared ksweep_shared_${W}
done

# ---- MPI-IO checkpoint cost -------------------------------------------------
for K in 0 100 25; do
    submit "io_${K}" --export=ALL --job-name=gol_io_${K} \
        $SCRIPTS/run_single_node.sh 6 12288 12288 500 gpu overlap io_${K}
done

echo
echo "Done. Watch with:  squeue -u \$USER"
echo "Submit log:        $LOGFILE"

