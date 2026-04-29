#!/bin/bash
#SBATCH --job-name=gol_verify
#SBATCH --partition=dcs-2024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=bench/results/slurm-%j.out

PROJ=/gpfs/u/home/PCPG/PCPGbrnr/scratch/parallel-group-project
module load xl_r spectrum-mpi cuda/11.2
cd $PROJ
mkdir -p bench/results

W=256; H=256; STEPS=200; SEED=1

# Serial reference (write once per init)
./bin/gol_serial $W $H $STEPS random      $SEED bench/results/ref_random.pbm
./bin/gol_serial $W $H $STEPS r-pentomino 0     bench/results/ref_rp.pbm

check() {
    local label=$1 ref=$2 out=$3
    if cmp -s "$ref" "$out"; then echo "OK   $label"
    else echo "FAIL $label"; fi
}

for K in naive shared overlap; do
    mpirun -gpu -n 1 ./bin/gol_mpi_cuda --w $W --h $H --steps $STEPS \
        --mode gpu --kernel $K --init random --seed $SEED \
        --dump-final bench/results/test_${K}_random.pbm
    check "1rank/$K/random" bench/results/ref_random.pbm bench/results/test_${K}_random.pbm

    mpirun -gpu -n 1 ./bin/gol_mpi_cuda --w $W --h $H --steps $STEPS \
        --mode gpu --kernel $K --init r-pentomino \
        --dump-final bench/results/test_${K}_rp.pbm
    check "1rank/$K/rp" bench/results/ref_rp.pbm bench/results/test_${K}_rp.pbm
done

# Multi-rank with deterministic init (random seeding differs per-rank)
mpirun -gpu -n 4 ./bin/gol_mpi_cuda --w $W --h $H --steps $STEPS \
    --mode gpu --kernel overlap --init r-pentomino \
    --dump-final bench/results/test_4rank_rp.pbm
check "4rank/overlap/rp" bench/results/ref_rp.pbm bench/results/test_4rank_rp.pbm
