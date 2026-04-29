#!/bin/bash
# Single-node (1-6 ranks, all on one DCS node) submission template.
#
# Usage:
#   sbatch scripts/run_single_node.sh <ranks> <W> <H> <steps> <mode> <kernel> <label>

#SBATCH --job-name=gol_single
#SBATCH --partition=dcs-2024
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:6
#SBATCH --time=00:30:00
#SBATCH --output=bench/results/slurm-%j.out

RCS_ID=PCPGbrnr
PROJ=/gpfs/u/home/PCPG/${RCS_ID}/scratch/parallel-group-project

RANKS=${1:-6}
W=${2:-8192}
H=${3:-8192}
STEPS=${4:-1000}
MODE=${5:-gpu}
KERNEL=${6:-overlap}
LABEL=${7:-single_node}
CSV=$PROJ/bench/results/results.csv

module load xl_r spectrum-mpi cuda/11.2

cd $PROJ
mkdir -p bench/results

for TRIAL in 1 2 3; do
    mpirun -gpu -n $RANKS \
        ./bin/gol_mpi_cuda \
            --w $W --h $H --steps $STEPS \
            --mode $MODE --kernel $KERNEL \
            --csv $CSV --label $LABEL --trial $TRIAL
done
