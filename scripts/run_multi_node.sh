#!/bin/bash
# Multi-node (12-48 ranks across 2/4/8 DCS nodes) submission template.
#
# Pass --nodes=N on the sbatch command line; ntasks scales as 6*N.
#
# Usage:
#   sbatch --nodes=$N --ntasks=$((N*6)) scripts/run_multi_node.sh \
#          <ranks> <W> <H> <steps> <mode> <kernel> <label>

#SBATCH --job-name=gol_multi
#SBATCH --partition=dcs-2024
#SBATCH --gres=gpu:6
#SBATCH --time=01:00:00
#SBATCH --output=bench/results/slurm-%j.out

RCS_ID=PCPGbrnr
PROJ=/gpfs/u/home/PCPG/${RCS_ID}/scratch/parallel-group-project

RANKS=${1:-24}
W=${2:-32768}
H=${3:-32768}
STEPS=${4:-1000}
MODE=${5:-gpu}
KERNEL=${6:-overlap}
LABEL=${7:-multi_node}
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
