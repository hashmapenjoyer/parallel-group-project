#!/bin/bash
# Single-node (1-6 ranks, all on one DCS node) submission template.
#
# Usage:
#   sbatch scripts/run_single_node.sh <ranks> <W> <H> <steps> <mode> <kernel> <label>
#
# Edit RCS_ID below if needed; also confirm the AiMOS allocation.

#SBATCH --partition=dcs
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=00:30:00
#SBATCH --job-name=gol_single
#SBATCH --output=bench/results/slurm-%j.out

RCS_ID=PCPGbrnr
PROJ=/gpfs/u/scratch/${RCS_ID}/parallel-group-project

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
    jsrun -n $RANKS -a 1 -c 4 -g 1 \
        ./bin/gol_mpi_cuda \
            --w $W --h $H --steps $STEPS \
            --mode $MODE --kernel $KERNEL \
            --csv $CSV --label $LABEL --trial $TRIAL
done
