#!/bin/bash
# Smoke test: 1 node, 1 rank, small grid. Use to confirm the binary
# launches before queueing big jobs.
#
#   sbatch scripts/run_debug.sh

#SBATCH --job-name=gol_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=bench/results/slurm-%j.out

RCS_ID=PCPGbrnr
PROJ=/gpfs/u/scratch/${RCS_ID}/parallel-group-project

module load xl_r spectrum-mpi cuda/11.2
cd $PROJ
mkdir -p bench/results

jsrun -n 1 -a 1 -c 4 -g 1 \
    ./bin/gol_mpi_cuda --w 1024 --h 1024 --steps 100 \
        --mode gpu --kernel naive --label debug
