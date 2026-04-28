# Massively Parallel Game of Life

MPI + CUDA implementation of Conway's Game of Life targeting RPI's
AiMOS supercomputer (POWER9 + V100 DCS nodes). 2D Cartesian
decomposition with toroidal halo exchange; CUDA-aware Spectrum MPI
passes device pointers straight to `MPI_Isend` / `MPI_Irecv`.

## Build

On AiMOS DCS:

```sh
module load xl_r spectrum-mpi cuda/11.2
make all
```

Local development build (no CUDA):

```sh
make gol_serial gol_mpi_cpu
```

Targets produced in `bin/`:

| Binary          | Description                                            |
|-----------------|--------------------------------------------------------|
| `gol_serial`    | Single-process CPU reference. Use as the oracle.       |
| `gol_mpi_cpu`   | MPI parallel, CPU compute step.                        |
| `gol_mpi_cuda`  | MPI parallel, CUDA compute step (requires CUDA-aware MPI). |
| `diff_grids`    | Raw-grid byte-comparison tool.                         |

## Run

```sh
# Serial
./bin/gol_serial 1024 1024 100 random 1 final.pbm

# MPI / CPU, 4 ranks
mpirun -n 4 ./bin/gol_mpi_cpu --w 1024 --h 1024 --steps 100 --mode cpu

# MPI / CUDA on AiMOS (jsrun), 6 ranks per DCS node
jsrun -n 6 -a 1 -c 4 -g 1 ./bin/gol_mpi_cuda \
      --w 32768 --h 32768 --steps 1000 --mode gpu --kernel overlap \
      --csv bench/results/run.csv --label strong_6
```

## Layout

```
include/   public headers (clockcycle.h, gol_kernel.h, gol_mpi.h, gol_io.h)
src/       all .c / .cu sources
tools/     diff_grids
scripts/   SLURM submission templates
bench/     run_all.sh harness + plot.py + results CSVs + oracle/
paper/     ACM sigconf LaTeX (main.tex + refs.bib + figures/)
```

## Verification

Run the serial oracle and the parallel build with the same parameters,
have both checkpoint the final state, then diff:

```sh
./bin/gol_serial 256 256 100 r-pentomino 1 /tmp/serial.pbm  # also dumps raw
mpirun -n 4 ./bin/gol_mpi_cpu --w 256 --h 256 --steps 100 \
       --init r-pentomino --checkpoint-every 100 --checkpoint /tmp/parallel.bin
./bin/diff_grids 256 256 /tmp/serial.bin /tmp/parallel.bin
```

(The serial binary writes PBM by default; for a raw dump compatible with
`diff_grids`, run the MPI binary on a single rank with `--checkpoint`.)

## Source repository

Mirror: `https://github.com/<user>/parallel-group-project` (GitHub link
included here in case the Submitty tarball is too large.)

## Key design decisions

- **Cells:** `uint8_t`, double-buffered, halo'd tile of `(local_w+2) x
  (local_h+2)` per rank.
- **Decomposition:** `MPI_Dims_create` chooses a balanced 2D grid;
  `MPI_Cart_create` with `periods={1,1}` for a torus.
- **Halo exchange:** E/W column exchange first (using
  `MPI_Type_vector`), then N/S row exchange (whole rows of length
  `local_w+2`). Corner cells are populated implicitly because the N/S
  rows include the W/E halos written by the previous phase.
- **CUDA kernels:** three variants -- naive global memory, shared-memory
  tiled (32x8 block + 1-cell halo), and an interior/boundary split for
  comm/comp overlap. The overlap variant launches the interior kernel
  on one stream, exchanges halos concurrently, then launches the
  boundary-only kernel.
- **MPI I/O:** optional collective subarray-view checkpoint write.
- **Timing:** POWER9 cycle counter (`mftb`) on AiMOS, `clock_gettime`
  fallback on x86.
