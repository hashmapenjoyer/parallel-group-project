#ifndef GOL_IO_H
#define GOL_IO_H

#include <mpi.h>
#include <stdint.h>
#include "gol_mpi.h"

/* Seed the local interior tile with random cells (density in [0,1]). */
void gol_seed_random(uint8_t* grid, const gol_topology_t* t,
                     uint64_t seed, double density);

/* Seed a named pattern (blinker/glider/r-pentomino) at the global center.
 * Each rank only writes the cells it owns. Other patterns are no-ops. */
void gol_seed_pattern(uint8_t* grid, const gol_topology_t* t,
                      int global_w, int global_h, const char* name);

/* Collective MPI-IO checkpoint of the current grid (interior only) to `path`.
 * Layout on disk: raw uint8_t, row-major, global_h rows of global_w bytes. */
double gol_checkpoint_write(const uint8_t* grid, const gol_topology_t* t,
                            int global_w, int global_h, const char* path);

/* Rank 0 dumps a portable bitmap (P4) of the gathered grid. Small grids only
 * (uses MPI_Gather; intended for verification, not benchmarking). */
void gol_dump_pbm(const uint8_t* grid, const gol_topology_t* t,
                  int global_w, int global_h, const char* path);

#endif
