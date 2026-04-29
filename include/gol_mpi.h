#ifndef GOL_MPI_H
#define GOL_MPI_H

#include <mpi.h>
#include <stdint.h>

typedef struct {
    MPI_Comm cart;
    int dims[2];           /* dims[0] = rows of rank grid, dims[1] = cols */
    int coords[2];
    int rank, size;
    int north, south, east, west;
    int local_w, local_h;  /* interior tile size (no halo) */
    size_t alloc_w, alloc_h; /* = local_w + 2, local_h + 2 */
    MPI_Datatype col_type;   /* one column of length local_h, stride alloc_w (CPU mode) */
    /* Device scratch buffers used only in the GPU build; left NULL in CPU
     * mode. Spectrum MPI's CUDA-aware path doesn't handle strided derived
     * datatypes, so we pack columns into contiguous device buffers before
     * MPI_Isend. Declared unconditionally to keep struct layout stable
     * across compilation units. */
    uint8_t* d_col_send_w;
    uint8_t* d_col_send_e;
    uint8_t* d_col_recv_w;
    uint8_t* d_col_recv_e;
    /* Selects the halo-exchange path at runtime. The CUDA-built binary is
     * also used for `--mode cpu`, in which case the grid is host memory and
     * the GPU pack/unpack path must NOT be taken (it would deref a host
     * pointer as a device pointer). */
    int on_device;
} gol_topology_t;

void gol_topology_init(gol_topology_t* t, int global_w, int global_h, int on_device);
void gol_topology_free(gol_topology_t* t);

/*
 * Exchange the four face halos. For CUDA-aware MPI, `grid` may be a device
 * pointer; the caller is responsible for stream synchronization before/after.
 *
 * Order is E/W first, N/S second so corner cells are populated implicitly.
 */
void gol_halo_exchange(uint8_t* grid, const gol_topology_t* t);

#endif

