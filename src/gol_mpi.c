/*
 * MPI infrastructure: 2D Cartesian topology with periodic boundaries
 * and a halo exchange that exploits the E/W-then-N/S ordering trick to
 * populate corner halos implicitly.
 *
 * The grid each rank owns is row-major uint8_t of allocated dimensions
 * (local_w + 2) x (local_h + 2). The interior is at x in [1..local_w],
 * y in [1..local_h]. Halo cells are at x in {0, local_w+1} and similarly
 * for y. Corners are at the four (0/lw+1, 0/lh+1) intersections.
 *
 * Halo exchange algorithm:
 *   Phase 1 (E/W): post Isend/Irecv for the two interior edge columns
 *                  using an MPI_Type_vector that stride-skips full rows.
 *                  After Waitall the W and E halos hold valid data for
 *                  y in [1..local_h]; corners are still uninitialized.
 *   Phase 2 (N/S): send full rows of length (local_w+2). Because the
 *                  row at y=1 now contains the W and E halo values that
 *                  Phase 1 just wrote, the corner cells are carried
 *                  inside the N/S messages -- no diagonal exchange.
 *
 * On AiMOS Spectrum MPI is CUDA-aware, so `grid` may be a device pointer
 * passed straight to MPI_Isend / MPI_Irecv. The caller is responsible
 * for any cudaStreamSynchronize before calling this function (so the
 * device data is consistent) and after (so subsequent kernels see the
 * fresh halos).
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "gol_mpi.h"

void gol_topology_init(gol_topology_t* t, int global_w, int global_h) {
    MPI_Comm_size(MPI_COMM_WORLD, &t->size);
    MPI_Comm_rank(MPI_COMM_WORLD, &t->rank);

    t->dims[0] = 0; t->dims[1] = 0;
    MPI_Dims_create(t->size, 2, t->dims);

    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, t->dims, periods, /*reorder=*/1, &t->cart);
    MPI_Cart_coords(t->cart, t->rank, 2, t->coords);

    /* dim 0 = rows of the rank grid (N/S axis), dim 1 = cols (E/W axis). */
    MPI_Cart_shift(t->cart, 0, 1, &t->north, &t->south);
    MPI_Cart_shift(t->cart, 1, 1, &t->west,  &t->east);

    if (global_h % t->dims[0] != 0 || global_w % t->dims[1] != 0) {
        if (t->rank == 0) {
            fprintf(stderr,
                "error: global %dx%d not divisible by rank grid %dx%d\n",
                global_w, global_h, t->dims[1], t->dims[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    t->local_w = global_w / t->dims[1];
    t->local_h = global_h / t->dims[0];
    t->alloc_w = (size_t)t->local_w + 2;
    t->alloc_h = (size_t)t->local_h + 2;

    /* Column datatype: local_h elements with stride alloc_w. */
    MPI_Type_vector(t->local_h, 1, (int)t->alloc_w, MPI_BYTE, &t->col_type);
    MPI_Type_commit(&t->col_type);
}

void gol_topology_free(gol_topology_t* t) {
    MPI_Type_free(&t->col_type);
    MPI_Comm_free(&t->cart);
}

void gol_halo_exchange(uint8_t* grid, const gol_topology_t* t) {
    const size_t aw = t->alloc_w;
    const int lw = t->local_w, lh = t->local_h;
    MPI_Request reqs[4];
    MPI_Status  stats[4];
    const int TAG_EW = 100, TAG_NS = 200;

    /* ---- Phase 1: E / W ----
     * Send first interior column to W, last interior column to E.
     * Receive into W and E halo columns.
     */
    uint8_t* col_send_w = grid + 1 * aw + 1;        /* (1, 1) */
    uint8_t* col_send_e = grid + 1 * aw + lw;       /* (lw, 1) */
    uint8_t* col_recv_w = grid + 1 * aw + 0;        /* (0, 1) */
    uint8_t* col_recv_e = grid + 1 * aw + (lw + 1); /* (lw+1, 1) */

    MPI_Irecv(col_recv_w, 1, t->col_type, t->west, TAG_EW, t->cart, &reqs[0]);
    MPI_Irecv(col_recv_e, 1, t->col_type, t->east, TAG_EW, t->cart, &reqs[1]);
    MPI_Isend(col_send_w, 1, t->col_type, t->west, TAG_EW, t->cart, &reqs[2]);
    MPI_Isend(col_send_e, 1, t->col_type, t->east, TAG_EW, t->cart, &reqs[3]);
    MPI_Waitall(4, reqs, stats);

    /* ---- Phase 2: N / S ----
     * Whole rows (length alloc_w) carry corner cells implicitly.
     */
    uint8_t* row_send_n = grid + 1 * aw + 0;       /* y = 1     */
    uint8_t* row_send_s = grid + (size_t)lh * aw + 0; /* y = lh */
    uint8_t* row_recv_n = grid + 0 * aw + 0;       /* y = 0     */
    uint8_t* row_recv_s = grid + (size_t)(lh + 1) * aw + 0; /* y = lh+1 */
    int row_count = (int)aw;

    MPI_Irecv(row_recv_n, row_count, MPI_BYTE, t->north, TAG_NS, t->cart, &reqs[0]);
    MPI_Irecv(row_recv_s, row_count, MPI_BYTE, t->south, TAG_NS, t->cart, &reqs[1]);
    MPI_Isend(row_send_n, row_count, MPI_BYTE, t->north, TAG_NS, t->cart, &reqs[2]);
    MPI_Isend(row_send_s, row_count, MPI_BYTE, t->south, TAG_NS, t->cart, &reqs[3]);
    MPI_Waitall(4, reqs, stats);
}
