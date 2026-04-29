/*
 * MPI infrastructure: 2D Cartesian topology with periodic boundaries
 * and a halo exchange that exploits the E/W-then-N/S ordering trick to
 * populate corner halos implicitly.
 *
 * The grid each rank owns is row-major uint8_t of allocated dimensions
 * (local_w + 2) x (local_h + 2). Interior is x in [1..local_w], y in
 * [1..local_h]. Halos are at x in {0, local_w+1} and similarly for y.
 *
 * Halo exchange algorithm (CPU build):
 *   Phase 1 (E/W) uses MPI_Type_vector to send strided columns.
 *   Phase 2 (N/S) sends whole rows; corner halos are carried implicitly.
 *
 * GPU build differences:
 *   Spectrum MPI's CUDA-aware support handles contiguous device pointers
 *   correctly but does NOT handle strided derived datatypes -- the host
 *   pack code (opal_generic_simple_pack) tries to walk a device pointer
 *   one byte at a time and segfaults. So in the GPU build we manually
 *   pack columns into contiguous device scratch buffers using a tiny
 *   CUDA kernel, send/receive contiguous bytes, and unpack on receipt.
 *   Rows are already contiguous and pass straight through.
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "gol_mpi.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "gol_kernel.h"
#endif

/* Pick a 2D rank decomposition (a, b) with a*b == size, a | global_h,
 * b | global_w. Among valid pairs prefer the one with smallest |a-b| so
 * the local tile is as square as possible. MPI_Dims_create is the standard
 * choice but it picks the most-square factorization unconditionally, which
 * for size==6 gives 2x3 -- and a power-of-two grid is never divisible by 3.
 * Returns 1 on success and writes dims; returns 0 if no valid pair exists. */
static int pick_compatible_dims(int size, int global_w, int global_h, int dims[2]) {
    int best_a = 0, best_b = 0;
    int best_diff = -1;
    for (int a = 1; a <= size; ++a) {
        if (size % a != 0) continue;
        int b = size / a;
        if (global_h % a != 0 || global_w % b != 0) continue;
        int diff = a > b ? a - b : b - a;
        if (best_diff < 0 || diff < best_diff) {
            best_a = a; best_b = b; best_diff = diff;
        }
    }
    if (best_diff < 0) return 0;
    dims[0] = best_a; dims[1] = best_b;
    return 1;
}

void gol_topology_init(gol_topology_t* t, int global_w, int global_h, int on_device) {
    MPI_Comm_size(MPI_COMM_WORLD, &t->size);
    MPI_Comm_rank(MPI_COMM_WORLD, &t->rank);
    t->on_device = on_device;

    if (!pick_compatible_dims(t->size, global_w, global_h, t->dims)) {
        if (t->rank == 0) {
            fprintf(stderr,
                "error: cannot decompose %d ranks over a %dx%d grid "
                "(no factorization a*b=%d with a|H and b|W)\n",
                t->size, global_w, global_h, t->size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, t->dims, periods, /*reorder=*/1, &t->cart);
    MPI_Cart_coords(t->cart, t->rank, 2, t->coords);

    MPI_Cart_shift(t->cart, 0, 1, &t->north, &t->south);
    MPI_Cart_shift(t->cart, 1, 1, &t->west,  &t->east);

    t->local_w = global_w / t->dims[1];
    t->local_h = global_h / t->dims[0];
    t->alloc_w = (size_t)t->local_w + 2;
    t->alloc_h = (size_t)t->local_h + 2;

    MPI_Type_vector(t->local_h, 1, (int)t->alloc_w, MPI_BYTE, &t->col_type);
    MPI_Type_commit(&t->col_type);

    t->d_col_send_w = t->d_col_send_e = NULL;
    t->d_col_recv_w = t->d_col_recv_e = NULL;
#ifdef USE_CUDA
    if (on_device) {
        size_t cb = (size_t)t->local_h;
        cudaMalloc((void**)&t->d_col_send_w, cb);
        cudaMalloc((void**)&t->d_col_send_e, cb);
        cudaMalloc((void**)&t->d_col_recv_w, cb);
        cudaMalloc((void**)&t->d_col_recv_e, cb);
    }
#else
    (void)on_device;
#endif
}

void gol_topology_free(gol_topology_t* t) {
    MPI_Type_free(&t->col_type);
    MPI_Comm_free(&t->cart);
#ifdef USE_CUDA
    if (t->on_device) {
        cudaFree(t->d_col_send_w);
        cudaFree(t->d_col_send_e);
        cudaFree(t->d_col_recv_w);
        cudaFree(t->d_col_recv_e);
    }
#endif
}

void gol_halo_exchange(uint8_t* grid, const gol_topology_t* t) {
    const size_t aw = t->alloc_w;
    const int lw = t->local_w, lh = t->local_h;
    MPI_Request reqs[4];
    MPI_Status  stats[4];
    /* Direction-keyed tags. With a periodic dim of size 1 every neighbor is
     * self, and a single tag would let MPI's FIFO matching pair the W-recv
     * with the W-send (it should pair with the E-send). Tagging by the
     * *destination* direction makes recv-from-W match send-to-E unambiguously. */
    const int TAG_TO_W = 100, TAG_TO_E = 101;
    const int TAG_TO_N = 200, TAG_TO_S = 201;

    /* If a Cartesian dim has size 1 every neighbor along that axis is self.
     * Spectrum MPI's CUDA-aware loopback path can't dereference device
     * pointers (we get MPI_ERR_BUFFER), so do the wrap locally instead. */
    const int self_ew = (t->dims[1] == 1);
    const int self_ns = (t->dims[0] == 1);

    /* ---- Phase 1: E / W ----
     * GPU path: pack into contiguous device buffers, exchange bytes, unpack.
     * CPU path: ship the strided column directly with MPI_Type_vector.
     * Dispatch is at runtime, not #ifdef, because the CUDA-built binary is
     * also used for --mode cpu. */
#ifdef USE_CUDA
    if (t->on_device) {
        if (self_ew) {
            /* Wrap: W halo <- E interior column, E halo <- W interior column. */
            gol_pack_col  (t->d_col_send_e, grid + 1 * aw + lw,      (int)aw, lh, 0);
            gol_pack_col  (t->d_col_send_w, grid + 1 * aw + 1,       (int)aw, lh, 0);
            gol_unpack_col(grid + 1 * aw + 0,        (int)aw, t->d_col_send_e, lh, 0);
            gol_unpack_col(grid + 1 * aw + (lw + 1), (int)aw, t->d_col_send_w, lh, 0);
            cudaStreamSynchronize(0);
        } else {
            gol_pack_col(t->d_col_send_w, grid + 1 * aw + 1,  (int)aw, lh, 0);
            gol_pack_col(t->d_col_send_e, grid + 1 * aw + lw, (int)aw, lh, 0);
            cudaStreamSynchronize(0);

            /* recv-from-west matches west neighbor's send-to-east, hence TAG_TO_E. */
            MPI_Irecv(t->d_col_recv_w, lh, MPI_BYTE, t->west, TAG_TO_E, t->cart, &reqs[0]);
            MPI_Irecv(t->d_col_recv_e, lh, MPI_BYTE, t->east, TAG_TO_W, t->cart, &reqs[1]);
            MPI_Isend(t->d_col_send_w, lh, MPI_BYTE, t->west, TAG_TO_W, t->cart, &reqs[2]);
            MPI_Isend(t->d_col_send_e, lh, MPI_BYTE, t->east, TAG_TO_E, t->cart, &reqs[3]);
            MPI_Waitall(4, reqs, stats);

            gol_unpack_col(grid + 1 * aw + 0,        (int)aw, t->d_col_recv_w, lh, 0);
            gol_unpack_col(grid + 1 * aw + (lw + 1), (int)aw, t->d_col_recv_e, lh, 0);
            cudaStreamSynchronize(0);
        }
    } else
#endif
    {
        uint8_t* col_send_w = grid + 1 * aw + 1;
        uint8_t* col_send_e = grid + 1 * aw + lw;
        uint8_t* col_recv_w = grid + 1 * aw + 0;
        uint8_t* col_recv_e = grid + 1 * aw + (lw + 1);

        if (self_ew) {
            for (int y = 0; y < lh; ++y) {
                col_recv_w[(size_t)y * aw] = col_send_e[(size_t)y * aw];
                col_recv_e[(size_t)y * aw] = col_send_w[(size_t)y * aw];
            }
        } else {
            MPI_Irecv(col_recv_w, 1, t->col_type, t->west, TAG_TO_E, t->cart, &reqs[0]);
            MPI_Irecv(col_recv_e, 1, t->col_type, t->east, TAG_TO_W, t->cart, &reqs[1]);
            MPI_Isend(col_send_w, 1, t->col_type, t->west, TAG_TO_W, t->cart, &reqs[2]);
            MPI_Isend(col_send_e, 1, t->col_type, t->east, TAG_TO_E, t->cart, &reqs[3]);
            MPI_Waitall(4, reqs, stats);
        }
    }

    /* ---- Phase 2: N / S ----
     * Whole rows of length alloc_w are contiguous in memory and so pass
     * straight through CUDA-aware MPI without packing. The N/S row
     * payload includes the W and E halo columns we just wrote, so
     * corner halos arrive implicitly.
     */
    uint8_t* row_send_n = grid + 1 * aw + 0;
    uint8_t* row_send_s = grid + (size_t)lh * aw + 0;
    uint8_t* row_recv_n = grid + 0 * aw + 0;
    uint8_t* row_recv_s = grid + (size_t)(lh + 1) * aw + 0;
    int row_count = (int)aw;

    if (self_ns) {
#ifdef USE_CUDA
        if (t->on_device) {
            cudaMemcpyAsync(row_recv_n, row_send_s, (size_t)row_count, cudaMemcpyDeviceToDevice, 0);
            cudaMemcpyAsync(row_recv_s, row_send_n, (size_t)row_count, cudaMemcpyDeviceToDevice, 0);
            cudaStreamSynchronize(0);
        } else
#endif
        {
            for (int x = 0; x < row_count; ++x) row_recv_n[x] = row_send_s[x];
            for (int x = 0; x < row_count; ++x) row_recv_s[x] = row_send_n[x];
        }
    } else {
        MPI_Irecv(row_recv_n, row_count, MPI_BYTE, t->north, TAG_TO_S, t->cart, &reqs[0]);
        MPI_Irecv(row_recv_s, row_count, MPI_BYTE, t->south, TAG_TO_N, t->cart, &reqs[1]);
        MPI_Isend(row_send_n, row_count, MPI_BYTE, t->north, TAG_TO_N, t->cart, &reqs[2]);
        MPI_Isend(row_send_s, row_count, MPI_BYTE, t->south, TAG_TO_S, t->cart, &reqs[3]);
        MPI_Waitall(4, reqs, stats);
    }
}

