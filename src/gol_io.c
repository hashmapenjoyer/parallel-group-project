/*
 * I/O for the parallel Game of Life:
 *
 *   - Local seeding (random + named patterns) into the interior of a
 *     halo'd tile.
 *   - Collective MPI-IO checkpoint write of the global grid.
 *   - Rank-0 PBM dump for visual sanity-checking on small grids.
 *
 * The MPI-IO checkpoint uses a 2D subarray view so each rank writes
 * exactly its own tile into the right offset of one shared file. This
 * is the same pattern used by climate / stencil codes for parallel
 * checkpointing -- every rank participates in one collective write.
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clockcycle.h"
#include "gol_io.h"
#include "gol_mpi.h"

static uint64_t xorshift64(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

void gol_seed_random(uint8_t* grid, const gol_topology_t* t,
                     uint64_t seed, double density) {
    // Per-rank deterministic seed so output is reproducible per (seed, P).
    uint64_t s = seed ^ ((uint64_t)t->rank * 0x9E3779B97F4A7C15ULL);
    if (s == 0) s = 1;
    uint64_t threshold = (uint64_t)(density * (double)UINT64_MAX);
    const size_t aw = t->alloc_w;
    for (int y = 1; y <= t->local_h; ++y) {
        for (int x = 1; x <= t->local_w; ++x) {
            uint64_t r = xorshift64(&s);
            grid[y * aw + x] = (r < threshold) ? 1 : 0;
        }
    }
}

void gol_seed_pattern(uint8_t* grid, const gol_topology_t* t, int global_w, int global_h, const char* name) {
    int cx = global_w / 2, cy = global_h / 2;
    int my_x0 = t->coords[1] * t->local_w; // global x of my (1,?) col
    int my_y0 = t->coords[0] * t->local_h;

    int pts[5][2] = {{0,0}};
    int npts = 0;
    if (!strcmp(name, "blinker")) {
        pts[0][0]=-1; pts[0][1]=0;
        pts[1][0]= 0; pts[1][1]=0;
        pts[2][0]= 1; pts[2][1]=0;
        npts = 3;
    } else if (!strcmp(name, "glider")) {
        pts[0][0]=1; pts[0][1]=0;
        pts[1][0]=2; pts[1][1]=1;
        pts[2][0]=0; pts[2][1]=2;
        pts[3][0]=1; pts[3][1]=2;
        pts[4][0]=2; pts[4][1]=2;
        npts = 5;
    } else if (!strcmp(name, "r-pentomino")) {
        pts[0][0]=1; pts[0][1]=0;
        pts[1][0]=2; pts[1][1]=0;
        pts[2][0]=0; pts[2][1]=1;
        pts[3][0]=1; pts[3][1]=1;
        pts[4][0]=1; pts[4][1]=2;
        npts = 5;
    }
    const size_t aw = t->alloc_w;
    for (int i = 0; i < npts; ++i) {
        int gx = cx + pts[i][0];
        int gy = cy + pts[i][1];
        int lx = gx - my_x0;
        int ly = gy - my_y0;
        if (lx >= 0 && lx < t->local_w && ly >= 0 && ly < t->local_h) {
            grid[(ly + 1) * aw + (lx + 1)] = 1;
        }
    }
}

double gol_checkpoint_write(const uint8_t* grid, const gol_topology_t* t,
                            int global_w, int global_h, const char* path) {
    // Pack the local interior into a contiguous buffer to avoid having
    // to describe a strided file view AND a strided memory view.
    size_t local_bytes = (size_t)t->local_w * t->local_h;
    uint8_t* buf = (uint8_t*)malloc(local_bytes);
    const size_t aw = t->alloc_w;
    for (int y = 0; y < t->local_h; ++y) {
        memcpy(buf + (size_t)y * t->local_w,
               grid + (size_t)(y + 1) * aw + 1,
               (size_t)t->local_w);
    }

    int gsizes[2] = {global_h, global_w};
    int lsizes[2] = {t->local_h, t->local_w};
    int starts[2] = {t->coords[0] * t->local_h, t->coords[1] * t->local_w};

    MPI_Datatype filetype;
    MPI_Type_create_subarray(2, gsizes, lsizes, starts,
                             MPI_ORDER_C, MPI_BYTE, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File fh;
    MPI_File_open(t->cart, path,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_BYTE, filetype, "native", MPI_INFO_NULL);

    uint64_t t0 = clock_now();
    MPI_File_write_all(fh, buf, (int)local_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
    uint64_t t1 = clock_now();

    MPI_File_close(&fh);
    MPI_Type_free(&filetype);
    free(buf);
    return clock_seconds(t0, t1);
}

void gol_dump_pbm(const uint8_t* grid, const gol_topology_t* t,
                  int global_w, int global_h, const char* path) {
    // Pack interior, gather to rank 0, write a P4 PBM.
    size_t local_bytes = (size_t)t->local_w * t->local_h;
    uint8_t* sendbuf = (uint8_t*)malloc(local_bytes);
    const size_t aw = t->alloc_w;
    for (int y = 0; y < t->local_h; ++y) {
        memcpy(sendbuf + (size_t)y * t->local_w,
               grid + (size_t)(y + 1) * aw + 1,
               (size_t)t->local_w);
    }

    uint8_t* recvbuf = NULL;
    if (t->rank == 0) recvbuf = (uint8_t*)malloc((size_t)global_w * global_h);

    // Gather assumes rank order matches Cart order (which Cart_create may
    // have reordered). For a small verification dump this is fine; we
    // reconstruct using each rank's coords below.
    int* all_coords = NULL;
    if (t->rank == 0) all_coords = (int*)malloc((size_t)t->size * 2 * sizeof(int));
    MPI_Gather(t->coords, 2, MPI_INT, all_coords, 2, MPI_INT, 0, t->cart);

    int* recvcounts = NULL; int* displs = NULL;
    if (t->rank == 0) {
        recvcounts = (int*)malloc((size_t)t->size * sizeof(int));
        displs     = (int*)malloc((size_t)t->size * sizeof(int));
        for (int r = 0; r < t->size; ++r) {
            recvcounts[r] = (int)local_bytes;
            displs[r] = r * (int)local_bytes;
        }
    }
    MPI_Gatherv(sendbuf, (int)local_bytes, MPI_BYTE,
                recvbuf, recvcounts, displs, MPI_BYTE, 0, t->cart);

    if (t->rank == 0) {
        uint8_t* image = (uint8_t*)malloc((size_t)global_w * global_h);
        for (int r = 0; r < t->size; ++r) {
            int rx = all_coords[r * 2 + 1] * t->local_w;
            int ry = all_coords[r * 2 + 0] * t->local_h;
            for (int y = 0; y < t->local_h; ++y) {
                memcpy(image + (size_t)(ry + y) * global_w + rx,
                       recvbuf + (size_t)r * local_bytes + (size_t)y * t->local_w,
                       (size_t)t->local_w);
            }
        }
        FILE* f = fopen(path, "wb");
        if (f) {
            fprintf(f, "P4\n%d %d\n", global_w, global_h);
            int row_bytes = (global_w + 7) / 8;
            uint8_t* row = (uint8_t*)calloc(row_bytes, 1);
            for (int y = 0; y < global_h; ++y) {
                memset(row, 0, row_bytes);
                for (int x = 0; x < global_w; ++x)
                    if (image[y * global_w + x])
                        row[x / 8] |= (uint8_t)(0x80 >> (x % 8));
                fwrite(row, 1, row_bytes, f);
            }
            free(row);
            fclose(f);
        }
        free(image);
        free(recvbuf); free(recvcounts); free(displs); free(all_coords);
    }
    free(sendbuf);
}
