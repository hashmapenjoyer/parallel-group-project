/*
 * CUDA kernels for one Game of Life timestep (B3/S23, toroidal at the
 * domain level via halos managed by the MPI layer).
 *
 * Grid layout: row-major uint8_t, allocated dimensions (lw+2) x (lh+2)
 * with a one-cell halo on every side. Interior indices are x in [1..lw],
 * y in [1..lh]. The pitch in bytes equals (lw+2) -- we use plain cudaMalloc
 * (not cudaMallocPitch), so stride == width.
 *
 * Four entry points:
 *   gol_step_gpu          -- naive global-memory kernel, full interior
 *   gol_step_gpu_shared   -- shared-memory tiled variant, full interior
 *   gol_step_gpu_interior -- skip the outermost ring of interior cells
 *   gol_step_gpu_boundary -- only the outermost ring (cells touching halos)
 *
 * The interior/boundary split is what enables comm/comp overlap: launch
 * the interior kernel on stream A, post the halo Isend/Irecv, then once
 * MPI_Waitall returns launch the boundary kernel on stream B.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "gol_kernel.h"

#define BX 32
#define BY 8

#define IDX(x, y, stride) ((y) * (stride) + (x))

__device__ __forceinline__ uint8_t gol_rule(uint8_t c, int n) {
    return (uint8_t)((c && (n == 2 || n == 3)) || (!c && n == 3));
}

// naive global-memory kernel
__global__ void k_gol_naive(const uint8_t* __restrict__ in,
                            uint8_t* __restrict__ out,
                            int lw, int lh, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x > lw || y > lh) return;

    int n = 0;
    n += in[IDX(x - 1, y - 1, stride)];
    n += in[IDX(x,     y - 1, stride)];
    n += in[IDX(x + 1, y - 1, stride)];
    n += in[IDX(x - 1, y,     stride)];
    n += in[IDX(x + 1, y,     stride)];
    n += in[IDX(x - 1, y + 1, stride)];
    n += in[IDX(x,     y + 1, stride)];
    n += in[IDX(x + 1, y + 1, stride)];

    out[IDX(x, y, stride)] = gol_rule(in[IDX(x, y, stride)], n);
}

// shared-memory tiled kernel
// Each block loads a (BX+2) x (BY+2) tile into shared memory cooperatively.
// Threads with low ids do the extra halo loads.

__global__ void k_gol_shared(const uint8_t* __restrict__ in,
                             uint8_t* __restrict__ out,
                             int lw, int lh, int stride) {
    __shared__ uint8_t s[(BY + 2)][(BX + 2)];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * BX + tx + 1;
    int gy = blockIdx.y * BY + ty + 1;

    //  Center cell valid even pas the interior, since halos exist.
    if (gx <= lw + 1 && gy <= lh + 1)
        s[ty + 1][tx + 1] = in[IDX(gx, gy, stride)];

    // Halo edges
    if (tx == 0 && gx >= 1 && gy <= lh + 1)
        s[ty + 1][0] = in[IDX(gx - 1, gy, stride)];
    if (tx == BX - 1 && gx <= lw && gy <= lh + 1)
        s[ty + 1][BX + 1] = in[IDX(gx + 1, gy, stride)];
    if (ty == 0 && gy >= 1 && gx <= lw + 1)
        s[0][tx + 1] = in[IDX(gx, gy - 1, stride)];
    if (ty == BY - 1 && gy <= lh && gx <= lw + 1)
        s[BY + 1][tx + 1] = in[IDX(gx, gy + 1, stride)];

    // Halo corners
    if (tx == 0 && ty == 0)
        s[0][0] = in[IDX(gx - 1, gy - 1, stride)];
    if (tx == BX - 1 && ty == 0)
        s[0][BX + 1] = in[IDX(gx + 1, gy - 1, stride)];
    if (tx == 0 && ty == BY - 1)
        s[BY + 1][0] = in[IDX(gx - 1, gy + 1, stride)];
    if (tx == BX - 1 && ty == BY - 1)
        s[BY + 1][BX + 1] = in[IDX(gx + 1, gy + 1, stride)];

    __syncthreads();

    if (gx > lw || gy > lh) return;

    int n = 0;
    n += s[ty    ][tx    ]; n += s[ty    ][tx + 1]; n += s[ty    ][tx + 2];
    n += s[ty + 1][tx    ];                          n += s[ty + 1][tx + 2];
    n += s[ty + 2][tx    ]; n += s[ty + 2][tx + 1]; n += s[ty + 2][tx + 2];

    out[IDX(gx, gy, stride)] = gol_rule(s[ty + 1][tx + 1], n);
}

// interior-only kernel (skip outermost interior ring) 
__global__ void k_gol_interior(const uint8_t* __restrict__ in,
                               uint8_t* __restrict__ out,
                               int lw, int lh, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 2;
    if (x > lw - 1 || y > lh - 1) return;

    int n = 0;
    n += in[IDX(x - 1, y - 1, stride)];
    n += in[IDX(x,     y - 1, stride)];
    n += in[IDX(x + 1, y - 1, stride)];
    n += in[IDX(x - 1, y,     stride)];
    n += in[IDX(x + 1, y,     stride)];
    n += in[IDX(x - 1, y + 1, stride)];
    n += in[IDX(x,     y + 1, stride)];
    n += in[IDX(x + 1, y + 1, stride)];

    out[IDX(x, y, stride)] = gol_rule(in[IDX(x, y, stride)], n);
}

// Boundary-only kernel: outermost interior ring --------
// Launched as a 1D grid covering the perimeter of (lw,lh).
// 
__global__ void k_gol_boundary(const uint8_t* __restrict__ in,
                               uint8_t* __restrict__ out,
                               int lw, int lh, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int perim = 2 * lw + 2 * (lh - 2);
    if (tid >= perim) return;

    int x, y;
    if (tid < lw) {
        x = tid + 1; y = 1;
    } else if (tid < 2 * lw) {
        x = (tid - lw) + 1; y = lh;
    } else if (tid < 2 * lw + (lh - 2)) {
        x = 1; y = (tid - 2 * lw) + 2;
    } else {
        x = lw; y = (tid - 2 * lw - (lh - 2)) + 2;
    }

    int n = 0;
    n += in[IDX(x - 1, y - 1, stride)];
    n += in[IDX(x,     y - 1, stride)];
    n += in[IDX(x + 1, y - 1, stride)];
    n += in[IDX(x - 1, y,     stride)];
    n += in[IDX(x + 1, y,     stride)];
    n += in[IDX(x - 1, y + 1, stride)];
    n += in[IDX(x,     y + 1, stride)];
    n += in[IDX(x + 1, y + 1, stride)];

    out[IDX(x, y, stride)] = gol_rule(in[IDX(x, y, stride)], n);
}

// host wrappers

extern "C" void gol_step_gpu(const uint8_t* d_in, uint8_t* d_out,
                             int lw, int lh, cudaStream_t stream) {
    int stride = lw + 2;
    dim3 block(BX, BY);
    dim3 grid((lw + BX - 1) / BX, (lh + BY - 1) / BY);
    k_gol_naive<<<grid, block, 0, stream>>>(d_in, d_out, lw, lh, stride);
}

extern "C" void gol_step_gpu_shared(const uint8_t* d_in, uint8_t* d_out,
                                    int lw, int lh, cudaStream_t stream) {
    int stride = lw + 2;
    dim3 block(BX, BY);
    dim3 grid((lw + BX - 1) / BX, (lh + BY - 1) / BY);
    k_gol_shared<<<grid, block, 0, stream>>>(d_in, d_out, lw, lh, stride);
}

extern "C" void gol_step_gpu_interior(const uint8_t* d_in, uint8_t* d_out,
                                      int lw, int lh, cudaStream_t stream) {
    int stride = lw + 2;
    int iw = lw - 2, ih = lh - 2;
    if (iw <= 0 || ih <= 0) return;
    dim3 block(BX, BY);
    dim3 grid((iw + BX - 1) / BX, (ih + BY - 1) / BY);
    k_gol_interior<<<grid, block, 0, stream>>>(d_in, d_out, lw, lh, stride);
}

extern "C" void gol_step_gpu_boundary(const uint8_t* d_in, uint8_t* d_out,
                                      int lw, int lh, cudaStream_t stream) {
    int stride = lw + 2;
    int perim = 2 * lw + 2 * (lh - 2);
    if (perim <= 0) return;
    int threads = 128;
    int blocks = (perim + threads - 1) / threads;
    k_gol_boundary<<<blocks, threads, 0, stream>>>(d_in, d_out, lw, lh, stride);
}

/* -------- column pack / unpack (for halo exchange) -------- */
__global__ void k_pack_col(uint8_t* __restrict__ dst,
                           const uint8_t* __restrict__ src,
                           int stride, int height) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < height) dst[y] = src[y * stride];
}

__global__ void k_unpack_col(uint8_t* __restrict__ dst, int stride,
                             const uint8_t* __restrict__ src, int height) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < height) dst[y * stride] = src[y];
}

extern "C" void gol_pack_col(uint8_t* dst, const uint8_t* src,
                             int stride, int height, cudaStream_t stream) {
    int threads = 128;
    int blocks = (height + threads - 1) / threads;
    k_pack_col<<<blocks, threads, 0, stream>>>(dst, src, stride, height);
}

extern "C" void gol_unpack_col(uint8_t* dst, int stride,
                               const uint8_t* src, int height,
                               cudaStream_t stream) {
    int threads = 128;
    int blocks = (height + threads - 1) / threads;
    k_unpack_col<<<blocks, threads, 0, stream>>>(dst, stride, src, height);
}
