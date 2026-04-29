#ifndef GOL_KERNEL_H
#define GOL_KERNEL_H

#include <stdint.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Buffers d_in/d_out are device allocations of size (lw+2)*(lh+2) bytes,
 * row-major, with a 1-cell halo on every side. Halos must be valid before
 * a "full" launch; for "interior" launches halos are not read.
 */

void gol_step_gpu(const uint8_t* d_in, uint8_t* d_out,
                  int lw, int lh, cudaStream_t stream);

void gol_step_gpu_shared(const uint8_t* d_in, uint8_t* d_out,
                         int lw, int lh, cudaStream_t stream);

/* Skip the outermost ring of interior cells; used while halos are in flight. */
void gol_step_gpu_interior(const uint8_t* d_in, uint8_t* d_out,
                           int lw, int lh, cudaStream_t stream);

/* Compute only the outermost ring of interior cells (the ones that read halos). */
void gol_step_gpu_boundary(const uint8_t* d_in, uint8_t* d_out,
                           int lw, int lh, cudaStream_t stream);

/* Pack a strided column of `height` bytes from `src` (stride = `stride`)
 * into a contiguous device buffer `dst`. Spectrum MPI's host-side pack
 * cannot read device pointers, so we pack manually before MPI_Isend. */
void gol_pack_col(uint8_t* dst, const uint8_t* src, int stride, int height,
                  cudaStream_t stream);
void gol_unpack_col(uint8_t* dst, int stride, const uint8_t* src, int height,
                    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* USE_CUDA */
#endif
