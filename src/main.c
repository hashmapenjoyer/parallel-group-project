/*
 * Driver for the parallel Game of Life. Owns the timestep loop, argument
 * parsing, allocation of the two grid buffers (with halos), and CSV
 * timing output.
 *
 * Compute mode is selected at runtime via --mode cpu|gpu and (for gpu)
 * --kernel naive|shared|overlap. The overlap variant runs the interior
 * kernel concurrently with halo_exchange, then a boundary-only kernel.
 *
 * The CSV is appended one row per (rank-0) timestep group; per-phase
 * timings are accumulated across the whole run and emitted once at the
 * end. This keeps the in-loop work limited to two clock_now() calls.
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clockcycle.h"
#include "gol_io.h"
#include "gol_mpi.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "gol_kernel.h"
#endif

void gol_step_cpu(const uint8_t* in, uint8_t* out, int lw, int lh);

typedef struct {
    int global_w, global_h;
    int steps;
    int trial;
    const char* mode;     // "cpu" or "gpu"
    const char* kernel;   // "naive" / "shared" / "overlap"
    const char* init;     // "random" / "blinker" / "glider" / "r-pentomino"
    uint64_t seed;
    double density;
    int checkpoint_every;
    const char* checkpoint_path;
    const char* dump_final;
    const char* csv_path;
    const char* label;
} args_t;

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s [--w WIDTH] [--h HEIGHT] [--steps N] [--mode cpu|gpu]\n"
        "          [--kernel naive|shared|overlap] [--init NAME]\n"
        "          [--seed S] [--density D] [--trial T]\n"
        "          [--checkpoint-every K] [--checkpoint PATH]\n"
        "          [--dump-final PATH] [--csv PATH] [--label STR]\n",
        prog);
}

static void parse_args(int argc, char** argv, args_t* args) {
    args->global_w = 1024; args->global_h = 1024;
    args->steps = 100; args->trial = 0;
    args->mode = "gpu"; args->kernel = "naive"; args->init = "random";
    args->seed = 1; args->density = 0.3;
    args->checkpoint_every = 0;
    args->checkpoint_path = "checkpoint.bin";
    args->dump_final = NULL; args->csv_path = NULL; args->label = "";
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        const char* v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if      (!strcmp(k, "--w") && v)               { args->global_w = atoi(v); ++i; }
        else if (!strcmp(k, "--h") && v)               { args->global_h = atoi(v); ++i; }
        else if (!strcmp(k, "--steps") && v)           { args->steps = atoi(v); ++i; }
        else if (!strcmp(k, "--mode") && v)            { args->mode = v; ++i; }
        else if (!strcmp(k, "--kernel") && v)          { args->kernel = v; ++i; }
        else if (!strcmp(k, "--init") && v)            { args->init = v; ++i; }
        else if (!strcmp(k, "--seed") && v)            { args->seed = strtoull(v, NULL, 10); ++i; }
        else if (!strcmp(k, "--density") && v)         { args->density = atof(v); ++i; }
        else if (!strcmp(k, "--trial") && v)           { args->trial = atoi(v); ++i; }
        else if (!strcmp(k, "--checkpoint-every") && v){ args->checkpoint_every = atoi(v); ++i; }
        else if (!strcmp(k, "--checkpoint") && v)      { args->checkpoint_path = v; ++i; }
        else if (!strcmp(k, "--dump-final") && v)      { args->dump_final = v; ++i; }
        else if (!strcmp(k, "--csv") && v)             { args->csv_path = v; ++i; }
        else if (!strcmp(k, "--label") && v)           { args->label = v; ++i; }
        else if (!strcmp(k, "--help") || !strcmp(k, "-h")) { usage(argv[0]); exit(0); }
        else { fprintf(stderr, "unknown arg: %s\n", k); usage(argv[0]); exit(1); }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    args_t a;
    parse_args(argc, argv, &a);

    gol_topology_t topo;
    /* Resolve rank before topology init so we can bind the GPU first --
     * gol_topology_init() does cudaMalloc for halo scratch when on_device,
     * and Spectrum MPI's CUDA-aware path rejects device pointers whose
     * CUcontext doesn't match the rank's bound device with MPI_ERR_BUFFER. */
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int use_gpu = !strcmp(a.mode, "gpu");
#ifdef USE_CUDA
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (n_devices > 0) cudaSetDevice(world_rank % n_devices);
    if (use_gpu && n_devices == 0 && world_rank == 0) {
        fprintf(stderr, "warning: --mode gpu but no CUDA device visible; falling back to CPU\n");
        use_gpu = 0;
    }
#else
    if (use_gpu && world_rank == 0)
        fprintf(stderr, "warning: built without USE_CUDA; falling back to CPU\n");
    use_gpu = 0;
#endif

    gol_topology_init(&topo, a.global_w, a.global_h, use_gpu);

    size_t bytes = topo.alloc_w * topo.alloc_h;
    uint8_t *cur = NULL, *nxt = NULL;

#ifdef USE_CUDA
    if (use_gpu) {
        cudaMalloc((void**) &cur, bytes);
        cudaMalloc((void**) &nxt, bytes);
        cudaMemset(cur, 0, bytes);
        cudaMemset(nxt, 0, bytes);
    } else
#endif
    {
        cur = (uint8_t*) calloc(bytes, 1);
        nxt = (uint8_t*) calloc(bytes, 1);
    }

    // Seed into a host buffer, then copy to device if needed.
    uint8_t* seedbuf = use_gpu ? (uint8_t*)calloc(bytes, 1) : cur;
    if (!strcmp(a.init, "random"))
        gol_seed_random(seedbuf, &topo, a.seed, a.density);
    else
        gol_seed_pattern(seedbuf, &topo, a.global_w, a.global_h, a.init);

#ifdef USE_CUDA
    if (use_gpu) {
        cudaMemcpy(cur, seedbuf, bytes, cudaMemcpyHostToDevice);
        free(seedbuf);
    }
    cudaStream_t s_int = 0, s_bdy = 0;
    int overlap = use_gpu && !strcmp(a.kernel, "overlap");
    int shared  = use_gpu && !strcmp(a.kernel, "shared");
    if (overlap) {
        cudaStreamCreate(&s_int);
        cudaStreamCreate(&s_bdy);
    }
#endif

    // Per-phase timing accumulators.
    double t_halo = 0.0, t_kernel = 0.0, t_io = 0.0;
    uint64_t T0 = clock_now();

    for (int s = 0; s < a.steps; ++s) {
#ifdef USE_CUDA
        if (use_gpu && overlap) {
            uint64_t k0 = clock_now();
            gol_step_gpu_interior(cur, nxt, topo.local_w, topo.local_h, s_int);
            uint64_t h0 = clock_now();
            gol_halo_exchange(cur, &topo);
            uint64_t h1 = clock_now();
            cudaStreamSynchronize(s_int);
            gol_step_gpu_boundary(cur, nxt, topo.local_w, topo.local_h, s_bdy);
            cudaStreamSynchronize(s_bdy);
            uint64_t k1 = clock_now();
            t_halo += clock_seconds(h0, h1);
            t_kernel += clock_seconds(k0, k1) - clock_seconds(h0, h1);
        } else if (use_gpu) {
            uint64_t h0 = clock_now();
            gol_halo_exchange(cur, &topo);
            uint64_t h1 = clock_now();
            if (shared) gol_step_gpu_shared(cur, nxt, topo.local_w, topo.local_h, 0);
            else        gol_step_gpu       (cur, nxt, topo.local_w, topo.local_h, 0);
            cudaError_t kerr = cudaGetLastError();
            cudaError_t serr = cudaStreamSynchronize(0);
            if (s == 0 && topo.rank == 0 && (kerr != cudaSuccess || serr != cudaSuccess)) {
                fprintf(stderr, "[gol] cuda error: launch=%s sync=%s\n",
                        cudaGetErrorString(kerr), cudaGetErrorString(serr));
            }
            uint64_t k1 = clock_now();
            t_halo += clock_seconds(h0, h1);
            t_kernel += clock_seconds(h1, k1);
        } else
#endif
        {
            uint64_t h0 = clock_now();
            gol_halo_exchange(cur, &topo);
            uint64_t h1 = clock_now();
            gol_step_cpu(cur, nxt, topo.local_w, topo.local_h);
            uint64_t k1 = clock_now();
            t_halo += clock_seconds(h0, h1);
            t_kernel += clock_seconds(h1, k1);
        }

        uint8_t* tmp = cur; cur = nxt; nxt = tmp;

        if (a.checkpoint_every > 0 && (s + 1) % a.checkpoint_every == 0) {
#ifdef USE_CUDA
            uint8_t* host_buf = NULL;
            if (use_gpu) {
                host_buf = (uint8_t*)malloc(bytes);
                cudaMemcpy(host_buf, cur, bytes, cudaMemcpyDeviceToHost);
            }
            const uint8_t* src = use_gpu ? host_buf : cur;
#else
            const uint8_t* src = cur;
#endif
            t_io += gol_checkpoint_write(src, &topo, a.global_w, a.global_h,
                                         a.checkpoint_path);
#ifdef USE_CUDA
            if (use_gpu) free(host_buf);
#endif
        }
    }

    uint64_t T1 = clock_now();
    double t_total = clock_seconds(T0, T1);

    // Final dump (host-side gather; do this only on small grids).
    if (a.dump_final) {
#ifdef USE_CUDA
        uint8_t* host_buf = use_gpu ? (uint8_t*)malloc(bytes) : cur;
        if (use_gpu) cudaMemcpy(host_buf, cur, bytes, cudaMemcpyDeviceToHost);
        gol_dump_pbm(host_buf, &topo, a.global_w, a.global_h, a.dump_final);
        if (use_gpu) free(host_buf);
#else
        gol_dump_pbm(cur, &topo, a.global_w, a.global_h, a.dump_final);
#endif
    }

    // CSV timing. Append-only; rank 0 writes; header if file doesn't exist.
    if (a.csv_path && topo.rank == 0) {
        FILE* f = fopen(a.csv_path, "r");
        int new_file = (f == NULL);
        if (f) fclose(f);
        f = fopen(a.csv_path, "a");
        if (f) {
            if (new_file) {
                fprintf(f, "label,ranks,dims_x,dims_y,mode,kernel,"
                           "global_w,global_h,steps,trial,"
                           "t_total,t_halo,t_kernel,t_io\n");
            }
            fprintf(f, "%s,%d,%d,%d,%s,%s,%d,%d,%d,%d,"
                       "%.6f,%.6f,%.6f,%.6f\n",
                    a.label, topo.size, topo.dims[1], topo.dims[0],
                    a.mode, a.kernel,
                    a.global_w, a.global_h, a.steps, a.trial,
                    t_total, t_halo, t_kernel, t_io);
            fclose(f);
        }
    }

    if (topo.rank == 0) {
        fprintf(stderr,
            "[gol] %s ranks=%d dims=%dx%d %dx%d steps=%d -> "
            "total=%.4fs halo=%.4fs kernel=%.4fs io=%.4fs\n",
            a.label, topo.size, topo.dims[1], topo.dims[0],
            a.global_w, a.global_h, a.steps,
            t_total, t_halo, t_kernel, t_io);
    }

#ifdef USE_CUDA
    if (use_gpu) {
        if (overlap) { cudaStreamDestroy(s_int); cudaStreamDestroy(s_bdy); }
        cudaFree(cur); cudaFree(nxt);
    } else
#endif
    {
        free(cur); free(nxt);
    }

    gol_topology_free(&topo);
    MPI_Finalize();
    return 0;
}

