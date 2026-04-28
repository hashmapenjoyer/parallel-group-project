/*
 * Serial reference implementation of Conway's Game of Life (B3/S23).
 * Used as the ground-truth oracle for both the CUDA kernel and the
 * full MPI+CUDA build. Toroidal boundaries.
 *
 *   ./gol_serial <W> <H> <STEPS> [pattern|random] [seed] [out.pbm]
 *
 * The grid is stored as row-major uint8_t (1 = alive, 0 = dead) with
 * NO halo padding. Wraparound is computed via modulo. Performance is
 * not the goal here -- correctness is.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clockcycle.h"

static inline uint8_t cell(const uint8_t* g, int w, int h, int x, int y) {
    int xx = (x + w) % w;
    int yy = (y + h) % h;
    return g[yy * w + xx];
}

static void step(const uint8_t* in, uint8_t* out, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int n = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    n += cell(in, w, h, x + dx, y + dy);
                }
            }
            uint8_t c = in[y * w + x];
            out[y * w + x] = (uint8_t)((c && (n == 2 || n == 3)) || (!c && n == 3));
        }
    }
}

static void seed_random(uint8_t* g, int w, int h, uint64_t seed, double density) {
    /* xorshift64 -- deterministic and platform-independent. */
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    uint64_t threshold = (uint64_t)(density * (double)UINT64_MAX);
    for (int i = 0; i < w * h; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        g[i] = (s < threshold) ? 1 : 0;
    }
}

static void seed_pattern(uint8_t* g, int w, int h, const char* name) {
    int cx = w / 2, cy = h / 2;
    if (!strcmp(name, "blinker")) {
        g[(cy) * w + (cx - 1)] = 1;
        g[(cy) * w + (cx)]     = 1;
        g[(cy) * w + (cx + 1)] = 1;
    } else if (!strcmp(name, "glider")) {
        int p[5][2] = {{1,0},{2,1},{0,2},{1,2},{2,2}};
        for (int i = 0; i < 5; ++i)
            g[(cy + p[i][1]) * w + (cx + p[i][0])] = 1;
    } else if (!strcmp(name, "r-pentomino")) {
        int p[5][2] = {{1,0},{2,0},{0,1},{1,1},{1,2}};
        for (int i = 0; i < 5; ++i)
            g[(cy + p[i][1]) * w + (cx + p[i][0])] = 1;
    }
}

static void dump_pbm(const uint8_t* g, int w, int h, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    fprintf(f, "P4\n%d %d\n", w, h);
    int row_bytes = (w + 7) / 8;
    uint8_t* row = (uint8_t*)calloc(row_bytes, 1);
    for (int y = 0; y < h; ++y) {
        memset(row, 0, row_bytes);
        for (int x = 0; x < w; ++x)
            if (g[y * w + x]) row[x / 8] |= (uint8_t)(0x80 >> (x % 8));
        fwrite(row, 1, row_bytes, f);
    }
    free(row);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s W H STEPS [pattern|random] [seed] [out.pbm]\n",
                argv[0]);
        return 1;
    }
    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    int steps = atoi(argv[3]);
    const char* mode = (argc > 4) ? argv[4] : "random";
    uint64_t seed = (argc > 5) ? strtoull(argv[5], NULL, 10) : 1;
    const char* outpath = (argc > 6) ? argv[6] : NULL;

    uint8_t* a = (uint8_t*)calloc((size_t)w * h, 1);
    uint8_t* b = (uint8_t*)calloc((size_t)w * h, 1);
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }

    if (!strcmp(mode, "random")) seed_random(a, w, h, seed, 0.3);
    else seed_pattern(a, w, h, mode);

    uint64_t t0 = clock_now();
    for (int s = 0; s < steps; ++s) {
        step(a, b, w, h);
        uint8_t* tmp = a; a = b; b = tmp;
    }
    uint64_t t1 = clock_now();

    int alive = 0;
    for (int i = 0; i < w * h; ++i) alive += a[i];
    fprintf(stderr, "serial %dx%d %d steps: %.4f s, %d alive\n",
            w, h, steps, clock_seconds(t0, t1), alive);

    if (outpath) dump_pbm(a, w, h, outpath);

    free(a); free(b);
    return 0;
}
