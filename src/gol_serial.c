/*
 * Serial reference implementation of Conway's Game of Life.
 * Used as the ground-truth oracle for both the CUDA kernel and the
 * full MPI+CUDA build.
 *
 *   ./gol_serial <WIDTH> <HEIGHT> <STEPS> [pattern|random] [seed] [out.pbm]
 *
 * Uses toroidal (wrapping) boundary conditions.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clockcycle.h"

#define RANDOM_DENSITY 0.3

static inline uint8_t cell(const uint8_t* grid, int width, int height, int x, int y) {
    int xx = (x + width) % width;
    int yy = (y + height) % height;
    return grid[yy * width + xx];
}

static void step(const uint8_t* in_grid, uint8_t* out_grid, int width, int height) {
    // Loop over each grid square
    int idx = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++idx) {

            // Count # of alive neighbors
            int n = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    n += cell(in_grid, width, height, x + dx, y + dy);
                }
            }

            // Update rule:
            // Each alive cell survives if and only if it has two or three neighbors.
            // Each dead cell with exactly three neighbors becomes alive.
            uint8_t curr = in_grid[idx];
            out_grid[idx] = (uint8_t) ((curr && (n == 2 || n == 3)) || (!curr && n == 3));
        }
    }
}

static void seed_random(uint8_t* grid, int width, int height, uint64_t seed, double density) {
    // xorshift64 - deterministic and platform-independent random numbers
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    uint64_t threshold = (uint64_t) (density * (double) UINT64_MAX);
    for (int i = 0; i < width * height; ++i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        grid[i] = (s < threshold) ? 1 : 0;
    }
}

static void seed_pattern(uint8_t* g, int w, int h, const char* name) {
    int cx = w / 2, cy = h / 2;

    // Default patterns
    if (!strcmp(name, "blinker")) {
        g[(cy) * w + (cx - 1)] = 1;
        g[(cy) * w + (cx)] = 1;
        g[(cy) * w + (cx + 1)] = 1;
    } else if (!strcmp(name, "glider")) {
        int p[5][2] = {{1,0}, {2,1}, {0,2}, {1,2}, {2,2}};
        for (int i = 0; i < 5; ++i) {
            g[(cy + p[i][1]) * w + (cx + p[i][0])] = 1;
        }
    } else if (!strcmp(name, "r-pentomino")) {
        int p[5][2] = {{1,0},{2,0},{0,1},{1,1},{1,2}};
        for (int i = 0; i < 5; ++i) {
            g[(cy + p[i][1]) * w + (cx + p[i][0])] = 1;
        }
    }
}

static void dump_pbm(const uint8_t* grid, int width, int height, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return;
    }

    fprintf(f, "P4\n%d %d\n", width, height);
    int row_bytes = (width + 7) / 8;
    uint8_t* row = (uint8_t*) calloc(row_bytes, 1);

    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_bytes);
        for (int x = 0; x < width; ++x)
            if (grid[y * width + x]) row[x / 8] |= (uint8_t) (0x80 >> (x % 8));
        fwrite(row, 1, row_bytes, f);
    }

    free(row);
    fclose(f);
}

int main(int argc, char** argv) {
    // Command-line args
    if (argc < 4) {
        fprintf(stderr, "usage: %s WIDTH HEIGHT STEPS [pattern|random] [seed] [out.pbm]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int steps = atoi(argv[3]);
    const char* mode = (argc > 4) ? argv[4] : "random";
    uint64_t seed = (argc > 5) ? strtoull(argv[5], NULL, 10) : 1;
    const char* outpath = (argc > 6) ? argv[6] : NULL;

    // Allocate grids as 1-d arrays
    // A point (x,y) in space is located at grid[y * width + x]
    uint8_t* in_grid = (uint8_t*) calloc(width * height, 1);
    uint8_t* out_grid = (uint8_t*) calloc(width * height, 1);
    if (!in_grid || !out_grid) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    // Initialize grid
    if (!strcmp(mode, "random")) {
        seed_random(in_grid, width, height, seed, RANDOM_DENSITY);
    }
    else {
        seed_pattern(in_grid, width, height, mode);
    }

    // Main loop
    uint64_t t0 = clock_now();
    for (int s = 0; s < steps; ++s) {
        step(in_grid, out_grid, width, height);

        uint8_t* tmp = in_grid;
        in_grid = out_grid;
        out_grid = tmp;
    }
    uint64_t t1 = clock_now();

    // Text Output
    int alive = 0;
    for (int i = 0; i < width * height; ++i) {
        alive += in_grid[i];
    }
    fprintf(stderr, "serial %dx%d %d steps: %.4f s, %d alive\n", width, height, steps, clock_seconds(t0, t1), alive);

    // Dump output file
    if (outpath) dump_pbm(in_grid, width, height, outpath);

    // Frees
    free(in_grid);
    free(out_grid);
    
    return EXIT_SUCCESS;
}
