/*
 * Compare two raw uint8_t grids of identical dimensions and report
 * the first mismatch (with coordinates) plus a total mismatch count.
 *
 *   ./diff_grids W H file_a file_b
 *
 * Used to verify the parallel build against the serial oracle: run the
 * MPI binary with --checkpoint <path> and the serial binary with the
 * same dims, then point this tool at the two files.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s W H file_a file_b\n", argv[0]);
        return 1;
    }
    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    size_t n = (size_t)w * h;

    FILE* fa = fopen(argv[3], "rb");
    FILE* fb = fopen(argv[4], "rb");
    if (!fa || !fb) { perror("open"); return 1; }

    uint8_t* a = (uint8_t*)malloc(n);
    uint8_t* b = (uint8_t*)malloc(n);
    if (fread(a, 1, n, fa) != n || fread(b, 1, n, fb) != n) {
        fprintf(stderr, "short read\n");
        return 1;
    }
    fclose(fa); fclose(fb);

    size_t mismatches = 0;
    size_t first = (size_t)-1;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (first == (size_t)-1) first = i;
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        printf("OK: %dx%d grids identical\n", w, h);
        return 0;
    }
    int fx = (int)(first % (size_t)w);
    int fy = (int)(first / (size_t)w);
    printf("MISMATCH: %zu / %zu cells differ; first at (%d, %d): a=%d b=%d\n",
           mismatches, n, fx, fy, a[first], b[first]);
    free(a); free(b);
    return 2;
}
