/*
 * CPU compute step for the MPI-only build (mode=cpu). Operates on a
 * halo'd tile of allocated dims (lw+2) x (lh+2). Halos must already
 * be filled by a preceding gol_halo_exchange().
 */

#include <stdint.h>
#include <stddef.h>

void gol_step_cpu(const uint8_t* in, uint8_t* out, int lw, int lh) {
    const size_t aw = (size_t)lw + 2;
    for (int y = 1; y <= lh; ++y) {
        for (int x = 1; x <= lw; ++x) {
            int n =
                in[(y - 1) * aw + (x - 1)] +
                in[(y - 1) * aw + (x    )] +
                in[(y - 1) * aw + (x + 1)] +
                in[(y    ) * aw + (x - 1)] +
                in[(y    ) * aw + (x + 1)] +
                in[(y + 1) * aw + (x - 1)] +
                in[(y + 1) * aw + (x    )] +
                in[(y + 1) * aw + (x + 1)];
            uint8_t c = in[y * aw + x];
            out[y * aw + x] = (uint8_t)((c && (n == 2 || n == 3)) || (!c && n == 3));
        }
    }
}
