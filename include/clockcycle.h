#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H

#include <stdint.h>

#if defined(__powerpc64__) || defined(__PPC64__)
static inline uint64_t clock_now(void) {
    unsigned int tbl, tbu0, tbu1;
    do {
        __asm__ __volatile__("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__("mftb  %0" : "=r"(tbl));
        __asm__ __volatile__("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);
    return (((uint64_t)tbu0) << 32) | tbl;
}
/* POWER9 timebase on AiMOS DCS nodes runs at 512 MHz. */
#define CLOCK_FREQ_HZ 512000000ULL
#else
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include <time.h>
static inline uint64_t clock_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#define CLOCK_FREQ_HZ 1000000000ULL
#endif

static inline double clock_seconds(uint64_t start, uint64_t end) {
    return (double)(end - start) / (double)CLOCK_FREQ_HZ;
}

#endif
