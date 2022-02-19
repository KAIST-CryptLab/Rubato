#ifndef __UTIL_H__
#define __UTIL_H__

#include <sys/random.h>
#include <x86intrin.h>
#include <stdint.h>

static uint64_t NCLOCKS_START;
static uint64_t NCLOCKS_END;
static uint64_t NCLOCKS;
static double CYCLES_PER_ITER;

#if 0
#define CLOCK_START() {NCLOCKS = RDTSC_START();}
#define CLOCK_END() {NCLOCKS = RDTSC_END() - NCLOCKS;}
#else
  #ifndef REPEAT
  #define REPEAT 4096
  #endif

  #ifndef OUTER_REPEAT
  #define OUTER_REPEAT 30
  #endif

  #ifndef WARMUP
  #define WARMUP REPEAT/4
  #endif

  #define MEASURE(x)                                                                  \
  for (int RDTSC_BENCH_ITER = 0; RDTSC_BENCH_ITER < WARMUP; RDTSC_BENCH_ITER++)       \
  {                                                                                   \
      {x};                                                                            \
  }                                                                                   \
  NCLOCKS = UINT64_MAX;                                                               \
  for (int RDTSC_OUTER_ITER = 0; RDTSC_OUTER_ITER < OUTER_REPEAT; RDTSC_OUTER_ITER++) \
  {                                                                                   \
      uint64_t RDTSC_TEMP_CLK = RDTSC_START();                                        \
      for (int RDTSC_BENCH_ITER = 0; RDTSC_BENCH_ITER < REPEAT; RDTSC_BENCH_ITER++)   \
      {                                                                               \
          {x};                                                                        \
      }                                                                               \
      RDTSC_TEMP_CLK = RDTSC_END() - RDTSC_TEMP_CLK;                                  \
      if (RDTSC_TEMP_CLK < NCLOCKS)                                                   \
          NCLOCKS = RDTSC_TEMP_CLK;                                                   \
  }                                                                                   \
  CYCLES_PER_ITER = (double) NCLOCKS / REPEAT;
#endif

uint64_t RDTSC_START();
uint64_t RDTSC_END();

void cbd_eta42_epi32(__m256i* out, uint8_t *buf);

void rand_bytes(void*, size_t);
void p256_x32(__m256i);
void p256_x64(__m256i);

#endif /* UTIL_H */
