#include "util.h"

#include <openssl/aes.h>
#include <x86intrin.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

static unsigned char rxbuf[256];
static size_t rxcnt = 0;

uint64_t RDTSC_START()
{
    unsigned cyc_high, cyc_low;
    __asm volatile("" ::: /* pretend to clobber */ "memory");
    __asm volatile(
            "cpuid\n\t"
            "rdtsc\n\t"
            "mov %%edx, %0\n\t"
            "mov %%eax, %1\n\t"
            : "=r"(cyc_high), "=r"(cyc_low)::"%rax", "%rbx", "%rcx", "%rdx");
    return ((uint64_t)cyc_high << 32) | cyc_low;
}

uint64_t RDTSC_END()
{
    unsigned cyc_high, cyc_low;
    __asm volatile(
            "rdtscp\n\t"
            "mov %%edx, %0\n\t"
            "mov %%eax, %1\n\t"
            "cpuid\n\t"
            : "=r"(cyc_high), "=r"(cyc_low)::"%rax", "%rbx", "%rcx", "%rdx");
    return ((uint64_t)cyc_high << 32) | cyc_low;
}

void rand_bytes(void* buf, size_t sz)
{
    size_t idx = 0;
    size_t cnt = 0;
    while (sz > 0)
    {
        if (rxcnt == 0)
        {
            if (getrandom(rxbuf, 256, 0) != 256)
            {
                puts("Invalid random");
                exit(-1);
            }

            rxcnt = 256;
        }

        cnt = sz < rxcnt ? sz : rxcnt;

        memcpy((uint8_t *)buf + idx, rxbuf + (256 - rxcnt), cnt);
        sz -= cnt;
        rxcnt -= cnt;
        idx += cnt;
    }
}

inline void cbd_eta42_epi32(__m256i* out, uint8_t *buf)
{
	const __m256i mask = _mm256_set1_epi32(0x1fffff);
	const __m256i masklo2 = _mm256_set1_epi32(0x55555555);
	const __m256i masklo4 = _mm256_set1_epi32(0x33333333);
	const __m256i masklo8 = _mm256_set1_epi32(0x0f0f0f0f);
	const __m256i masklo16 = _mm256_set1_epi32(0x00ff00ff);
	const __m256i masklo32 = _mm256_set1_epi32(0x0000ffff);

	__m256i u0, u1, v0, v1;

	u0 = _mm256_and_si256(mask, *(__m256i *)buf);
	u1 = _mm256_and_si256(mask, *(__m256i *)(buf + 32));

	v0 = _mm256_srli_epi32(u0, 1);
	v1 = _mm256_srli_epi32(u1, 1);
	u0 = _mm256_and_si256(u0, masklo2);
	u1 = _mm256_and_si256(u1, masklo2);
	v0 = _mm256_and_si256(v0, masklo2);
	v1 = _mm256_and_si256(v1, masklo2);
	u0 = _mm256_add_epi32(u0, v0);
	u1 = _mm256_add_epi32(u1, v1);

	v0 = _mm256_srli_epi32(u0, 2);
	v1 = _mm256_srli_epi32(u1, 2);
	u0 = _mm256_and_si256(u0, masklo4);
	u1 = _mm256_and_si256(u1, masklo4);
	v0 = _mm256_and_si256(v0, masklo4);
	v1 = _mm256_and_si256(v1, masklo4);
	u0 = _mm256_add_epi32(u0, v0);
	u1 = _mm256_add_epi32(u1, v1);

	v0 = _mm256_srli_epi32(u0, 4);
	v1 = _mm256_srli_epi32(u1, 4);
	u0 = _mm256_and_si256(u0, masklo8);
	u1 = _mm256_and_si256(u1, masklo8);
	v0 = _mm256_and_si256(v0, masklo8);
	v1 = _mm256_and_si256(v1, masklo8);
	u0 = _mm256_add_epi32(u0, v0);
	u1 = _mm256_add_epi32(u1, v1);

	v0 = _mm256_srli_epi32(u0, 8);
	v1 = _mm256_srli_epi32(u1, 8);
	u0 = _mm256_and_si256(u0, masklo16);
	u1 = _mm256_and_si256(u1, masklo16);
	v0 = _mm256_and_si256(v0, masklo16);
	v1 = _mm256_and_si256(v1, masklo16);
	u0 = _mm256_add_epi32(u0, v0);
	u1 = _mm256_add_epi32(u1, v1);

	v0 = _mm256_srli_epi32(u0, 16);
	v1 = _mm256_srli_epi32(u1, 16);
	u0 = _mm256_and_si256(u0, masklo32);
	u1 = _mm256_and_si256(u1, masklo32);
	v0 = _mm256_and_si256(v0, masklo32);
	v1 = _mm256_and_si256(v1, masklo32);
	u0 = _mm256_add_epi32(u0, v0);
	u1 = _mm256_add_epi32(u1, v1);

	*out = _mm256_sub_epi32(u0, u1);
}

void p256_x32(__m256i in) {
    __attribute__((aligned(32))) uint32_t v[8];
    _mm256_store_si256((__m256i*)v, in);
    printf("[%08x %08x %08x %08x %08x %08x %08x %08x]\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

void p256_x64(__m256i in) {
    __attribute__((aligned(32))) uint32_t v[8];
    _mm256_store_si256((__m256i*)v, in);
    printf("[%08x%08x %08x%08x %08x%08x %08x%08x]\n", v[1], v[0], v[3], v[2], v[5], v[4], v[7], v[6]);
}
