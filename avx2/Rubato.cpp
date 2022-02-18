#include <cstring>
#include <cassert>
#include <immintrin.h>
#include "Rubato.h"
#ifndef NDBUG
#include <iostream>
#endif

extern "C" {
#include "util.h"
}

// v -> u
#define TRANSPOSE36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, s)\
{\
    u0 = _mm256_unpacklo_epi32(v0, v1);\
    u1 = _mm256_unpackhi_epi32(v0, v1);\
    u2 = _mm256_unpacklo_epi32(v2, v3);\
    u3 = _mm256_unpackhi_epi32(v2, v3);\
    \
    v0 = _mm256_unpacklo_epi64(u0, u2);\
    v1 = _mm256_unpackhi_epi64(u0, u2);\
    v2 = _mm256_unpacklo_epi64(u1, u3);\
    v3 = _mm256_unpackhi_epi64(u1, u3);\
    \
    s = _mm256_setzero_si256();\
    u0 = _mm256_unpacklo_epi32(v4, v5);\
    u1 = _mm256_unpackhi_epi32(v4, v5);\
    \
    v4 = _mm256_unpacklo_epi64(u0, s);\
    v5 = _mm256_unpackhi_epi64(u0, s);\
    u4 = _mm256_unpacklo_epi64(u1, s);\
    u5 = _mm256_unpackhi_epi64(u1, s);\
    \
    u0 = _mm256_permute2x128_si256(v0, v4, 0x20);\
    u1 = _mm256_permute2x128_si256(v1, v5, 0x20);\
    u2 = _mm256_permute2x128_si256(v2, u4, 0x20);\
    u3 = _mm256_permute2x128_si256(v3, u5, 0x20);\
    u4 = _mm256_permute2x128_si256(v0, v4, 0x31);\
    u5 = _mm256_permute2x128_si256(v1, v5, 0x31);\
}

#define TRANSPOSE64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3)\
{\
    v0 = _mm256_unpacklo_epi32(u0, u1);\
    v1 = _mm256_unpackhi_epi32(u0, u1);\
    v2 = _mm256_unpacklo_epi32(u2, u3);\
    v3 = _mm256_unpackhi_epi32(u2, u3);\
    \
    u0 = _mm256_unpacklo_epi64(v0, v2);\
    u1 = _mm256_unpackhi_epi64(v0, v2);\
    u2 = _mm256_unpacklo_epi64(v1, v3);\
    u3 = _mm256_unpackhi_epi64(v1, v3);\
    \
    v0 = _mm256_unpacklo_epi32(u4, u5);\
    v1 = _mm256_unpackhi_epi32(u4, u5);\
    v2 = _mm256_unpacklo_epi32(u6, u7);\
    v3 = _mm256_unpackhi_epi32(u6, u7);\
    \
    u4 = _mm256_unpacklo_epi64(v0, v2);\
    u5 = _mm256_unpackhi_epi64(v0, v2);\
    u6 = _mm256_unpacklo_epi64(v1, v3);\
    u7 = _mm256_unpackhi_epi64(v1, v3);\
    \
    v0 = _mm256_permute2x128_si256(u0, u4, 0x20);\
    v1 = _mm256_permute2x128_si256(u1, u5, 0x20);\
    u4 = _mm256_permute2x128_si256(u0, u4, 0x31);\
    u5 = _mm256_permute2x128_si256(u1, u5, 0x31);\
    u0 = v0;\
    u1 = v1;\
    \
    v2 = _mm256_permute2x128_si256(u2, u6, 0x20);\
    v3 = _mm256_permute2x128_si256(u3, u7, 0x20);\
    u6 = _mm256_permute2x128_si256(u2, u6, 0x31);\
    u7 = _mm256_permute2x128_si256(u3, u7, 0x31);\
    u2 = v2;\
    u3 = v3;\
}

#define RED36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q)\
{\
    u0 = _mm256_cmpgt_epi32(v0, mod4q);\
    u1 = _mm256_cmpgt_epi32(v1, mod4q);\
    u2 = _mm256_cmpgt_epi32(v2, mod4q);\
    u3 = _mm256_cmpgt_epi32(v3, mod4q);\
    u4 = _mm256_cmpgt_epi32(v4, mod4q);\
    u5 = _mm256_cmpgt_epi32(v5, mod4q);\
    \
    u0 = _mm256_and_si256(mod4q, u0);\
    u1 = _mm256_and_si256(mod4q, u1);\
    u2 = _mm256_and_si256(mod4q, u2);\
    u3 = _mm256_and_si256(mod4q, u3);\
    u4 = _mm256_and_si256(mod4q, u4);\
    u5 = _mm256_and_si256(mod4q, u5);\
    \
    v0 = _mm256_sub_epi32(v0, u0);\
    v1 = _mm256_sub_epi32(v1, u1);\
    v2 = _mm256_sub_epi32(v2, u2);\
    v3 = _mm256_sub_epi32(v3, u3);\
    v4 = _mm256_sub_epi32(v4, u4);\
    v5 = _mm256_sub_epi32(v5, u5);\
    \
    u0 = _mm256_cmpgt_epi32(v0, mod2q);\
    u1 = _mm256_cmpgt_epi32(v1, mod2q);\
    u2 = _mm256_cmpgt_epi32(v2, mod2q);\
    u3 = _mm256_cmpgt_epi32(v3, mod2q);\
    u4 = _mm256_cmpgt_epi32(v4, mod2q);\
    u5 = _mm256_cmpgt_epi32(v5, mod2q);\
    \
    u0 = _mm256_and_si256(mod2q, u0);\
    u1 = _mm256_and_si256(mod2q, u1);\
    u2 = _mm256_and_si256(mod2q, u2);\
    u3 = _mm256_and_si256(mod2q, u3);\
    u4 = _mm256_and_si256(mod2q, u4);\
    u5 = _mm256_and_si256(mod2q, u5);\
    \
    v0 = _mm256_sub_epi32(v0, u0);\
    v1 = _mm256_sub_epi32(v1, u1);\
    v2 = _mm256_sub_epi32(v2, u2);\
    v3 = _mm256_sub_epi32(v3, u3);\
    v4 = _mm256_sub_epi32(v4, u4);\
    v5 = _mm256_sub_epi32(v5, u5);\
    \
    u0 = _mm256_cmpgt_epi32(v0, mod1q);\
    u1 = _mm256_cmpgt_epi32(v1, mod1q);\
    u2 = _mm256_cmpgt_epi32(v2, mod1q);\
    u3 = _mm256_cmpgt_epi32(v3, mod1q);\
    u4 = _mm256_cmpgt_epi32(v4, mod1q);\
    u5 = _mm256_cmpgt_epi32(v5, mod1q);\
    \
    u0 = _mm256_and_si256(mod1q, u0);\
    u1 = _mm256_and_si256(mod1q, u1);\
    u2 = _mm256_and_si256(mod1q, u2);\
    u3 = _mm256_and_si256(mod1q, u3);\
    u4 = _mm256_and_si256(mod1q, u4);\
    u5 = _mm256_and_si256(mod1q, u5);\
    \
    v0 = _mm256_sub_epi32(v0, u0);\
    v1 = _mm256_sub_epi32(v1, u1);\
    v2 = _mm256_sub_epi32(v2, u2);\
    v3 = _mm256_sub_epi32(v3, u3);\
    v4 = _mm256_sub_epi32(v4, u4);\
    v5 = _mm256_sub_epi32(v5, u5);\
}

#define RED64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, mod8q)\
{\
    v0 = _mm256_cmpgt_epi32(u0, mod8q);\
    v1 = _mm256_cmpgt_epi32(u1, mod8q);\
    v2 = _mm256_cmpgt_epi32(u2, mod8q);\
    v3 = _mm256_cmpgt_epi32(u3, mod8q);\
    \
    v0 = _mm256_and_si256(v0, mod8q);\
    v1 = _mm256_and_si256(v1, mod8q);\
    v2 = _mm256_and_si256(v2, mod8q);\
    v3 = _mm256_and_si256(v3, mod8q);\
    \
    u0 = _mm256_sub_epi32(u0, v0);\
    u1 = _mm256_sub_epi32(u1, v1);\
    u2 = _mm256_sub_epi32(u2, v2);\
    u3 = _mm256_sub_epi32(u3, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u4, mod8q);\
    v1 = _mm256_cmpgt_epi32(u5, mod8q);\
    v2 = _mm256_cmpgt_epi32(u6, mod8q);\
    v3 = _mm256_cmpgt_epi32(u7, mod8q);\
    \
    v0 = _mm256_and_si256(v0, mod8q);\
    v1 = _mm256_and_si256(v1, mod8q);\
    v2 = _mm256_and_si256(v2, mod8q);\
    v3 = _mm256_and_si256(v3, mod8q);\
    \
    u4 = _mm256_sub_epi32(u4, v0);\
    u5 = _mm256_sub_epi32(u5, v1);\
    u6 = _mm256_sub_epi32(u6, v2);\
    u7 = _mm256_sub_epi32(u7, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u0, mod4q);\
    v1 = _mm256_cmpgt_epi32(u1, mod4q);\
    v2 = _mm256_cmpgt_epi32(u2, mod4q);\
    v3 = _mm256_cmpgt_epi32(u3, mod4q);\
    \
    v0 = _mm256_and_si256(v0, mod4q);\
    v1 = _mm256_and_si256(v1, mod4q);\
    v2 = _mm256_and_si256(v2, mod4q);\
    v3 = _mm256_and_si256(v3, mod4q);\
    \
    u0 = _mm256_sub_epi32(u0, v0);\
    u1 = _mm256_sub_epi32(u1, v1);\
    u2 = _mm256_sub_epi32(u2, v2);\
    u3 = _mm256_sub_epi32(u3, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u4, mod4q);\
    v1 = _mm256_cmpgt_epi32(u5, mod4q);\
    v2 = _mm256_cmpgt_epi32(u6, mod4q);\
    v3 = _mm256_cmpgt_epi32(u7, mod4q);\
    \
    v0 = _mm256_and_si256(v0, mod4q);\
    v1 = _mm256_and_si256(v1, mod4q);\
    v2 = _mm256_and_si256(v2, mod4q);\
    v3 = _mm256_and_si256(v3, mod4q);\
    \
    u4 = _mm256_sub_epi32(u4, v0);\
    u5 = _mm256_sub_epi32(u5, v1);\
    u6 = _mm256_sub_epi32(u6, v2);\
    u7 = _mm256_sub_epi32(u7, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u0, mod2q);\
    v1 = _mm256_cmpgt_epi32(u1, mod2q);\
    v2 = _mm256_cmpgt_epi32(u2, mod2q);\
    v3 = _mm256_cmpgt_epi32(u3, mod2q);\
    \
    v0 = _mm256_and_si256(v0, mod2q);\
    v1 = _mm256_and_si256(v1, mod2q);\
    v2 = _mm256_and_si256(v2, mod2q);\
    v3 = _mm256_and_si256(v3, mod2q);\
    \
    u0 = _mm256_sub_epi32(u0, v0);\
    u1 = _mm256_sub_epi32(u1, v1);\
    u2 = _mm256_sub_epi32(u2, v2);\
    u3 = _mm256_sub_epi32(u3, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u4, mod2q);\
    v1 = _mm256_cmpgt_epi32(u5, mod2q);\
    v2 = _mm256_cmpgt_epi32(u6, mod2q);\
    v3 = _mm256_cmpgt_epi32(u7, mod2q);\
    \
    v0 = _mm256_and_si256(v0, mod2q);\
    v1 = _mm256_and_si256(v1, mod2q);\
    v2 = _mm256_and_si256(v2, mod2q);\
    v3 = _mm256_and_si256(v3, mod2q);\
    \
    u4 = _mm256_sub_epi32(u4, v0);\
    u5 = _mm256_sub_epi32(u5, v1);\
    u6 = _mm256_sub_epi32(u6, v2);\
    u7 = _mm256_sub_epi32(u7, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u0, mod1q);\
    v1 = _mm256_cmpgt_epi32(u1, mod1q);\
    v2 = _mm256_cmpgt_epi32(u2, mod1q);\
    v3 = _mm256_cmpgt_epi32(u3, mod1q);\
    \
    v0 = _mm256_and_si256(v0, mod1q);\
    v1 = _mm256_and_si256(v1, mod1q);\
    v2 = _mm256_and_si256(v2, mod1q);\
    v3 = _mm256_and_si256(v3, mod1q);\
    \
    u0 = _mm256_sub_epi32(u0, v0);\
    u1 = _mm256_sub_epi32(u1, v1);\
    u2 = _mm256_sub_epi32(u2, v2);\
    u3 = _mm256_sub_epi32(u3, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u4, mod1q);\
    v1 = _mm256_cmpgt_epi32(u5, mod1q);\
    v2 = _mm256_cmpgt_epi32(u6, mod1q);\
    v3 = _mm256_cmpgt_epi32(u7, mod1q);\
    \
    v0 = _mm256_and_si256(v0, mod1q);\
    v1 = _mm256_and_si256(v1, mod1q);\
    v2 = _mm256_and_si256(v2, mod1q);\
    v3 = _mm256_and_si256(v3, mod1q);\
    \
    u4 = _mm256_sub_epi32(u4, v0);\
    u5 = _mm256_sub_epi32(u5, v1);\
    u6 = _mm256_sub_epi32(u6, v2);\
    u7 = _mm256_sub_epi32(u7, v3);\
}

// u -> v
#define MIX36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q, s, bufs)\
{\
    v0 = _mm256_add_epi32(u0, u1);\
    v1 = _mm256_add_epi32(u2, u3);\
    s = _mm256_add_epi32(u4, u5);\
    v0 = _mm256_add_epi32(v0, v1);\
    s = _mm256_add_epi32(s, v0);\
    \
    v0 = _mm256_cmpgt_epi32(s, mod4q);\
    v0 = _mm256_and_si256(mod4q, v0);\
    s = _mm256_sub_epi32(s, v0);\
    \
    v0 = _mm256_cmpgt_epi32(s, mod2q);\
    v0 = _mm256_and_si256(mod2q, v0);\
    s = _mm256_sub_epi32(s, v0);\
    \
    v0 = _mm256_cmpgt_epi32(s, mod1q);\
    v0 = _mm256_and_si256(mod1q, v0);\
    s = _mm256_sub_epi32(s, v0);\
    \
    v0 = _mm256_add_epi32(s, u0);\
    v1 = _mm256_add_epi32(s, u1);\
    v2 = _mm256_add_epi32(s, u2);\
    v3 = _mm256_add_epi32(s, u3);\
    v4 = _mm256_add_epi32(s, u4);\
    v5 = _mm256_add_epi32(s, u5);\
    \
    v0 = _mm256_add_epi32(v0, u1);\
    v1 = _mm256_add_epi32(v1, u2);\
    v2 = _mm256_add_epi32(v2, u3);\
    v3 = _mm256_add_epi32(v3, u4);\
    v4 = _mm256_add_epi32(v4, u5);\
    v5 = _mm256_add_epi32(v5, u0);\
    \
    v0 = _mm256_add_epi32(v0, u2);\
    v1 = _mm256_add_epi32(v1, u3);\
    v2 = _mm256_add_epi32(v2, u4);\
    v3 = _mm256_add_epi32(v3, u5);\
    v4 = _mm256_add_epi32(v4, u0);\
    v5 = _mm256_add_epi32(v5, u1);\
    \
    _mm256_store_si256((__m256i *)bufs, v0);\
    _mm256_store_si256((__m256i *)(bufs+8), v1);\
    _mm256_store_si256((__m256i *)(bufs+16), v2);\
    _mm256_store_si256((__m256i *)(bufs+24), v3);\
    _mm256_store_si256((__m256i *)(bufs+32), v4);\
    _mm256_store_si256((__m256i *)(bufs+40), v5);\
    \
    u0 = _mm256_add_epi32(u0, u0);\
    u1 = _mm256_add_epi32(u1, u1);\
    u2 = _mm256_add_epi32(u2, u2);\
    u3 = _mm256_add_epi32(u3, u3);\
    u4 = _mm256_add_epi32(u4, u4);\
    u5 = _mm256_add_epi32(u5, u5);\
    \
    v0 = _mm256_cmpgt_epi32(u0, mod1q);\
    v1 = _mm256_cmpgt_epi32(u1, mod1q);\
    v2 = _mm256_cmpgt_epi32(u2, mod1q);\
    v3 = _mm256_cmpgt_epi32(u3, mod1q);\
    v4 = _mm256_cmpgt_epi32(u4, mod1q);\
    v5 = _mm256_cmpgt_epi32(u5, mod1q);\
    \
    v0 = _mm256_and_si256(mod1q, v0);\
    v1 = _mm256_and_si256(mod1q, v1);\
    v2 = _mm256_and_si256(mod1q, v2);\
    v3 = _mm256_and_si256(mod1q, v3);\
    v4 = _mm256_and_si256(mod1q, v4);\
    v5 = _mm256_and_si256(mod1q, v5);\
    \
    u0 = _mm256_sub_epi32(u0, v0);\
    u1 = _mm256_sub_epi32(u1, v1);\
    u2 = _mm256_sub_epi32(u2, v2);\
    u3 = _mm256_sub_epi32(u3, v3);\
    u4 = _mm256_sub_epi32(u4, v4);\
    u5 = _mm256_sub_epi32(u5, v5);\
    \
    v0 = _mm256_add_epi32(*(__m256i *)bufs, u0);\
    v1 = _mm256_add_epi32(*(__m256i *)(bufs+8), u1);\
    v2 = _mm256_add_epi32(*(__m256i *)(bufs+16), u2);\
    v3 = _mm256_add_epi32(*(__m256i *)(bufs+24), u3);\
    v4 = _mm256_add_epi32(*(__m256i *)(bufs+32), u4);\
    v5 = _mm256_add_epi32(*(__m256i *)(bufs+40), u5);\
    \
    v0 = _mm256_add_epi32(v0, u2);\
    v1 = _mm256_add_epi32(v1, u3);\
    v2 = _mm256_add_epi32(v2, u4);\
    v3 = _mm256_add_epi32(v3, u5);\
    v4 = _mm256_add_epi32(v4, u0);\
    v5 = _mm256_add_epi32(v5, u1);\
    \
    v0 = _mm256_add_epi32(v0, u3);\
    v1 = _mm256_add_epi32(v1, u4);\
    v2 = _mm256_add_epi32(v2, u5);\
    v3 = _mm256_add_epi32(v3, u0);\
    v4 = _mm256_add_epi32(v4, u1);\
    v5 = _mm256_add_epi32(v5, u2);\
}

#define MIX64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, s, bufs)\
{\
    v0 = _mm256_add_epi32(u0, u1);\
    v1 = _mm256_add_epi32(u2, u3);\
    v2 = _mm256_add_epi32(u4, u5);\
    v3 = _mm256_add_epi32(u6, u7);\
    \
    v0 = _mm256_add_epi32(v0, v1);\
    v2 = _mm256_add_epi32(v2, v3);\
    \
    s = _mm256_add_epi32(v0, v2);\
    v0 = _mm256_cmpgt_epi32(s, mod4q);\
    v0 = _mm256_and_si256(mod4q, v0);\
    s = _mm256_sub_epi32(s, v0);\
    \
    v0 = _mm256_cmpgt_epi32(s, mod2q);\
    v0 = _mm256_and_si256(mod2q, v0);\
    s = _mm256_sub_epi32(s, v0);\
    \
    v0 = _mm256_cmpgt_epi32(s, mod1q);\
    v0 = _mm256_and_si256(mod1q, v0);\
    s = _mm256_sub_epi32(s, v0);\
    \
    v0 = _mm256_add_epi32(u2, u4);\
    v1 = _mm256_add_epi32(u3, u5);\
    v2 = _mm256_add_epi32(u4, u6);\
    v3 = _mm256_add_epi32(u5, u7);\
    \
    v0 = _mm256_add_epi32(v0, s);\
    v1 = _mm256_add_epi32(v1, s);\
    v2 = _mm256_add_epi32(v2, s);\
    v3 = _mm256_add_epi32(v3, s);\
    \
    v0 = _mm256_add_epi32(v0, u5);\
    v1 = _mm256_add_epi32(v1, u6);\
    v2 = _mm256_add_epi32(v2, u7);\
    v3 = _mm256_add_epi32(v3, u0);\
    \
    _mm256_store_si256(bufs, v0);\
    _mm256_store_si256(bufs+1, v1);\
    _mm256_store_si256(bufs+2, v2);\
    _mm256_store_si256(bufs+3, v3);\
    \
    v0 = _mm256_add_epi32(u6, u0);\
    v1 = _mm256_add_epi32(u7, u1);\
    v2 = _mm256_add_epi32(u0, u2);\
    v3 = _mm256_add_epi32(u1, u3);\
    \
    v0 = _mm256_add_epi32(v0, s);\
    v1 = _mm256_add_epi32(v1, s);\
    v2 = _mm256_add_epi32(v2, s);\
    v3 = _mm256_add_epi32(v3, s);\
    \
    v0 = _mm256_add_epi32(v0, u1);\
    v1 = _mm256_add_epi32(v1, u2);\
    v2 = _mm256_add_epi32(v2, u3);\
    v3 = _mm256_add_epi32(v3, u4);\
    \
    _mm256_store_si256(bufs+4, v0);\
    _mm256_store_si256(bufs+5, v1);\
    _mm256_store_si256(bufs+6, v2);\
    _mm256_store_si256(bufs+7, v3);\
    \
    u0 = _mm256_add_epi32(u0, u0);\
    u1 = _mm256_add_epi32(u1, u1);\
    u2 = _mm256_add_epi32(u2, u2);\
    u3 = _mm256_add_epi32(u3, u3);\
    u4 = _mm256_add_epi32(u4, u4);\
    u5 = _mm256_add_epi32(u5, u5);\
    u6 = _mm256_add_epi32(u6, u6);\
    u7 = _mm256_add_epi32(u7, u7);\
    \
    v0 = _mm256_cmpgt_epi32(u0, mod1q);\
    v1 = _mm256_cmpgt_epi32(u1, mod1q);\
    v2 = _mm256_cmpgt_epi32(u2, mod1q);\
    v3 = _mm256_cmpgt_epi32(u3, mod1q);\
    \
    v0 = _mm256_and_si256(v0, mod1q);\
    v1 = _mm256_and_si256(v1, mod1q);\
    v2 = _mm256_and_si256(v2, mod1q);\
    v3 = _mm256_and_si256(v3, mod1q);\
    \
    u0 = _mm256_sub_epi32(u0, v0);\
    u1 = _mm256_sub_epi32(u1, v1);\
    u2 = _mm256_sub_epi32(u2, v2);\
    u3 = _mm256_sub_epi32(u3, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u4, mod1q);\
    v1 = _mm256_cmpgt_epi32(u5, mod1q);\
    v2 = _mm256_cmpgt_epi32(u6, mod1q);\
    v3 = _mm256_cmpgt_epi32(u7, mod1q);\
    \
    v0 = _mm256_and_si256(v0, mod1q);\
    v1 = _mm256_and_si256(v1, mod1q);\
    v2 = _mm256_and_si256(v2, mod1q);\
    v3 = _mm256_and_si256(v3, mod1q);\
    \
    u4 = _mm256_sub_epi32(u4, v0);\
    u5 = _mm256_sub_epi32(u5, v1);\
    u6 = _mm256_sub_epi32(u6, v2);\
    u7 = _mm256_sub_epi32(u7, v3);\
    \
    v0 = _mm256_add_epi32(bufs[0], u1);\
    v1 = _mm256_add_epi32(bufs[1], u2);\
    v2 = _mm256_add_epi32(bufs[2], u3);\
    v3 = _mm256_add_epi32(bufs[3], u4);\
    \
    v0 = _mm256_add_epi32(v0, u2);\
    v1 = _mm256_add_epi32(v1, u3);\
    v2 = _mm256_add_epi32(v2, u4);\
    v3 = _mm256_add_epi32(v3, u5);\
    \
    v0 = _mm256_add_epi32(v0, u3);\
    v1 = _mm256_add_epi32(v1, u4);\
    v2 = _mm256_add_epi32(v2, u5);\
    v3 = _mm256_add_epi32(v3, u6);\
    \
    _mm256_store_si256(bufs+0, v0);\
    _mm256_store_si256(bufs+1, v1);\
    _mm256_store_si256(bufs+2, v2);\
    _mm256_store_si256(bufs+3, v3);\
    \
    v0 = _mm256_add_epi32(bufs[4], u5);\
    v1 = _mm256_add_epi32(bufs[5], u6);\
    v2 = _mm256_add_epi32(bufs[6], u7);\
    v3 = _mm256_add_epi32(bufs[7], u0);\
    \
    v0 = _mm256_add_epi32(v0, u6);\
    v1 = _mm256_add_epi32(v1, u7);\
    v2 = _mm256_add_epi32(v2, u0);\
    v3 = _mm256_add_epi32(v3, u1);\
    \
    v0 = _mm256_add_epi32(v0, u7);\
    v1 = _mm256_add_epi32(v1, u0);\
    v2 = _mm256_add_epi32(v2, u1);\
    v3 = _mm256_add_epi32(v3, u2);\
    \
    _mm256_store_si256(bufs+4, v0);\
    _mm256_store_si256(bufs+5, v1);\
    _mm256_store_si256(bufs+6, v2);\
    _mm256_store_si256(bufs+7, v3);\
    \
    u0 = _mm256_add_epi32(u0, u0);\
    u1 = _mm256_add_epi32(u1, u1);\
    u2 = _mm256_add_epi32(u2, u2);\
    u3 = _mm256_add_epi32(u3, u3);\
    u4 = _mm256_add_epi32(u4, u4);\
    u5 = _mm256_add_epi32(u5, u5);\
    u6 = _mm256_add_epi32(u6, u6);\
    u7 = _mm256_add_epi32(u7, u7);\
    \
    v0 = _mm256_cmpgt_epi32(u0, mod1q);\
    v1 = _mm256_cmpgt_epi32(u1, mod1q);\
    v2 = _mm256_cmpgt_epi32(u2, mod1q);\
    v3 = _mm256_cmpgt_epi32(u3, mod1q);\
    \
    v0 = _mm256_and_si256(v0, mod1q);\
    v1 = _mm256_and_si256(v1, mod1q);\
    v2 = _mm256_and_si256(v2, mod1q);\
    v3 = _mm256_and_si256(v3, mod1q);\
    \
    u0 = _mm256_sub_epi32(u0, v0);\
    u1 = _mm256_sub_epi32(u1, v1);\
    u2 = _mm256_sub_epi32(u2, v2);\
    u3 = _mm256_sub_epi32(u3, v3);\
    \
    v0 = _mm256_cmpgt_epi32(u4, mod1q);\
    v1 = _mm256_cmpgt_epi32(u5, mod1q);\
    v2 = _mm256_cmpgt_epi32(u6, mod1q);\
    v3 = _mm256_cmpgt_epi32(u7, mod1q);\
    \
    v0 = _mm256_and_si256(v0, mod1q);\
    v1 = _mm256_and_si256(v1, mod1q);\
    v2 = _mm256_and_si256(v2, mod1q);\
    v3 = _mm256_and_si256(v3, mod1q);\
    \
    u4 = _mm256_sub_epi32(u4, v0);\
    u5 = _mm256_sub_epi32(u5, v1);\
    u6 = _mm256_sub_epi32(u6, v2);\
    u7 = _mm256_sub_epi32(u7, v3);\
    \
    v0 = _mm256_add_epi32(bufs[0], u0);\
    v1 = _mm256_add_epi32(bufs[1], u1);\
    v2 = _mm256_add_epi32(bufs[2], u2);\
    v3 = _mm256_add_epi32(bufs[3], u3);\
    \
    v0 = _mm256_add_epi32(v0, u4);\
    v1 = _mm256_add_epi32(v1, u5);\
    v2 = _mm256_add_epi32(v2, u6);\
    v3 = _mm256_add_epi32(v3, u7);\
    \
    u4 = _mm256_add_epi32(bufs[4], u4);\
    u5 = _mm256_add_epi32(bufs[5], u5);\
    u6 = _mm256_add_epi32(bufs[6], u6);\
    u7 = _mm256_add_epi32(bufs[7], u7);\
    \
    u4 = _mm256_add_epi32(u0, u4);\
    u5 = _mm256_add_epi32(u1, u5);\
    u6 = _mm256_add_epi32(u2, u6);\
    u7 = _mm256_add_epi32(u3, u7);\
    \
    u0 = v0;\
    u1 = v1;\
    u2 = v2;\
    u3 = v3;\
}

alignas(32) const uint32_t idx[256][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0},
    {2, 0, 0, 0, 0, 0, 0, 0},
    {0, 2, 0, 0, 0, 0, 0, 0},
    {1, 2, 0, 0, 0, 0, 0, 0},
    {0, 1, 2, 0, 0, 0, 0, 0},
    {3, 0, 0, 0, 0, 0, 0, 0},
    {0, 3, 0, 0, 0, 0, 0, 0},
    {1, 3, 0, 0, 0, 0, 0, 0},
    {0, 1, 3, 0, 0, 0, 0, 0},
    {2, 3, 0, 0, 0, 0, 0, 0},
    {0, 2, 3, 0, 0, 0, 0, 0},
    {1, 2, 3, 0, 0, 0, 0, 0},
    {0, 1, 2, 3, 0, 0, 0, 0},
    {4, 0, 0, 0, 0, 0, 0, 0},
    {0, 4, 0, 0, 0, 0, 0, 0},
    {1, 4, 0, 0, 0, 0, 0, 0},
    {0, 1, 4, 0, 0, 0, 0, 0},
    {2, 4, 0, 0, 0, 0, 0, 0},
    {0, 2, 4, 0, 0, 0, 0, 0},
    {1, 2, 4, 0, 0, 0, 0, 0},
    {0, 1, 2, 4, 0, 0, 0, 0},
    {3, 4, 0, 0, 0, 0, 0, 0},
    {0, 3, 4, 0, 0, 0, 0, 0},
    {1, 3, 4, 0, 0, 0, 0, 0},
    {0, 1, 3, 4, 0, 0, 0, 0},
    {2, 3, 4, 0, 0, 0, 0, 0},
    {0, 2, 3, 4, 0, 0, 0, 0},
    {1, 2, 3, 4, 0, 0, 0, 0},
    {0, 1, 2, 3, 4, 0, 0, 0},
    {5, 0, 0, 0, 0, 0, 0, 0},
    {0, 5, 0, 0, 0, 0, 0, 0},
    {1, 5, 0, 0, 0, 0, 0, 0},
    {0, 1, 5, 0, 0, 0, 0, 0},
    {2, 5, 0, 0, 0, 0, 0, 0},
    {0, 2, 5, 0, 0, 0, 0, 0},
    {1, 2, 5, 0, 0, 0, 0, 0},
    {0, 1, 2, 5, 0, 0, 0, 0},
    {3, 5, 0, 0, 0, 0, 0, 0},
    {0, 3, 5, 0, 0, 0, 0, 0},
    {1, 3, 5, 0, 0, 0, 0, 0},
    {0, 1, 3, 5, 0, 0, 0, 0},
    {2, 3, 5, 0, 0, 0, 0, 0},
    {0, 2, 3, 5, 0, 0, 0, 0},
    {1, 2, 3, 5, 0, 0, 0, 0},
    {0, 1, 2, 3, 5, 0, 0, 0},
    {4, 5, 0, 0, 0, 0, 0, 0},
    {0, 4, 5, 0, 0, 0, 0, 0},
    {1, 4, 5, 0, 0, 0, 0, 0},
    {0, 1, 4, 5, 0, 0, 0, 0},
    {2, 4, 5, 0, 0, 0, 0, 0},
    {0, 2, 4, 5, 0, 0, 0, 0},
    {1, 2, 4, 5, 0, 0, 0, 0},
    {0, 1, 2, 4, 5, 0, 0, 0},
    {3, 4, 5, 0, 0, 0, 0, 0},
    {0, 3, 4, 5, 0, 0, 0, 0},
    {1, 3, 4, 5, 0, 0, 0, 0},
    {0, 1, 3, 4, 5, 0, 0, 0},
    {2, 3, 4, 5, 0, 0, 0, 0},
    {0, 2, 3, 4, 5, 0, 0, 0},
    {1, 2, 3, 4, 5, 0, 0, 0},
    {0, 1, 2, 3, 4, 5, 0, 0},
    {6, 0, 0, 0, 0, 0, 0, 0},
    {0, 6, 0, 0, 0, 0, 0, 0},
    {1, 6, 0, 0, 0, 0, 0, 0},
    {0, 1, 6, 0, 0, 0, 0, 0},
    {2, 6, 0, 0, 0, 0, 0, 0},
    {0, 2, 6, 0, 0, 0, 0, 0},
    {1, 2, 6, 0, 0, 0, 0, 0},
    {0, 1, 2, 6, 0, 0, 0, 0},
    {3, 6, 0, 0, 0, 0, 0, 0},
    {0, 3, 6, 0, 0, 0, 0, 0},
    {1, 3, 6, 0, 0, 0, 0, 0},
    {0, 1, 3, 6, 0, 0, 0, 0},
    {2, 3, 6, 0, 0, 0, 0, 0},
    {0, 2, 3, 6, 0, 0, 0, 0},
    {1, 2, 3, 6, 0, 0, 0, 0},
    {0, 1, 2, 3, 6, 0, 0, 0},
    {4, 6, 0, 0, 0, 0, 0, 0},
    {0, 4, 6, 0, 0, 0, 0, 0},
    {1, 4, 6, 0, 0, 0, 0, 0},
    {0, 1, 4, 6, 0, 0, 0, 0},
    {2, 4, 6, 0, 0, 0, 0, 0},
    {0, 2, 4, 6, 0, 0, 0, 0},
    {1, 2, 4, 6, 0, 0, 0, 0},
    {0, 1, 2, 4, 6, 0, 0, 0},
    {3, 4, 6, 0, 0, 0, 0, 0},
    {0, 3, 4, 6, 0, 0, 0, 0},
    {1, 3, 4, 6, 0, 0, 0, 0},
    {0, 1, 3, 4, 6, 0, 0, 0},
    {2, 3, 4, 6, 0, 0, 0, 0},
    {0, 2, 3, 4, 6, 0, 0, 0},
    {1, 2, 3, 4, 6, 0, 0, 0},
    {0, 1, 2, 3, 4, 6, 0, 0},
    {5, 6, 0, 0, 0, 0, 0, 0},
    {0, 5, 6, 0, 0, 0, 0, 0},
    {1, 5, 6, 0, 0, 0, 0, 0},
    {0, 1, 5, 6, 0, 0, 0, 0},
    {2, 5, 6, 0, 0, 0, 0, 0},
    {0, 2, 5, 6, 0, 0, 0, 0},
    {1, 2, 5, 6, 0, 0, 0, 0},
    {0, 1, 2, 5, 6, 0, 0, 0},
    {3, 5, 6, 0, 0, 0, 0, 0},
    {0, 3, 5, 6, 0, 0, 0, 0},
    {1, 3, 5, 6, 0, 0, 0, 0},
    {0, 1, 3, 5, 6, 0, 0, 0},
    {2, 3, 5, 6, 0, 0, 0, 0},
    {0, 2, 3, 5, 6, 0, 0, 0},
    {1, 2, 3, 5, 6, 0, 0, 0},
    {0, 1, 2, 3, 5, 6, 0, 0},
    {4, 5, 6, 0, 0, 0, 0, 0},
    {0, 4, 5, 6, 0, 0, 0, 0},
    {1, 4, 5, 6, 0, 0, 0, 0},
    {0, 1, 4, 5, 6, 0, 0, 0},
    {2, 4, 5, 6, 0, 0, 0, 0},
    {0, 2, 4, 5, 6, 0, 0, 0},
    {1, 2, 4, 5, 6, 0, 0, 0},
    {0, 1, 2, 4, 5, 6, 0, 0},
    {3, 4, 5, 6, 0, 0, 0, 0},
    {0, 3, 4, 5, 6, 0, 0, 0},
    {1, 3, 4, 5, 6, 0, 0, 0},
    {0, 1, 3, 4, 5, 6, 0, 0},
    {2, 3, 4, 5, 6, 0, 0, 0},
    {0, 2, 3, 4, 5, 6, 0, 0},
    {1, 2, 3, 4, 5, 6, 0, 0},
    {0, 1, 2, 3, 4, 5, 6, 0},
    {7, 0, 0, 0, 0, 0, 0, 0},
    {0, 7, 0, 0, 0, 0, 0, 0},
    {1, 7, 0, 0, 0, 0, 0, 0},
    {0, 1, 7, 0, 0, 0, 0, 0},
    {2, 7, 0, 0, 0, 0, 0, 0},
    {0, 2, 7, 0, 0, 0, 0, 0},
    {1, 2, 7, 0, 0, 0, 0, 0},
    {0, 1, 2, 7, 0, 0, 0, 0},
    {3, 7, 0, 0, 0, 0, 0, 0},
    {0, 3, 7, 0, 0, 0, 0, 0},
    {1, 3, 7, 0, 0, 0, 0, 0},
    {0, 1, 3, 7, 0, 0, 0, 0},
    {2, 3, 7, 0, 0, 0, 0, 0},
    {0, 2, 3, 7, 0, 0, 0, 0},
    {1, 2, 3, 7, 0, 0, 0, 0},
    {0, 1, 2, 3, 7, 0, 0, 0},
    {4, 7, 0, 0, 0, 0, 0, 0},
    {0, 4, 7, 0, 0, 0, 0, 0},
    {1, 4, 7, 0, 0, 0, 0, 0},
    {0, 1, 4, 7, 0, 0, 0, 0},
    {2, 4, 7, 0, 0, 0, 0, 0},
    {0, 2, 4, 7, 0, 0, 0, 0},
    {1, 2, 4, 7, 0, 0, 0, 0},
    {0, 1, 2, 4, 7, 0, 0, 0},
    {3, 4, 7, 0, 0, 0, 0, 0},
    {0, 3, 4, 7, 0, 0, 0, 0},
    {1, 3, 4, 7, 0, 0, 0, 0},
    {0, 1, 3, 4, 7, 0, 0, 0},
    {2, 3, 4, 7, 0, 0, 0, 0},
    {0, 2, 3, 4, 7, 0, 0, 0},
    {1, 2, 3, 4, 7, 0, 0, 0},
    {0, 1, 2, 3, 4, 7, 0, 0},
    {5, 7, 0, 0, 0, 0, 0, 0},
    {0, 5, 7, 0, 0, 0, 0, 0},
    {1, 5, 7, 0, 0, 0, 0, 0},
    {0, 1, 5, 7, 0, 0, 0, 0},
    {2, 5, 7, 0, 0, 0, 0, 0},
    {0, 2, 5, 7, 0, 0, 0, 0},
    {1, 2, 5, 7, 0, 0, 0, 0},
    {0, 1, 2, 5, 7, 0, 0, 0},
    {3, 5, 7, 0, 0, 0, 0, 0},
    {0, 3, 5, 7, 0, 0, 0, 0},
    {1, 3, 5, 7, 0, 0, 0, 0},
    {0, 1, 3, 5, 7, 0, 0, 0},
    {2, 3, 5, 7, 0, 0, 0, 0},
    {0, 2, 3, 5, 7, 0, 0, 0},
    {1, 2, 3, 5, 7, 0, 0, 0},
    {0, 1, 2, 3, 5, 7, 0, 0},
    {4, 5, 7, 0, 0, 0, 0, 0},
    {0, 4, 5, 7, 0, 0, 0, 0},
    {1, 4, 5, 7, 0, 0, 0, 0},
    {0, 1, 4, 5, 7, 0, 0, 0},
    {2, 4, 5, 7, 0, 0, 0, 0},
    {0, 2, 4, 5, 7, 0, 0, 0},
    {1, 2, 4, 5, 7, 0, 0, 0},
    {0, 1, 2, 4, 5, 7, 0, 0},
    {3, 4, 5, 7, 0, 0, 0, 0},
    {0, 3, 4, 5, 7, 0, 0, 0},
    {1, 3, 4, 5, 7, 0, 0, 0},
    {0, 1, 3, 4, 5, 7, 0, 0},
    {2, 3, 4, 5, 7, 0, 0, 0},
    {0, 2, 3, 4, 5, 7, 0, 0},
    {1, 2, 3, 4, 5, 7, 0, 0},
    {0, 1, 2, 3, 4, 5, 7, 0},
    {6, 7, 0, 0, 0, 0, 0, 0},
    {0, 6, 7, 0, 0, 0, 0, 0},
    {1, 6, 7, 0, 0, 0, 0, 0},
    {0, 1, 6, 7, 0, 0, 0, 0},
    {2, 6, 7, 0, 0, 0, 0, 0},
    {0, 2, 6, 7, 0, 0, 0, 0},
    {1, 2, 6, 7, 0, 0, 0, 0},
    {0, 1, 2, 6, 7, 0, 0, 0},
    {3, 6, 7, 0, 0, 0, 0, 0},
    {0, 3, 6, 7, 0, 0, 0, 0},
    {1, 3, 6, 7, 0, 0, 0, 0},
    {0, 1, 3, 6, 7, 0, 0, 0},
    {2, 3, 6, 7, 0, 0, 0, 0},
    {0, 2, 3, 6, 7, 0, 0, 0},
    {1, 2, 3, 6, 7, 0, 0, 0},
    {0, 1, 2, 3, 6, 7, 0, 0},
    {4, 6, 7, 0, 0, 0, 0, 0},
    {0, 4, 6, 7, 0, 0, 0, 0},
    {1, 4, 6, 7, 0, 0, 0, 0},
    {0, 1, 4, 6, 7, 0, 0, 0},
    {2, 4, 6, 7, 0, 0, 0, 0},
    {0, 2, 4, 6, 7, 0, 0, 0},
    {1, 2, 4, 6, 7, 0, 0, 0},
    {0, 1, 2, 4, 6, 7, 0, 0},
    {3, 4, 6, 7, 0, 0, 0, 0},
    {0, 3, 4, 6, 7, 0, 0, 0},
    {1, 3, 4, 6, 7, 0, 0, 0},
    {0, 1, 3, 4, 6, 7, 0, 0},
    {2, 3, 4, 6, 7, 0, 0, 0},
    {0, 2, 3, 4, 6, 7, 0, 0},
    {1, 2, 3, 4, 6, 7, 0, 0},
    {0, 1, 2, 3, 4, 6, 7, 0},
    {5, 6, 7, 0, 0, 0, 0, 0},
    {0, 5, 6, 7, 0, 0, 0, 0},
    {1, 5, 6, 7, 0, 0, 0, 0},
    {0, 1, 5, 6, 7, 0, 0, 0},
    {2, 5, 6, 7, 0, 0, 0, 0},
    {0, 2, 5, 6, 7, 0, 0, 0},
    {1, 2, 5, 6, 7, 0, 0, 0},
    {0, 1, 2, 5, 6, 7, 0, 0},
    {3, 5, 6, 7, 0, 0, 0, 0},
    {0, 3, 5, 6, 7, 0, 0, 0},
    {1, 3, 5, 6, 7, 0, 0, 0},
    {0, 1, 3, 5, 6, 7, 0, 0},
    {2, 3, 5, 6, 7, 0, 0, 0},
    {0, 2, 3, 5, 6, 7, 0, 0},
    {1, 2, 3, 5, 6, 7, 0, 0},
    {0, 1, 2, 3, 5, 6, 7, 0},
    {4, 5, 6, 7, 0, 0, 0, 0},
    {0, 4, 5, 6, 7, 0, 0, 0},
    {1, 4, 5, 6, 7, 0, 0, 0},
    {0, 1, 4, 5, 6, 7, 0, 0},
    {2, 4, 5, 6, 7, 0, 0, 0},
    {0, 2, 4, 5, 6, 7, 0, 0},
    {1, 2, 4, 5, 6, 7, 0, 0},
    {0, 1, 2, 4, 5, 6, 7, 0},
    {3, 4, 5, 6, 7, 0, 0, 0},
    {0, 3, 4, 5, 6, 7, 0, 0},
    {1, 3, 4, 5, 6, 7, 0, 0},
    {0, 1, 3, 4, 5, 6, 7, 0},
    {2, 3, 4, 5, 6, 7, 0, 0},
    {0, 2, 3, 4, 5, 6, 7, 0},
    {1, 2, 3, 4, 5, 6, 7, 0},
    {0, 1, 2, 3, 4, 5, 6, 7}
};

// Montgomery reduction from [0 2 4 6] [1 3 5 7] [8 a c e] [9 b d f]
//                        to [0 1 2 3 4 5 6 7]   [8 9 a b c d e f]
inline void mred_b16(__m256i u0, __m256i u1, __m256i u2, __m256i u3, __m256i *r0, __m256i *r1)
{
    const __m256i mod1q = _mm256_set1_epi32(Q);
    const __m256i perm = _mm256_set_epi32(7, 5, 6, 4, 3, 1, 2, 0);
    const __m256i RR = _mm256_set1_epi64x(R2_MOD_Q);
    const __m256i Qbar = _mm256_set1_epi32(Qbar_MOD_R);
    __m256i v0, v1, v2, v3;

    v0 = _mm256_permutevar8x32_epi32(u0, perm); // [0lo, 2lo, 0hi, 2hi, 4lo, 6lo, 4hi, 6hi]
    v1 = _mm256_permutevar8x32_epi32(u1, perm); // [1lo, 3lo, 1hi, 3hi, 5lo, 7lo, 5hi, 7hi]
    v2 = _mm256_permutevar8x32_epi32(u2, perm);
    v3 = _mm256_permutevar8x32_epi32(u3, perm);

    v0 = _mm256_unpacklo_epi32(v0, v1); // [0lo, 1lo, 2lo, 3lo, 4lo, 5lo, 6lo, 7lo]
    v2 = _mm256_unpacklo_epi32(v2, v3);

    v0 = _mm256_mullo_epi32(Qbar, v0); // [0, 1, 2, 3, 4, 5, 6, 7]
    v2 = _mm256_mullo_epi32(Qbar, v2);

    v1 = _mm256_srli_epi64(v0, 32);
    v3 = _mm256_srli_epi64(v2, 32);

    v0 = _mm256_mul_epu32(mod1q, v0); // [0, 2, 4, 6]
    v1 = _mm256_mul_epu32(mod1q, v1); // [1, 3, 5, 7]
    v2 = _mm256_mul_epu32(mod1q, v2);
    v3 = _mm256_mul_epu32(mod1q, v3);

    u0 = _mm256_add_epi64(u0, v0);
    u1 = _mm256_add_epi64(u1, v1);
    u2 = _mm256_add_epi64(u2, v2);
    u3 = _mm256_add_epi64(u3, v3);

    v0 = _mm256_permutevar8x32_epi32(u0, perm); // [0lo, 2lo, 0hi, 2hi, 4lo, 6lo, 4hi, 6hi]
    v1 = _mm256_permutevar8x32_epi32(u1, perm); // [1lo, 3lo, 1hi, 3hi, 5lo, 7lo, 5hi, 7hi]
    v2 = _mm256_permutevar8x32_epi32(u2, perm);
    v3 = _mm256_permutevar8x32_epi32(u3, perm);

    u0 = _mm256_unpackhi_epi32(v0, v1); // [0hi, 1hi, 2hi, 3hi, ...]
    u2 = _mm256_unpackhi_epi32(v2, v3);

    v0 = _mm256_cmpgt_epi32(u0, mod1q);
    v2 = _mm256_cmpgt_epi32(u2, mod1q);

    v0 = _mm256_and_si256(mod1q, v0);
    v2 = _mm256_and_si256(mod1q, v2);

    *r0 = _mm256_sub_epi32(u0, v0);
    *r1 = _mm256_sub_epi32(u2, v2);
}

void Rubato::get_coeffs(uint32_t *output)
{
    memcpy(output, coeffs_, sizeof(uint32_t) * XOF_ELEMENT_COUNT);
}

void Rubato::get_round_keys(uint32_t *output)
{
#if BLOCKSIZE == 16 || BLOCKSIZE == 64
    for (int i = 0; i < XOF_ELEMENT_COUNT; i++)
    {
        output[i] = (uint64_t)round_keys_[i] * RINV_MOD_Q % Q;
    }
#elif BLOCKSIZE == 36
    int cnt = 0;
    for (int r = 0; r <= ROUNDS; r++)
    {
        for (int row = 0; row < 6; row++)
        {
            for (int col = 0; col < 6; col++)
            {
                output[cnt++] = (uint64_t)round_keys_[r * 48 + row * 8 + col] * RINV_MOD_Q % Q;
            }
        }
    }
#else
    abort();
#endif
}

void Rubato::gen_noise_b16()
{
    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i zero = _mm256_setzero_si256();

    xof_noise_->squeeze((uint8_t *)noise_, BLOCKSIZE*4);
    __m256i u0, u1, v0, v1, s0, s1, t0, t1;
    __m256i *cdf_tb = (__m256i *)CDF_TABLE;

    u0 = _mm256_load_si256((__m256i *)noise_);
    u1 = _mm256_load_si256((__m256i *)(noise_+8));

    s0 = _mm256_and_si256(u0, ones);
    s1 = _mm256_and_si256(u1, ones);

    u0 = _mm256_srli_epi32(u0, 1);
    u1 = _mm256_srli_epi32(u1, 1);

    v0 = _mm256_setzero_si256();
    v1 = _mm256_setzero_si256();
    for (int i = 0; i < CDF_TABLE_LEN; i++)
    {
        t0 = _mm256_sub_epi32(cdf_tb[i], u0);
        t1 = _mm256_sub_epi32(cdf_tb[i], u1);
        t0 = _mm256_srli_epi32(t0, 31);
        t1 = _mm256_srli_epi32(t1, 31);
        v0 = _mm256_add_epi32(v0, t0);
        v1 = _mm256_add_epi32(v1, t1);
    }
    t0 = _mm256_sub_epi32(zero, s0);
    t1 = _mm256_sub_epi32(zero, s1);

    v0 = _mm256_xor_si256(v0, t0);
    v1 = _mm256_xor_si256(v1, t1);

    v0 = _mm256_add_epi32(v0, s0);
    v1 = _mm256_add_epi32(v1, s1);

    _mm256_store_si256((__m256i *)noise_, v0);
    _mm256_store_si256((__m256i *)(noise_+8), v1);
}

void Rubato::gen_noise_b36()
{
    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i zero = _mm256_setzero_si256();

    xof_noise_->squeeze((uint8_t *)noise_, BLOCKSIZE*4);
    __m256i u0, u1, u2, v0, v1, v2, s0, s1, s2, t0, t1, t2;
    __m256i *cdf_tb = (__m256i *)CDF_TABLE;

    u0 = _mm256_load_si256((__m256i *)noise_);
    u1 = _mm256_load_si256((__m256i *)(noise_+8));
    u2 = _mm256_load_si256((__m256i *)(noise_+16));

    s0 = _mm256_and_si256(u0, ones);
    s1 = _mm256_and_si256(u1, ones);
    s2 = _mm256_and_si256(u2, ones);

    u0 = _mm256_srli_epi32(u0, 1);
    u1 = _mm256_srli_epi32(u1, 1);
    u2 = _mm256_srli_epi32(u2, 1);

    v0 = _mm256_setzero_si256();
    v1 = _mm256_setzero_si256();
    v2 = _mm256_setzero_si256();
    for (int i = 0; i < CDF_TABLE_LEN; i++)
    {
        t0 = _mm256_sub_epi32(cdf_tb[i], u0);
        t1 = _mm256_sub_epi32(cdf_tb[i], u1);
        t2 = _mm256_sub_epi32(cdf_tb[i], u2);
        t0 = _mm256_srli_epi32(t0, 31);
        t1 = _mm256_srli_epi32(t1, 31);
        t2 = _mm256_srli_epi32(t2, 31);
        v0 = _mm256_add_epi32(v0, t0);
        v1 = _mm256_add_epi32(v1, t1);
        v2 = _mm256_add_epi32(v2, t2);
    }
    t0 = _mm256_sub_epi32(zero, s0);
    t1 = _mm256_sub_epi32(zero, s1);
    t2 = _mm256_sub_epi32(zero, s2);

    v0 = _mm256_xor_si256(v0, t0);
    v1 = _mm256_xor_si256(v1, t1);
    v2 = _mm256_xor_si256(v2, t2);

    v0 = _mm256_add_epi32(v0, s0);
    v1 = _mm256_add_epi32(v1, s1);
    v2 = _mm256_add_epi32(v2, s2);

    _mm256_store_si256((__m256i *)noise_, v0);
    _mm256_store_si256((__m256i *)(noise_+8), v1);
    _mm256_store_si256((__m256i *)(noise_+16), v2);

    u0 = _mm256_load_si256((__m256i *)(noise_+24));
    u1 = _mm256_load_si256((__m256i *)(noise_+32));

    s0 = _mm256_and_si256(u0, ones);
    s1 = _mm256_and_si256(u1, ones);

    u0 = _mm256_srli_epi32(u0, 1);
    u1 = _mm256_srli_epi32(u1, 1);

    v0 = _mm256_setzero_si256();
    v1 = _mm256_setzero_si256();
    for (int i = 0; i < CDF_TABLE_LEN; i++)
    {
        t0 = _mm256_sub_epi32(cdf_tb[i], u0);
        t1 = _mm256_sub_epi32(cdf_tb[i], u1);
        t0 = _mm256_srli_epi32(t0, 31);
        t1 = _mm256_srli_epi32(t1, 31);
        v0 = _mm256_add_epi32(v0, t0);
        v1 = _mm256_add_epi32(v1, t1);
    }
    t0 = _mm256_sub_epi32(zero, s0);
    t1 = _mm256_sub_epi32(zero, s1);

    v0 = _mm256_xor_si256(v0, t0);
    v1 = _mm256_xor_si256(v1, t1);

    v0 = _mm256_add_epi32(v0, s0);
    v1 = _mm256_add_epi32(v1, s1);

    _mm256_store_si256((__m256i *)(noise_+24), v0);
    _mm256_store_si256((__m256i *)(noise_+32), v1);
}

void Rubato::gen_noise_b64()
{
    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i zero = _mm256_setzero_si256();

    xof_noise_->squeeze((uint8_t *)noise_, BLOCKSIZE*4);
    __m256i u0, u1, u2, u3, v0, v1, v2, v3, t0, t1, t2, t3;
    __m256i *cdf_tb = (__m256i *)CDF_TABLE;
    __m256i buf[4];

    u0 = _mm256_load_si256((__m256i *)noise_);
    u1 = _mm256_load_si256((__m256i *)(noise_+8));
    u2 = _mm256_load_si256((__m256i *)(noise_+16));
    u3 = _mm256_load_si256((__m256i *)(noise_+24));

    t0 = _mm256_and_si256(u0, ones);
    t1 = _mm256_and_si256(u1, ones);
    t2 = _mm256_and_si256(u2, ones);
    t3 = _mm256_and_si256(u3, ones);

    _mm256_store_si256(buf+0, t0);
    _mm256_store_si256(buf+1, t1);
    _mm256_store_si256(buf+2, t2);
    _mm256_store_si256(buf+3, t3);

    u0 = _mm256_srli_epi32(u0, 1);
    u1 = _mm256_srli_epi32(u1, 1);
    u2 = _mm256_srli_epi32(u2, 1);
    u3 = _mm256_srli_epi32(u3, 1);

    v0 = _mm256_setzero_si256();
    v1 = _mm256_setzero_si256();
    v2 = _mm256_setzero_si256();
    v3 = _mm256_setzero_si256();

    for (int i = 0; i < CDF_TABLE_LEN; i++)
    {
        t0 = _mm256_sub_epi32(cdf_tb[i], u0);
        t1 = _mm256_sub_epi32(cdf_tb[i], u1);
        t2 = _mm256_sub_epi32(cdf_tb[i], u2);
        t3 = _mm256_sub_epi32(cdf_tb[i], u3);

        t0 = _mm256_srli_epi32(t0, 31);
        t1 = _mm256_srli_epi32(t1, 31);
        t2 = _mm256_srli_epi32(t2, 31);
        t3 = _mm256_srli_epi32(t3, 31);

        v0 = _mm256_add_epi32(v0, t0);
        v1 = _mm256_add_epi32(v1, t1);
        v2 = _mm256_add_epi32(v2, t2);
        v3 = _mm256_add_epi32(v3, t3);
    }
    u0 = _mm256_load_si256(buf+0);
    u1 = _mm256_load_si256(buf+1);
    u2 = _mm256_load_si256(buf+2);
    u3 = _mm256_load_si256(buf+3);

    t0 = _mm256_sub_epi32(zero, u0);
    t1 = _mm256_sub_epi32(zero, u1);
    t2 = _mm256_sub_epi32(zero, u2);
    t3 = _mm256_sub_epi32(zero, u3);

    v0 = _mm256_xor_si256(v0, t0);
    v1 = _mm256_xor_si256(v1, t1);
    v2 = _mm256_xor_si256(v2, t2);
    v3 = _mm256_xor_si256(v3, t3);

    v0 = _mm256_add_epi32(v0, u0);
    v1 = _mm256_add_epi32(v1, u1);
    v2 = _mm256_add_epi32(v2, u2);
    v3 = _mm256_add_epi32(v3, u3);

    _mm256_store_si256((__m256i *)(noise_+0), v0);
    _mm256_store_si256((__m256i *)(noise_+8), v1);
    _mm256_store_si256((__m256i *)(noise_+16), v2);
    _mm256_store_si256((__m256i *)(noise_+24), v3);

    u0 = _mm256_load_si256((__m256i *)(noise_+32));
    u1 = _mm256_load_si256((__m256i *)(noise_+40));
    u2 = _mm256_load_si256((__m256i *)(noise_+48));
    u3 = _mm256_load_si256((__m256i *)(noise_+56));

    t0 = _mm256_and_si256(u0, ones);
    t1 = _mm256_and_si256(u1, ones);
    t2 = _mm256_and_si256(u2, ones);
    t3 = _mm256_and_si256(u3, ones);

    _mm256_store_si256(buf+0, t0);
    _mm256_store_si256(buf+1, t1);
    _mm256_store_si256(buf+2, t2);
    _mm256_store_si256(buf+3, t3);

    u0 = _mm256_srli_epi32(u0, 1);
    u1 = _mm256_srli_epi32(u1, 1);
    u2 = _mm256_srli_epi32(u2, 1);
    u3 = _mm256_srli_epi32(u3, 1);

    v0 = _mm256_setzero_si256();
    v1 = _mm256_setzero_si256();
    v2 = _mm256_setzero_si256();
    v3 = _mm256_setzero_si256();

    for (int i = 0; i < CDF_TABLE_LEN; i++)
    {
        t0 = _mm256_sub_epi32(cdf_tb[i], u0);
        t1 = _mm256_sub_epi32(cdf_tb[i], u1);
        t2 = _mm256_sub_epi32(cdf_tb[i], u2);
        t3 = _mm256_sub_epi32(cdf_tb[i], u3);

        t0 = _mm256_srli_epi32(t0, 31);
        t1 = _mm256_srli_epi32(t1, 31);
        t2 = _mm256_srli_epi32(t2, 31);
        t3 = _mm256_srli_epi32(t3, 31);

        v0 = _mm256_add_epi32(v0, t0);
        v1 = _mm256_add_epi32(v1, t1);
        v2 = _mm256_add_epi32(v2, t2);
        v3 = _mm256_add_epi32(v3, t3);
    }
    u0 = _mm256_load_si256(buf+0);
    u1 = _mm256_load_si256(buf+1);
    u2 = _mm256_load_si256(buf+2);
    u3 = _mm256_load_si256(buf+3);

    t0 = _mm256_sub_epi32(zero, u0);
    t1 = _mm256_sub_epi32(zero, u1);
    t2 = _mm256_sub_epi32(zero, u2);
    t3 = _mm256_sub_epi32(zero, u3);

    v0 = _mm256_xor_si256(v0, t0);
    v1 = _mm256_xor_si256(v1, t1);
    v2 = _mm256_xor_si256(v2, t2);
    v3 = _mm256_xor_si256(v3, t3);

    v0 = _mm256_add_epi32(v0, u0);
    v1 = _mm256_add_epi32(v1, u1);
    v2 = _mm256_add_epi32(v2, u2);
    v3 = _mm256_add_epi32(v3, u3);

    _mm256_store_si256((__m256i *)(noise_+32), v0);
    _mm256_store_si256((__m256i *)(noise_+40), v1);
    _mm256_store_si256((__m256i *)(noise_+48), v2);
    _mm256_store_si256((__m256i *)(noise_+56), v3);
}

void Rubato::init(uint64_t nonce, uint64_t counter)
{
    uint8_t buf[32];
    *(uint64_t *)buf = nonce;
    *(uint64_t *)(buf + 8) = counter;
    *(uint64_t *)(buf + 16) = seed_;
    *(uint64_t *)(buf + 24) = 0;
    xof_coeff_->absorb_once(buf, 16);
    xof_noise_->absorb_once(buf, 32);
    gen_coeffs();
#if BLOCKSIZE == 16
    keyschedule_b16();
    gen_noise_b16();
#elif BLOCKSIZE == 36
    keyschedule_b36();
    gen_noise_b36();
#elif BLOCKSIZE == 64
    keyschedule_b64();
    gen_noise_b64();
#else
    abort();
#endif
}

void Rubato::crypt(float input[OUTPUTSIZE], uint32_t output[OUTPUTSIZE])
{
#if BLOCKSIZE == 16
    crypt_b16(input, output);
#elif BLOCKSIZE == 36
    crypt_b36(input, output);
#elif BLOCKSIZE == 64
    crypt_b64(input, output);
#else
    abort();
#endif
}

void Rubato::gen_coeffs()
{
    // Set random vector from SHAKE256
    unsigned int ctr = 0;

    const __m256i mod1q = _mm256_set1_epi32(Q);
    const __m256i mask = _mm256_set1_epi32(Q_BIT_MASK);
    __m256i u0, u1, u2, u3, v0, v1, v2, v3;

    int g0, g1, g2, g3;

    uint64_t idx0, idx1, idx2, idx3;

    alignas(32) uint8_t buf[32*4] = {0,};

    while (ctr <= XOF_ELEMENT_COUNT - 8*4)
    {
        xof_coeff_->squeeze(buf, 32*4);
        u0 = _mm256_load_si256((__m256i *)buf);
        u1 = _mm256_load_si256((__m256i *)buf + 1);
        u2 = _mm256_load_si256((__m256i *)buf + 2);
        u3 = _mm256_load_si256((__m256i *)buf + 3);

        u0 = _mm256_and_si256(u0, mask);
        u1 = _mm256_and_si256(u1, mask);
        u2 = _mm256_and_si256(u2, mask);
        u3 = _mm256_and_si256(u3, mask);

        v0 = _mm256_cmpgt_epi32(mod1q, u0);
        v1 = _mm256_cmpgt_epi32(mod1q, u1);
        v2 = _mm256_cmpgt_epi32(mod1q, u2);
        v3 = _mm256_cmpgt_epi32(mod1q, u3);

        g0 = _mm256_movemask_ps((__m256)v0);
        g1 = _mm256_movemask_ps((__m256)v1);
        g2 = _mm256_movemask_ps((__m256)v2);
        g3 = _mm256_movemask_ps((__m256)v3);

        u0 = _mm256_permutevar8x32_epi32(u0, *(__m256i *)&idx[g0][0]);
        u1 = _mm256_permutevar8x32_epi32(u1, *(__m256i *)&idx[g1][0]);
        u2 = _mm256_permutevar8x32_epi32(u2, *(__m256i *)&idx[g2][0]);
        u3 = _mm256_permutevar8x32_epi32(u3, *(__m256i *)&idx[g3][0]);

        _mm256_storeu_si256((__m256i *)(coeffs_ + ctr), u0);
        ctr += _mm_popcnt_u32(g0);
        _mm256_storeu_si256((__m256i *)(coeffs_ + ctr), u1);
        ctr += _mm_popcnt_u32(g1);
        _mm256_storeu_si256((__m256i *)(coeffs_ + ctr), u2);
        ctr += _mm_popcnt_u32(g2);
        _mm256_storeu_si256((__m256i *)(coeffs_ + ctr), u3);
        ctr += _mm_popcnt_u32(g3);
    }

    while (ctr < XOF_ELEMENT_COUNT)
    {
        xof_coeff_->squeeze(buf, 32);
        u0 = _mm256_load_si256((__m256i *)buf);
        u0 = _mm256_and_si256(u0, mask);
        v0 = _mm256_cmpgt_epi32(mod1q, u0);
        g0 = _mm256_movemask_ps((__m256)v0);

        u0 = _mm256_permutevar8x32_epi32(u0, *(__m256i *)&idx[g0][0]);
        _mm256_storeu_si256((__m256i *)(coeffs_ + ctr), u0);
        ctr += _mm_popcnt_u32(g0);
    }
}

void Rubato::keyschedule_b16()
{
    __m256i u0, u1, u2, u3, v0, v1, v2, v3;
    // Compute round keys
    // round_keys[i] = (key_[i] * coeffs_[i]) % MODULUS;
    for (int r = 0; r <= ROUNDS; r++)
    {
        u0 = _mm256_load_si256((__m256i *)(coeffs_ + BLOCKSIZE * r));
        u2 = _mm256_load_si256((__m256i *)(coeffs_ + BLOCKSIZE * r + 8));
        v0 = _mm256_load_si256((__m256i *)key_);
        v2 = _mm256_load_si256((__m256i *)(key_ + 8));

        u1 = _mm256_srli_epi64(u0, 32);
        u3 = _mm256_srli_epi64(u2, 32);
        v1 = _mm256_srli_epi64(v0, 32);
        v3 = _mm256_srli_epi64(v2, 32);

        u0 = _mm256_mul_epu32(v0, u0);	// [0, 2, 4, 6]
        u1 = _mm256_mul_epu32(v1, u1);  // [1, 3, 5, 7]
        u2 = _mm256_mul_epu32(v2, u2);
        u3 = _mm256_mul_epu32(v3, u3);

        mred_b16(u0, u1, u2, u3, &u0, &u2);

        _mm256_store_si256((__m256i *)(round_keys_ + BLOCKSIZE * r), u0);
        _mm256_store_si256((__m256i *)(round_keys_ + BLOCKSIZE * r + 8), u2);
    }
}

void Rubato::keyschedule_b36()
{
    __m256i mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
    __m256i u0, u1, u2, u3, v0, v1, v2, v3;
    __m256i* rk_ptr = (__m256i *)round_keys_;
    uint32_t* rv_ptr = coeffs_;

    // Compute round keys
    // round_keys[i] = (key_[i] * coeffs_[i]) % MODULUS;
    for (int r = 0; r <= ROUNDS; r++)
    {
        for (int j = 0; j < 3; j++)
        {
            u0 = _mm256_loadu_si256((__m256i *)rv_ptr);
            u2 = _mm256_loadu_si256((__m256i *)(rv_ptr + 6));

            u0 = _mm256_and_si256(u0, mask);
            u2 = _mm256_and_si256(u2, mask);

            v0 = _mm256_load_si256((__m256i *)(key_ + 16*j));
            v2 = _mm256_load_si256((__m256i *)(key_ + 16*j + 8));

            u1 = _mm256_srli_epi64(u0, 32);
            u3 = _mm256_srli_epi64(u2, 32);
            v1 = _mm256_srli_epi64(v0, 32);
            v3 = _mm256_srli_epi64(v2, 32);

            u0 = _mm256_mul_epu32(v0, u0);	// [0, 2, 4, 6]
            u1 = _mm256_mul_epu32(v1, u1);  // [1, 3, 5, 7]
            u2 = _mm256_mul_epu32(v2, u2);
            u3 = _mm256_mul_epu32(v3, u3);

            mred_b16(u0, u1, u2, u3, &u0, &u2);

            _mm256_store_si256(rk_ptr, u0);
            _mm256_store_si256(rk_ptr+1, u2);

            rv_ptr += 12;
            rk_ptr += 2;
        }
    }
}

void Rubato::keyschedule_b64()
{
    __m256i u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5;
    __m256i* rk_ptr = (__m256i *)round_keys_;
    __m256i* rv_ptr = (__m256i *)coeffs_;

    // Compute round keys
    // round_keys[i] = (key_[i] * coeffs_[i]) % MODULUS;
    for (int r = 0; r <= ROUNDS; r++)
    {
        for (int j = 0; j < 4; j++)
        {
            u0 = _mm256_load_si256(rv_ptr);
            u2 = _mm256_load_si256(rv_ptr+1);

            v0 = _mm256_load_si256((__m256i *)(key_ + 16*j));
            v2 = _mm256_load_si256((__m256i *)(key_ + 16*j + 8));

            u1 = _mm256_srli_epi64(u0, 32);
            u3 = _mm256_srli_epi64(u2, 32);
            v1 = _mm256_srli_epi64(v0, 32);
            v3 = _mm256_srli_epi64(v2, 32);

            u0 = _mm256_mul_epu32(v0, u0);	// [0, 2, 4, 6]
            u1 = _mm256_mul_epu32(v1, u1);  // [1, 3, 5, 7]
            u2 = _mm256_mul_epu32(v2, u2);
            u3 = _mm256_mul_epu32(v3, u3);

            mred_b16(u0, u1, u2, u3, &u0, &u2);

            _mm256_store_si256(rk_ptr, u0);
            _mm256_store_si256(rk_ptr+1, u2);

            rv_ptr += 2;
            rk_ptr += 2;
        }
    }
}

void Rubato::crypt_b16(float input[OUTPUTSIZE], uint32_t output[OUTPUTSIZE])
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i rot32 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
    const __m256i fmask = _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, 0);
    const __m256i lomask = _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);
    const __m256 delta = _mm256_set1_ps((float)Q / 16.0f);
    const __m256 half = _mm256_set1_ps(0.5f);

    // For (Mont) Reduction
    const __m256i mod1q = _mm256_set1_epi32(Q);
    const __m256i mod2q = _mm256_set1_epi32(Q*2);
    const __m256i mod4q = _mm256_set1_epi32(Q*4);

    alignas(32) uint32_t buf[BLOCKSIZE];

    __m256i s0, s1, u0, u1, u2, u3, v0, v1, v2, v3;
    __m256i* rk = (__m256i *)round_keys_;

    s0 = _mm256_load_si256((__m256i *)state_);
    s1 = _mm256_load_si256((__m256i *)(state_ + 8));

    for (int r = 0; r < ROUNDS; r++, rk+=2)
    {
        // ARK
        s0 = _mm256_add_epi32(s0, rk[0]); // [0 1 2 3 4 5 6 7]
        s1 = _mm256_add_epi32(s1, rk[1]); // [8 9 a b c d e f]

        u0 = _mm256_cmpgt_epi32(s0, mod1q);
        u1 = _mm256_cmpgt_epi32(s1, mod1q);

        u0 = _mm256_and_si256(u0, mod1q);
        u1 = _mm256_and_si256(u1, mod1q);

        s0 = _mm256_sub_epi32(s0, u0);
        s1 = _mm256_sub_epi32(s1, u1);

        // MixColumns
        u0 = _mm256_permute2x128_si256(s0, s1, 0x20); // [0 1 2 3 8 9 a b]
        u1 = _mm256_permute2x128_si256(s0, s1, 0x31); // [4 5 6 7 c d e f]
        u2 = _mm256_permute2x128_si256(s0, s1, 0x02); // [8 9 a b 0 1 2 3]
        u3 = _mm256_permute2x128_si256(s0, s1, 0x13); // [c d e f 4 5 6 7]

        s0 = _mm256_add_epi32(u0, u1);
        s1 = _mm256_add_epi32(u2, u3);

        v2 = _mm256_add_epi32(u1, u1); // [2*4     2*5     ...]
        v3 = _mm256_add_epi32(u2, u2); // [2*8     2*9     ...]

        s0 = _mm256_add_epi32(s0, s1); // [0+4+8+c 1+5+9+d 2+6+a+e ...]

        v0 = _mm256_add_epi32(s0, u0); // [2*0+4+8+c 2*1+5+9+d ...]
        v1 = _mm256_add_epi32(s0, u1); // [2*4+8+c+0 2*5+9+d+1 ...]

        v0 = _mm256_add_epi32(v0, v2); // [2*0+3*4+8+c 2*1+3*5+9+d ...] -> [0 1 2 3 8 9 a b]
        v1 = _mm256_add_epi32(v1, v3); // [2*4+3*8+c+0 2*5+3*9+d+1 ...] -> [4 5 6 7 c d e f]

        // Reduction
        u0 = _mm256_cmpgt_epi32(v0, mod4q);
        u1 = _mm256_cmpgt_epi32(v1, mod4q);

        u0 = _mm256_and_si256(mod4q, u0);
        u1 = _mm256_and_si256(mod4q, u1);

        v0 = _mm256_sub_epi32(v0, u0);
        v1 = _mm256_sub_epi32(v1, u1);

        u0 = _mm256_cmpgt_epi32(v0, mod2q);
        u1 = _mm256_cmpgt_epi32(v1, mod2q);

        u0 = _mm256_and_si256(mod2q, u0);
        u1 = _mm256_and_si256(mod2q, u1);

        v0 = _mm256_sub_epi32(v0, u0);
        v1 = _mm256_sub_epi32(v1, u1);

        u0 = _mm256_cmpgt_epi32(v0, mod1q);
        u1 = _mm256_cmpgt_epi32(v1, mod1q);

        u0 = _mm256_and_si256(mod1q, u0);
        u1 = _mm256_and_si256(mod1q, u1);

        v0 = _mm256_sub_epi32(v0, u0); // [0 1 2 3 8 9 a b]
        v1 = _mm256_sub_epi32(v1, u1); // [4 5 6 7 c d e f]

        // MixRows
        v2 = _mm256_permute2x128_si256(v0, v0, 0x01); // [8 9 a b 0 1 2 3]
        v3 = _mm256_permute2x128_si256(v1, v1, 0x01); // [c d e f 4 5 6 7]

        u0 = _mm256_unpacklo_epi32(v0, v1); // [0 4 1 5 ...]
        u1 = _mm256_unpackhi_epi32(v0, v1); // [2 6 3 7 ...]
        u2 = _mm256_unpacklo_epi32(v2, v3); // [8 c 9 d ...]
        u3 = _mm256_unpackhi_epi32(v2, v3); // [a e b f ...]

        v0 = _mm256_unpacklo_epi64(u0, u2); // [0 4 8 c ...]
        v1 = _mm256_unpackhi_epi64(u0, u2); // [1 5 9 d ...]
        v2 = _mm256_unpacklo_epi64(u1, u3); // [2 6 a e ...]
        v3 = _mm256_unpackhi_epi64(u1, u3); // [3 7 b f ...]

        u0 = _mm256_permute2x128_si256(v0, v2, 0x20); // [0 4 8 c 2 6 a e]
        u1 = _mm256_permute2x128_si256(v1, v3, 0x20); // [1 5 9 d 3 7 b f]
        u2 = _mm256_permute2x128_si256(v0, v2, 0x02); // [2 6 a e 0 4 8 c]
        u3 = _mm256_permute2x128_si256(v1, v3, 0x02); // [3 7 b f 1 5 9 d]

        s0 = _mm256_add_epi32(u0, u1);
        s1 = _mm256_add_epi32(u2, u3);

        v2 = _mm256_add_epi32(u1, u1);
        v3 = _mm256_add_epi32(u2, u2);

        s0 = _mm256_add_epi32(s0, s1);

        v0 = _mm256_add_epi32(s0, u0);
        v1 = _mm256_add_epi32(s0, u1);

        v0 = _mm256_add_epi32(v0, v2); // [0 4 8 c 2 6 a e]
        v1 = _mm256_add_epi32(v1, v3); // [1 5 9 d 3 7 b f]

        v2 = _mm256_permute2x128_si256(v0, v0, 0x01); // [2 6 a e 0 4 8 c]
        v3 = _mm256_permute2x128_si256(v1, v1, 0x01); // [3 b 7 f 1 5 9 d]

        u0 = _mm256_unpacklo_epi32(v0, v1);
        u1 = _mm256_unpackhi_epi32(v0, v1);
        u2 = _mm256_unpacklo_epi32(v2, v3);
        u3 = _mm256_unpackhi_epi32(v2, v3);

        v0 = _mm256_unpacklo_epi64(u0, u2);
        v1 = _mm256_unpackhi_epi64(u0, u2);
        v2 = _mm256_unpacklo_epi64(u1, u3);
        v3 = _mm256_unpackhi_epi64(u1, u3);

        s0 = _mm256_permute2x128_si256(v0, v1, 0x20); // [0 1 2 3 4 5 6 7]
        s1 = _mm256_permute2x128_si256(v2, v3, 0x20); // [8 9 a b c d e f]

        // Reduction
        u0 = _mm256_cmpgt_epi32(s0, mod4q);
        u1 = _mm256_cmpgt_epi32(s1, mod4q);

        u0 = _mm256_and_si256(mod4q, u0);
        u1 = _mm256_and_si256(mod4q, u1);

        s0 = _mm256_sub_epi32(s0, u0);
        s1 = _mm256_sub_epi32(s1, u1);

        u0 = _mm256_cmpgt_epi32(s0, mod2q);
        u1 = _mm256_cmpgt_epi32(s1, mod2q);

        u0 = _mm256_and_si256(mod2q, u0);
        u1 = _mm256_and_si256(mod2q, u1);

        s0 = _mm256_sub_epi32(s0, u0);
        s1 = _mm256_sub_epi32(s1, u1);

        u0 = _mm256_cmpgt_epi32(s0, mod1q);
        u1 = _mm256_cmpgt_epi32(s1, mod1q);

        u0 = _mm256_and_si256(mod1q, u0);
        u1 = _mm256_and_si256(mod1q, u1);

        s0 = _mm256_sub_epi32(s0, u0);
        s1 = _mm256_sub_epi32(s1, u1);

        // Fiestel
        u0 = _mm256_permutevar8x32_epi32(s0, rot32); // [7 0 1 2 3 4 5 6]
        u1 = _mm256_permutevar8x32_epi32(s1, rot32); // [f 8 9 a b c d e]


        v0 = _mm256_andnot_si256(fmask, u0); // [7 - - - - - - -]
        u0 = _mm256_and_si256(fmask, u0);    // [- 0 1 2 3 4 5 6]
        v1 = _mm256_and_si256(fmask, u1);    // [- 8 9 a b c d e]

        u2 = _mm256_or_si256(v0, v1); // [7 8 9 a b c d e]

        u1 = _mm256_srli_epi64(u0, 32);
        u3 = _mm256_srli_epi64(u2, 32);

        u0 = _mm256_mul_epu32(u0, u0);
        u1 = _mm256_mul_epu32(u1, u1);
        u2 = _mm256_mul_epu32(u2, u2);
        u3 = _mm256_mul_epu32(u3, u3);

        mred_b16(u0, u1, u2, u3, &u0, &u1);

        s0 = _mm256_add_epi32(s0, u0);
        s1 = _mm256_add_epi32(s1, u1);

        u0 = _mm256_cmpgt_epi32(s0, mod1q);
        u1 = _mm256_cmpgt_epi32(s1, mod1q);

        u0 = _mm256_and_si256(u0, mod1q);
        u1 = _mm256_and_si256(u1, mod1q);

        s0 = _mm256_sub_epi32(s0, u0);
        s1 = _mm256_sub_epi32(s1, u1);
    }

    // Finalization
    // MixColumns
    u0 = _mm256_permute2x128_si256(s0, s1, 0x20); // [0 1 2 3 8 9 a b]
    u1 = _mm256_permute2x128_si256(s0, s1, 0x31); // [4 5 6 7 c d e f]
    u2 = _mm256_permute2x128_si256(s0, s1, 0x02); // [8 9 a b 0 1 2 3]
    u3 = _mm256_permute2x128_si256(s0, s1, 0x13); // [c d e f 4 5 6 7]

    s0 = _mm256_add_epi32(u0, u1);
    s1 = _mm256_add_epi32(u2, u3);

    v2 = _mm256_add_epi32(u1, u1); // [2*4     2*5     ...]
    v3 = _mm256_add_epi32(u2, u2); // [2*8     2*9     ...]

    s0 = _mm256_add_epi32(s0, s1); // [0+4+8+c 1+5+9+d 2+6+a+e ...]

    v0 = _mm256_add_epi32(s0, u0); // [2*0+4+8+c 2*1+5+9+d ...]
    v1 = _mm256_add_epi32(s0, u1); // [2*4+8+c+0 2*5+9+d+1 ...]

    v0 = _mm256_add_epi32(v0, v2); // [2*0+3*4+8+c 2*1+3*5+9+d ...] -> [0 1 2 3 8 9 a b]
    v1 = _mm256_add_epi32(v1, v3); // [2*4+3*8+c+0 2*5+3*9+d+1 ...] -> [4 5 6 7 c d e f]

    // Reduction
    u0 = _mm256_cmpgt_epi32(v0, mod4q);
    u1 = _mm256_cmpgt_epi32(v1, mod4q);

    u0 = _mm256_and_si256(mod4q, u0);
    u1 = _mm256_and_si256(mod4q, u1);

    v0 = _mm256_sub_epi32(v0, u0);
    v1 = _mm256_sub_epi32(v1, u1);

    u0 = _mm256_cmpgt_epi32(v0, mod2q);
    u1 = _mm256_cmpgt_epi32(v1, mod2q);

    u0 = _mm256_and_si256(mod2q, u0);
    u1 = _mm256_and_si256(mod2q, u1);

    v0 = _mm256_sub_epi32(v0, u0);
    v1 = _mm256_sub_epi32(v1, u1);

    u0 = _mm256_cmpgt_epi32(v0, mod1q);
    u1 = _mm256_cmpgt_epi32(v1, mod1q);

    u0 = _mm256_and_si256(mod1q, u0);
    u1 = _mm256_and_si256(mod1q, u1);

    v0 = _mm256_sub_epi32(v0, u0); // [0 1 2 3 8 9 a b]
    v1 = _mm256_sub_epi32(v1, u1); // [4 5 6 7 c d e f]

    // MixRows
    v2 = _mm256_permute2x128_si256(v0, v0, 0x01); // [8 9 a b 0 1 2 3]
    v3 = _mm256_permute2x128_si256(v1, v1, 0x01); // [c d e f 4 5 6 7]

    u0 = _mm256_unpacklo_epi32(v0, v1); // [0 4 1 5 ...]
    u1 = _mm256_unpackhi_epi32(v0, v1); // [2 6 3 7 ...]
    u2 = _mm256_unpacklo_epi32(v2, v3); // [8 c 9 d ...]
    u3 = _mm256_unpackhi_epi32(v2, v3); // [a e b f ...]

    v0 = _mm256_unpacklo_epi64(u0, u2); // [0 4 8 c ...]
    v1 = _mm256_unpackhi_epi64(u0, u2); // [1 5 9 d ...]
    v2 = _mm256_unpacklo_epi64(u1, u3); // [2 6 a e ...]
    v3 = _mm256_unpackhi_epi64(u1, u3); // [3 7 b f ...]

    u0 = _mm256_permute2x128_si256(v0, v2, 0x20); // [0 4 8 c 2 6 a e]
    u1 = _mm256_permute2x128_si256(v1, v3, 0x20); // [1 5 9 d 3 7 b f]
    u2 = _mm256_permute2x128_si256(v0, v2, 0x02); // [2 6 a e 0 4 8 c]
    u3 = _mm256_permute2x128_si256(v1, v3, 0x02); // [3 7 b f 1 5 9 d]

    s0 = _mm256_add_epi32(u0, u1);
    s1 = _mm256_add_epi32(u2, u3);

    v2 = _mm256_add_epi32(u1, u1);
    v3 = _mm256_add_epi32(u2, u2);

    s0 = _mm256_add_epi32(s0, s1);

    v0 = _mm256_add_epi32(s0, u0);
    v1 = _mm256_add_epi32(s0, u1);

    v0 = _mm256_add_epi32(v0, v2); // [0 4 8 c 2 6 a e]
    v1 = _mm256_add_epi32(v1, v3); // [1 5 9 d 3 7 b f]

    v2 = _mm256_permute2x128_si256(v0, v0, 0x01); // [2 6 a e 0 4 8 c]
    v3 = _mm256_permute2x128_si256(v1, v1, 0x01); // [3 b 7 f 1 5 9 d]

    u0 = _mm256_unpacklo_epi32(v0, v1);
    u1 = _mm256_unpackhi_epi32(v0, v1);
    u2 = _mm256_unpacklo_epi32(v2, v3);
    u3 = _mm256_unpackhi_epi32(v2, v3);

    v0 = _mm256_unpacklo_epi64(u0, u2);
    v1 = _mm256_unpackhi_epi64(u0, u2);
    v2 = _mm256_unpacklo_epi64(u1, u3);
    v3 = _mm256_unpackhi_epi64(u1, u3);

    s0 = _mm256_permute2x128_si256(v0, v1, 0x20); // [0 1 2 3 4 5 6 7]
    s1 = _mm256_permute2x128_si256(v2, v3, 0x20); // [8 9 a b c d e f]

    // ARK
    u0 = _mm256_add_epi32(s0, rk[0]);
    u2 = _mm256_add_epi32(s1, rk[1]);

    // Inverse Mont Transform
    u1 = _mm256_srli_epi64(u0, 32);
    u3 = _mm256_srli_epi64(u2, 32);

    u0 = _mm256_and_si256(u0, lomask);
    u1 = _mm256_and_si256(u1, lomask);
    u2 = _mm256_and_si256(u2, lomask);
    u3 = _mm256_and_si256(u3, lomask);

    mred_b16(u0, u1, u2, u3, &s0, &s1);

    s0 = _mm256_add_epi32(s0, *(__m256i *)noise_);
    s1 = _mm256_add_epi32(s1, *(__m256i *)(noise_+8));

    u0 = _mm256_loadu_si256((__m256i *)input);
    u1 = _mm256_loadu_si256((__m256i *)(input+8));

    u0 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(u0)));
    u1 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(u1)));

    u0 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(u0)));
    u1 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(u1)));

    u0 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(u0)));
    u1 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(u1)));

    s0 = _mm256_add_epi32(u0, s0);
    s1 = _mm256_add_epi32(u1, s1);

    // Reduction
    u0 = _mm256_cmpgt_epi32(zero, s0);
    u1 = _mm256_cmpgt_epi32(zero, s1);

    u0 = _mm256_and_si256(mod1q, u0);
    u1 = _mm256_and_si256(mod1q, u1);

    s0 = _mm256_add_epi32(s0, u0);
    s1 = _mm256_add_epi32(s1, u1);

    u0 = _mm256_max_epu32(s0, mod1q);
    u1 = _mm256_max_epu32(s1, mod1q);

    u0 = _mm256_cmpeq_epi32(s0, u0);
    u1 = _mm256_cmpeq_epi32(s1, u1);

    u0 = _mm256_and_si256(mod1q, u0);
    u1 = _mm256_and_si256(mod1q, u1);

    s0 = _mm256_sub_epi32(s0, u0);
    s1 = _mm256_sub_epi32(s1, u1);

    _mm256_store_si256((__m256i *)buf, s0);
    _mm256_store_si256((__m256i *)(buf + 8), s1);

    memcpy(output, buf, OUTPUTSIZE * 4);
}

void Rubato::crypt_b36(float input[OUTPUTSIZE], uint32_t output[BLOCKSIZE])
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i rot32 = _mm256_set_epi32(7, 6, 4, 3, 2, 1, 0, 5);
    const __m256i fmask = _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, 0);
    const __m256i lomask32 = _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);
    const __m256i lomask64 = _mm256_set_epi32(0, 0, -1, -1, 0, 0, -1, -1);
    const __m256 delta = _mm256_set1_ps((float)Q / 16.0f);
    const __m256 half = _mm256_set1_ps(0.5f);


    // For (Mont) Reduction
    const __m256i mod1q = _mm256_set1_epi32(Q);
    const __m256i mod2q = _mm256_set1_epi32(Q*2);
    const __m256i mod4q = _mm256_set1_epi32(Q*4);

    alignas(32) uint32_t bufs[48];
    alignas(32) uint32_t buft[48];

    __m256i u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, s;
    __m256i* rk = (__m256i *)round_keys_;

    u0 = _mm256_load_si256((__m256i *)state_);
    u1 = _mm256_load_si256((__m256i *)(state_ + 8));
    u2 = _mm256_load_si256((__m256i *)(state_ + 16));
    u3 = _mm256_load_si256((__m256i *)(state_ + 24));
    u4 = _mm256_load_si256((__m256i *)(state_ + 32));
    u5 = _mm256_load_si256((__m256i *)(state_ + 40));

    for (int r = 0; r < ROUNDS; r++, rk+=6)
    {
        // ARK
        u0 = _mm256_add_epi32(u0, rk[0]);
        u1 = _mm256_add_epi32(u1, rk[1]);
        u2 = _mm256_add_epi32(u2, rk[2]);
        u3 = _mm256_add_epi32(u3, rk[3]);
        u4 = _mm256_add_epi32(u4, rk[4]);
        u5 = _mm256_add_epi32(u5, rk[5]);

        v0 = _mm256_cmpgt_epi32(u0, mod1q);
        v1 = _mm256_cmpgt_epi32(u1, mod1q);
        v2 = _mm256_cmpgt_epi32(u2, mod1q);
        v3 = _mm256_cmpgt_epi32(u3, mod1q);
        v4 = _mm256_cmpgt_epi32(u4, mod1q);
        v5 = _mm256_cmpgt_epi32(u5, mod1q);

        v0 = _mm256_and_si256(v0, mod1q);
        v1 = _mm256_and_si256(v1, mod1q);
        v2 = _mm256_and_si256(v2, mod1q);
        v3 = _mm256_and_si256(v3, mod1q);
        v4 = _mm256_and_si256(v4, mod1q);
        v5 = _mm256_and_si256(v5, mod1q);

        u0 = _mm256_sub_epi32(u0, v0);
        u1 = _mm256_sub_epi32(u1, v1);
        u2 = _mm256_sub_epi32(u2, v2);
        u3 = _mm256_sub_epi32(u3, v3);
        u4 = _mm256_sub_epi32(u4, v4);
        u5 = _mm256_sub_epi32(u5, v5);

        // MixColumns
        MIX36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q, s, bufs); // u -> v
        RED36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q);
        TRANSPOSE36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, s); // v -> u
        // MixRows
        MIX36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q, s, bufs); // u -> v
        RED36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q); // v -> v
        TRANSPOSE36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, s); // v -> u

        // Fiestel
        _mm256_store_si256((__m256i *)bufs, u0);
        _mm256_store_si256((__m256i *)(bufs+8), u1);
        _mm256_store_si256((__m256i *)(bufs+16), u2);
        _mm256_store_si256((__m256i *)(bufs+24), u3);
        _mm256_store_si256((__m256i *)(bufs+32), u4);
        _mm256_store_si256((__m256i *)(bufs+40), u5);

        v0 = _mm256_permutevar8x32_epi32(u0, rot32); // [5 0 1 2 3 4 - -]
        v1 = _mm256_permutevar8x32_epi32(u1, rot32);
        v2 = _mm256_permutevar8x32_epi32(u2, rot32);
        v3 = _mm256_permutevar8x32_epi32(u3, rot32);
        v4 = _mm256_permutevar8x32_epi32(u4, rot32);
        v5 = _mm256_permutevar8x32_epi32(u5, rot32);

        u0 = _mm256_and_si256(v0, fmask);
        u1 = _mm256_blend_epi32(v0, v1, 0b111110);
        u2 = _mm256_blend_epi32(v1, v2, 0b111110);
        u3 = _mm256_blend_epi32(v2, v3, 0b111110);
        u4 = _mm256_blend_epi32(v3, v4, 0b111110);
        u5 = _mm256_blend_epi32(v4, v5, 0b111110);

        _mm256_store_si256((__m256i *)buft, u2);
        _mm256_store_si256((__m256i *)(buft+8), u3);
        _mm256_store_si256((__m256i *)(buft+16), u4);
        _mm256_store_si256((__m256i *)(buft+24), u5);

        u2 = _mm256_srli_epi64(u0, 32);
        u3 = _mm256_srli_epi64(u1, 32);

        u0 = _mm256_mul_epu32(u0, u0);
        u1 = _mm256_mul_epu32(u1, u1);
        u2 = _mm256_mul_epu32(u2, u2);
        u3 = _mm256_mul_epu32(u3, u3);

        mred_b16(u0, u2, u1, u3, &u0, &u1);
        u0 = _mm256_add_epi32(u0, *(__m256i *)bufs);
        u1 = _mm256_add_epi32(u1, *(__m256i *)(bufs+8));

        u2 = _mm256_load_si256((__m256i *)buft);
        u3 = _mm256_load_si256((__m256i *)(buft+8));
        u4 = _mm256_srli_epi64(u2, 32);
        u5 = _mm256_srli_epi64(u3, 32);

        u2 = _mm256_mul_epu32(u2, u2);
        u3 = _mm256_mul_epu32(u3, u3);
        u4 = _mm256_mul_epu32(u4, u4);
        u5 = _mm256_mul_epu32(u5, u5);

        mred_b16(u2, u4, u3, u5, &u2, &u3);
        u2 = _mm256_add_epi32(u2, *(__m256i *)(bufs+16));
        u3 = _mm256_add_epi32(u3, *(__m256i *)(bufs+24));

        u4 = _mm256_load_si256((__m256i *)(buft+16));
        u5 = _mm256_load_si256((__m256i *)(buft+24));
        v4 = _mm256_srli_epi64(u4, 32);
        v5 = _mm256_srli_epi64(u5, 32);

        u4 = _mm256_mul_epu32(u4, u4);
        u5 = _mm256_mul_epu32(u5, u5);
        v4 = _mm256_mul_epu32(v4, v4);
        v5 = _mm256_mul_epu32(v5, v5);

        mred_b16(u4, v4, u5, v5, &u4, &u5);
        u4 = _mm256_add_epi32(u4, *(__m256i *)(bufs+32));
        u5 = _mm256_add_epi32(u5, *(__m256i *)(bufs+40));

        v0 = _mm256_cmpgt_epi32(u0, mod1q);
        v1 = _mm256_cmpgt_epi32(u1, mod1q);
        v2 = _mm256_cmpgt_epi32(u2, mod1q);
        v3 = _mm256_cmpgt_epi32(u3, mod1q);
        v4 = _mm256_cmpgt_epi32(u4, mod1q);
        v5 = _mm256_cmpgt_epi32(u5, mod1q);

        v0 = _mm256_and_si256(mod1q, v0);
        v1 = _mm256_and_si256(mod1q, v1);
        v2 = _mm256_and_si256(mod1q, v2);
        v3 = _mm256_and_si256(mod1q, v3);
        v4 = _mm256_and_si256(mod1q, v4);
        v5 = _mm256_and_si256(mod1q, v5);

        u0 = _mm256_sub_epi32(u0, v0);
        u1 = _mm256_sub_epi32(u1, v1);
        u2 = _mm256_sub_epi32(u2, v2);
        u3 = _mm256_sub_epi32(u3, v3);
        u4 = _mm256_sub_epi32(u4, v4);
        u5 = _mm256_sub_epi32(u5, v5);
    }

    // Finalization
    // MixColumns
    MIX36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q, s, bufs); // u -> v
    RED36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q);
    TRANSPOSE36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, s); // v -> u

    // MixRows
    MIX36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q, s, bufs); // u -> v
    RED36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, mod1q, mod2q, mod4q); // v -> v
    TRANSPOSE36(u0, u1, u2, u3, u4, u5, v0, v1, v2, v3, v4, v5, s); // v -> u

    // ARK
    u0 = _mm256_add_epi32(u0, rk[0]);
    u1 = _mm256_add_epi32(u1, rk[1]);
    u2 = _mm256_add_epi32(u2, rk[2]);
    u3 = _mm256_add_epi32(u3, rk[3]);
    u4 = _mm256_add_epi32(u4, rk[4]);
    u5 = _mm256_add_epi32(u5, rk[5]);

    // Mont Inverse Transform
    _mm256_store_si256((__m256i *)buft, u2);
    _mm256_store_si256((__m256i *)(buft+8), u3);
    _mm256_store_si256((__m256i *)(buft+16), u4);
    _mm256_store_si256((__m256i *)(buft+24), u5);

    u2 = _mm256_srli_epi64(u0, 32);
    u3 = _mm256_srli_epi64(u1, 32);

    u0 = _mm256_and_si256(u0, lomask32);
    u1 = _mm256_and_si256(u1, lomask32);
    u2 = _mm256_and_si256(u2, lomask32);
    u3 = _mm256_and_si256(u3, lomask32);

    mred_b16(u0, u2, u1, u3, &u0, &u1);
    v0 = _mm256_loadu_si256((__m256i *)(noise_+0));
    v1 = _mm256_loadu_si256((__m256i *)(noise_+6));
    u0 = _mm256_add_epi32(v0, u0);
    u1 = _mm256_add_epi32(v1, u1);

    v0 = _mm256_loadu_si256((__m256i *)input);
    v1 = _mm256_loadu_si256((__m256i *)(input+6));

    v0 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v1)));

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_cmpgt_epi32(zero, u0);
    v1 = _mm256_cmpgt_epi32(zero, u1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_max_epu32(u0, mod1q);
    v1 = _mm256_max_epu32(u1, mod1q);

    v0 = _mm256_cmpeq_epi32(u0, v0);
    v1 = _mm256_cmpeq_epi32(u1, v1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_sub_epi32(u0, v0);
    u1 = _mm256_sub_epi32(u1, v1);

    _mm256_storeu_si256((__m256i *)bufs, u0);
    _mm256_storeu_si256((__m256i *)(bufs+6), u1);

    u0 = _mm256_load_si256((__m256i *)buft);
    u1 = _mm256_load_si256((__m256i *)(buft+8));

    u2 = _mm256_srli_epi64(u0, 32);
    u3 = _mm256_srli_epi64(u1, 32);

    u0 = _mm256_and_si256(u0, lomask32);
    u1 = _mm256_and_si256(u1, lomask32);
    u2 = _mm256_and_si256(u2, lomask32);
    u3 = _mm256_and_si256(u3, lomask32);

    mred_b16(u0, u2, u1, u3, &u0, &u1);
    v0 = _mm256_loadu_si256((__m256i *)(noise_+12));
    v1 = _mm256_loadu_si256((__m256i *)(noise_+18));
    u0 = _mm256_add_epi32(v0, u0);
    u1 = _mm256_add_epi32(v1, u1);

    v0 = _mm256_loadu_si256((__m256i *)(input+12));
    v1 = _mm256_loadu_si256((__m256i *)(input+18));

    v0 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v1)));

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_cmpgt_epi32(zero, u0);
    v1 = _mm256_cmpgt_epi32(zero, u1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_max_epu32(u0, mod1q);
    v1 = _mm256_max_epu32(u1, mod1q);

    v0 = _mm256_cmpeq_epi32(u0, v0);
    v1 = _mm256_cmpeq_epi32(u1, v1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_sub_epi32(u0, v0);
    u1 = _mm256_sub_epi32(u1, v1);

    _mm256_storeu_si256((__m256i *)(bufs+12), u0);
    _mm256_storeu_si256((__m256i *)(bufs+18), u1);

    u0 = _mm256_load_si256((__m256i *)(buft+16));
    u1 = _mm256_load_si256((__m256i *)(buft+24));

    u2 = _mm256_srli_epi64(u0, 32);
    u3 = _mm256_srli_epi64(u1, 32);

    u0 = _mm256_and_si256(u0, lomask32);
    u1 = _mm256_and_si256(u1, lomask32);
    u2 = _mm256_and_si256(u2, lomask32);
    u3 = _mm256_and_si256(u3, lomask32);

    mred_b16(u0, u2, u1, u3, &u0, &u1);
    v0 = _mm256_loadu_si256((__m256i *)(noise_+24));
    v1 = _mm256_loadu_si256((__m256i *)(noise_+30));
    u0 = _mm256_add_epi32(v0, u0);
    u1 = _mm256_add_epi32(v1, u1);

    v0 = _mm256_loadu_si256((__m256i *)(input+24));
    v1 = _mm256_loadu_si256((__m256i *)(input+30));

    v0 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v1)));

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_cmpgt_epi32(zero, u0);
    v1 = _mm256_cmpgt_epi32(zero, u1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_max_epu32(u0, mod1q);
    v1 = _mm256_max_epu32(u1, mod1q);

    v0 = _mm256_cmpeq_epi32(u0, v0);
    v1 = _mm256_cmpeq_epi32(u1, v1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_sub_epi32(u0, v0);
    u1 = _mm256_sub_epi32(u1, v1);

    _mm256_storeu_si256((__m256i *)(bufs+24), u0);
    _mm256_storeu_si256((__m256i *)(bufs+30), u1);

    memcpy(output, bufs, OUTPUTSIZE * 4);
}

void Rubato::crypt_b64(float input[OUTPUTSIZE], uint32_t output[BLOCKSIZE])
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i rot32 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
    const __m256i fmask = _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, 0);
    const __m256i lomask32 = _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);
    const __m256i lomask64 = _mm256_set_epi32(0, 0, -1, -1, 0, 0, -1, -1);
    const __m256 delta = _mm256_set1_ps((float)Q / 16.0f);
    const __m256 half = _mm256_set1_ps(0.5f);

    // For (Mont) Reduction
    const __m256i mod1q = _mm256_set1_epi32(Q);
    const __m256i mod2q = _mm256_set1_epi32(Q*2);
    const __m256i mod4q = _mm256_set1_epi32(Q*4);
    const __m256i mod8q = _mm256_set1_epi32(Q*8);

    __m256i bufs[8];
    __m256i buft[8];

    __m256i u0, u1, u2, u3, u4, u5, u6, u7,v0, v1, v2, v3, s;
    __m256i* rk = (__m256i *)round_keys_;

    u0 = _mm256_load_si256((__m256i *)state_);
    u1 = _mm256_load_si256((__m256i *)(state_ + 8));
    u2 = _mm256_load_si256((__m256i *)(state_ + 16));
    u3 = _mm256_load_si256((__m256i *)(state_ + 24));
    u4 = _mm256_load_si256((__m256i *)(state_ + 32));
    u5 = _mm256_load_si256((__m256i *)(state_ + 40));
    u6 = _mm256_load_si256((__m256i *)(state_ + 48));
    u7 = _mm256_load_si256((__m256i *)(state_ + 56));

    for (int r = 0; r < ROUNDS; r++, rk+=8)
    {
        // ARK
        u0 = _mm256_add_epi32(u0, rk[0]);
        u1 = _mm256_add_epi32(u1, rk[1]);
        u2 = _mm256_add_epi32(u2, rk[2]);
        u3 = _mm256_add_epi32(u3, rk[3]);
        u4 = _mm256_add_epi32(u4, rk[4]);
        u5 = _mm256_add_epi32(u5, rk[5]);
        u6 = _mm256_add_epi32(u6, rk[6]);
        u7 = _mm256_add_epi32(u7, rk[7]);

        v0 = _mm256_cmpgt_epi32(u0, mod1q);
        v1 = _mm256_cmpgt_epi32(u1, mod1q);
        v2 = _mm256_cmpgt_epi32(u2, mod1q);
        v3 = _mm256_cmpgt_epi32(u3, mod1q);

        v0 = _mm256_and_si256(v0, mod1q);
        v1 = _mm256_and_si256(v1, mod1q);
        v2 = _mm256_and_si256(v2, mod1q);
        v3 = _mm256_and_si256(v3, mod1q);

        u0 = _mm256_sub_epi32(u0, v0);
        u1 = _mm256_sub_epi32(u1, v1);
        u2 = _mm256_sub_epi32(u2, v2);
        u3 = _mm256_sub_epi32(u3, v3);

        v0 = _mm256_cmpgt_epi32(u4, mod1q);
        v1 = _mm256_cmpgt_epi32(u5, mod1q);
        v2 = _mm256_cmpgt_epi32(u6, mod1q);
        v3 = _mm256_cmpgt_epi32(u7, mod1q);

        v0 = _mm256_and_si256(v0, mod1q);
        v1 = _mm256_and_si256(v1, mod1q);
        v2 = _mm256_and_si256(v2, mod1q);
        v3 = _mm256_and_si256(v3, mod1q);

        u4 = _mm256_sub_epi32(u4, v0);
        u5 = _mm256_sub_epi32(u5, v1);
        u6 = _mm256_sub_epi32(u6, v2);
        u7 = _mm256_sub_epi32(u7, v3);

        // MixColumns
        MIX64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, s, bufs);
        RED64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, mod8q);
        TRANSPOSE64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3);

        // MixRows
        MIX64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, s, bufs);
        RED64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, mod8q);
        TRANSPOSE64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3);

        _mm256_store_si256(bufs, u0);
        _mm256_store_si256((bufs+1), u1);
        _mm256_store_si256((bufs+2), u2);
        _mm256_store_si256((bufs+3), u3);
        _mm256_store_si256((bufs+4), u4);
        _mm256_store_si256((bufs+5), u5);
        _mm256_store_si256((bufs+6), u6);
        _mm256_store_si256((bufs+7), u7);

        u0 = _mm256_permutevar8x32_epi32(u0, rot32);
        u1 = _mm256_permutevar8x32_epi32(u1, rot32);
        u2 = _mm256_permutevar8x32_epi32(u2, rot32);
        u3 = _mm256_permutevar8x32_epi32(u3, rot32);
        u4 = _mm256_permutevar8x32_epi32(u4, rot32);
        u5 = _mm256_permutevar8x32_epi32(u5, rot32);
        u6 = _mm256_permutevar8x32_epi32(u6, rot32);
        u7 = _mm256_permutevar8x32_epi32(u7, rot32);

        v0 = _mm256_and_si256(u0, fmask);
        v1 = _mm256_blend_epi32(u0, u1, 0b11111110);
        v2 = _mm256_blend_epi32(u1, u2, 0b11111110);
        v3 = _mm256_blend_epi32(u2, u3, 0b11111110);
        _mm256_store_si256(buft, v0);
        _mm256_store_si256(buft+1, v1);
        _mm256_store_si256(buft+2, v2);
        _mm256_store_si256(buft+3, v3);

        v0 = _mm256_blend_epi32(u3, u4, 0b11111110);
        v1 = _mm256_blend_epi32(u4, u5, 0b11111110);
        v2 = _mm256_blend_epi32(u5, u6, 0b11111110);
        v3 = _mm256_blend_epi32(u6, u7, 0b11111110);
        _mm256_store_si256(buft+4, v0);
        _mm256_store_si256(buft+5, v1);
        _mm256_store_si256(buft+6, v2);
        _mm256_store_si256(buft+7, v3);

        u0 = _mm256_load_si256(buft);
        u1 = _mm256_load_si256(buft+1);
        u2 = _mm256_srli_epi64(u0, 32);
        u3 = _mm256_srli_epi64(u1, 32);

        u0 = _mm256_mul_epu32(u0, u0);
        u1 = _mm256_mul_epu32(u1, u1);
        u2 = _mm256_mul_epu32(u2, u2);
        u3 = _mm256_mul_epu32(u3, u3);

        mred_b16(u0, u2, u1, u3, &u0, &u1);
        u0 = _mm256_add_epi32(u0, bufs[0]);
        u1 = _mm256_add_epi32(u1, bufs[1]);

        u2 = _mm256_load_si256(buft+2);
        u3 = _mm256_load_si256(buft+3);
        u4 = _mm256_srli_epi64(u2, 32);
        u5 = _mm256_srli_epi64(u3, 32);

        u2 = _mm256_mul_epu32(u2, u2);
        u3 = _mm256_mul_epu32(u3, u3);
        u4 = _mm256_mul_epu32(u4, u4);
        u5 = _mm256_mul_epu32(u5, u5);

        mred_b16(u2, u4, u3, u5, &u2, &u3);
        u2 = _mm256_add_epi32(u2, bufs[2]);
        u3 = _mm256_add_epi32(u3, bufs[3]);

        _mm256_store_si256(bufs, u0);
        _mm256_store_si256(bufs+1, u1);
        _mm256_store_si256(bufs+2, u2);
        _mm256_store_si256(bufs+3, u3);

        u0 = _mm256_load_si256(buft+4);
        u1 = _mm256_load_si256(buft+5);
        u2 = _mm256_srli_epi64(u0, 32);
        u3 = _mm256_srli_epi64(u1, 32);

        u0 = _mm256_mul_epu32(u0, u0);
        u1 = _mm256_mul_epu32(u1, u1);
        u2 = _mm256_mul_epu32(u2, u2);
        u3 = _mm256_mul_epu32(u3, u3);

        mred_b16(u0, u2, u1, u3, &u0, &u1);
        u0 = _mm256_add_epi32(u0, bufs[4]);
        u1 = _mm256_add_epi32(u1, bufs[5]);

        u2 = _mm256_load_si256(buft+6);
        u3 = _mm256_load_si256(buft+7);
        u4 = _mm256_srli_epi64(u2, 32);
        u5 = _mm256_srli_epi64(u3, 32);

        u2 = _mm256_mul_epu32(u2, u2);
        u3 = _mm256_mul_epu32(u3, u3);
        u4 = _mm256_mul_epu32(u4, u4);
        u5 = _mm256_mul_epu32(u5, u5);

        mred_b16(u2, u4, u3, u5, &u2, &u3);
        u2 = _mm256_add_epi32(u2, bufs[6]);
        u3 = _mm256_add_epi32(u3, bufs[7]);

        u4 = u0;
        u5 = u1;
        u6 = u2;
        u7 = u3;

        u0 = _mm256_load_si256(bufs);
        u1 = _mm256_load_si256(bufs+1);
        u2 = _mm256_load_si256(bufs+2);
        u3 = _mm256_load_si256(bufs+3);

        v0 = _mm256_cmpgt_epi32(u0, mod1q);
        v1 = _mm256_cmpgt_epi32(u1, mod1q);
        v2 = _mm256_cmpgt_epi32(u2, mod1q);
        v3 = _mm256_cmpgt_epi32(u3, mod1q);

        v0 = _mm256_and_si256(v0, mod1q);
        v1 = _mm256_and_si256(v1, mod1q);
        v2 = _mm256_and_si256(v2, mod1q);
        v3 = _mm256_and_si256(v3, mod1q);

        u0 = _mm256_sub_epi32(u0, v0);
        u1 = _mm256_sub_epi32(u1, v1);
        u2 = _mm256_sub_epi32(u2, v2);
        u3 = _mm256_sub_epi32(u3, v3);

        v0 = _mm256_cmpgt_epi32(u4, mod1q);
        v1 = _mm256_cmpgt_epi32(u5, mod1q);
        v2 = _mm256_cmpgt_epi32(u6, mod1q);
        v3 = _mm256_cmpgt_epi32(u7, mod1q);

        v0 = _mm256_and_si256(v0, mod1q);
        v1 = _mm256_and_si256(v1, mod1q);
        v2 = _mm256_and_si256(v2, mod1q);
        v3 = _mm256_and_si256(v3, mod1q);

        u4 = _mm256_sub_epi32(u4, v0);
        u5 = _mm256_sub_epi32(u5, v1);
        u6 = _mm256_sub_epi32(u6, v2);
        u7 = _mm256_sub_epi32(u7, v3);
    }

    // MixColumns
    MIX64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, s, bufs);
    RED64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, mod8q);
    TRANSPOSE64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3);

    // MixRows
    MIX64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, s, bufs);
    RED64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, mod1q, mod2q, mod4q, mod8q);
    TRANSPOSE64(u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3);

    // ARK
    u0 = _mm256_add_epi32(u0, rk[0]);
    u1 = _mm256_add_epi32(u1, rk[1]);
    u2 = _mm256_add_epi32(u2, rk[2]);
    u3 = _mm256_add_epi32(u3, rk[3]);
    u4 = _mm256_add_epi32(u4, rk[4]);
    u5 = _mm256_add_epi32(u5, rk[5]);
    u6 = _mm256_add_epi32(u6, rk[6]);
    u7 = _mm256_add_epi32(u7, rk[7]);

    // Mont Inverse Transform
    _mm256_store_si256(bufs+2, u2);
    _mm256_store_si256(bufs+3, u3);
    _mm256_store_si256(bufs+4, u4);
    _mm256_store_si256(bufs+5, u5);
    _mm256_store_si256(bufs+6, u6);
    _mm256_store_si256(bufs+7, u7);

    u2 = _mm256_srli_epi64(u0, 32);
    u3 = _mm256_srli_epi64(u1, 32);

    u0 = _mm256_and_si256(u0, lomask32);
    u1 = _mm256_and_si256(u1, lomask32);
    u2 = _mm256_and_si256(u2, lomask32);
    u3 = _mm256_and_si256(u3, lomask32);

    mred_b16(u0, u2, u1, u3, &u0, &u1);
    u0 = _mm256_add_epi32(u0, *(__m256i *)(noise_+0));
    u1 = _mm256_add_epi32(u1, *(__m256i *)(noise_+8));

    v0 = _mm256_loadu_si256((__m256i *)(input+0));
    v1 = _mm256_loadu_si256((__m256i *)(input+8));

    v0 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v1)));

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_cmpgt_epi32(zero, u0);
    v1 = _mm256_cmpgt_epi32(zero, u1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_max_epu32(u0, mod1q);
    v1 = _mm256_max_epu32(u1, mod1q);

    v0 = _mm256_cmpeq_epi32(u0, v0);
    v1 = _mm256_cmpeq_epi32(u1, v1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_sub_epi32(u0, v0);
    u1 = _mm256_sub_epi32(u1, v1);

    _mm256_store_si256(bufs, u0);
    _mm256_store_si256(bufs+1, u1);

    u0 = _mm256_load_si256(bufs+2);
    u1 = _mm256_load_si256(bufs+3);

    u2 = _mm256_srli_epi64(u0, 32);
    u3 = _mm256_srli_epi64(u1, 32);

    u0 = _mm256_and_si256(u0, lomask32);
    u1 = _mm256_and_si256(u1, lomask32);
    u2 = _mm256_and_si256(u2, lomask32);
    u3 = _mm256_and_si256(u3, lomask32);

    mred_b16(u0, u2, u1, u3, &u0, &u1);
    u0 = _mm256_add_epi32(u0, *(__m256i *)(noise_+16));
    u1 = _mm256_add_epi32(u1, *(__m256i *)(noise_+24));

    v0 = _mm256_loadu_si256((__m256i *)(input+16));
    v1 = _mm256_loadu_si256((__m256i *)(input+24));

    v0 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v1)));

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_cmpgt_epi32(zero, u0);
    v1 = _mm256_cmpgt_epi32(zero, u1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_max_epu32(u0, mod1q);
    v1 = _mm256_max_epu32(u1, mod1q);

    v0 = _mm256_cmpeq_epi32(u0, v0);
    v1 = _mm256_cmpeq_epi32(u1, v1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_sub_epi32(u0, v0);
    u1 = _mm256_sub_epi32(u1, v1);

    _mm256_store_si256(bufs+2, u0);
    _mm256_store_si256(bufs+3, u1);

    u0 = _mm256_load_si256(bufs+4);
    u1 = _mm256_load_si256(bufs+5);

    u2 = _mm256_srli_epi64(u0, 32);
    u3 = _mm256_srli_epi64(u1, 32);

    u0 = _mm256_and_si256(u0, lomask32);
    u1 = _mm256_and_si256(u1, lomask32);
    u2 = _mm256_and_si256(u2, lomask32);
    u3 = _mm256_and_si256(u3, lomask32);

    mred_b16(u0, u2, u1, u3, &u0, &u1);
    u0 = _mm256_add_epi32(u0, *(__m256i *)(noise_+32));
    u1 = _mm256_add_epi32(u1, *(__m256i *)(noise_+40));

    v0 = _mm256_loadu_si256((__m256i *)(input+32));
    v1 = _mm256_loadu_si256((__m256i *)(input+40));

    v0 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v1)));

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_cmpgt_epi32(zero, u0);
    v1 = _mm256_cmpgt_epi32(zero, u1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_max_epu32(u0, mod1q);
    v1 = _mm256_max_epu32(u1, mod1q);

    v0 = _mm256_cmpeq_epi32(u0, v0);
    v1 = _mm256_cmpeq_epi32(u1, v1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_sub_epi32(u0, v0);
    u1 = _mm256_sub_epi32(u1, v1);

    _mm256_store_si256(bufs+4, u0);
    _mm256_store_si256(bufs+5, u1);

    u0 = _mm256_load_si256(bufs+6);
    u1 = _mm256_load_si256(bufs+7);

    u2 = _mm256_srli_epi64(u0, 32);
    u3 = _mm256_srli_epi64(u1, 32);

    u0 = _mm256_and_si256(u0, lomask32);
    u1 = _mm256_and_si256(u1, lomask32);
    u2 = _mm256_and_si256(u2, lomask32);
    u3 = _mm256_and_si256(u3, lomask32);

    mred_b16(u0, u2, u1, u3, &u0, &u1);
    u0 = _mm256_add_epi32(u0, *(__m256i *)(noise_+48));
    u1 = _mm256_add_epi32(u1, *(__m256i *)(noise_+56));

    v0 = _mm256_loadu_si256((__m256i *)(input+48));
    v1 = _mm256_loadu_si256((__m256i *)(input+56));

    v0 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_mul_ps(delta, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_add_ps(half, _mm256_castsi256_ps(v1)));

    v0 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v0)));
    v1 = _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v1)));

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_cmpgt_epi32(zero, u0);
    v1 = _mm256_cmpgt_epi32(zero, u1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_add_epi32(u0, v0);
    u1 = _mm256_add_epi32(u1, v1);

    v0 = _mm256_max_epu32(u0, mod1q);
    v1 = _mm256_max_epu32(u1, mod1q);

    v0 = _mm256_cmpeq_epi32(u0, v0);
    v1 = _mm256_cmpeq_epi32(u1, v1);

    v0 = _mm256_and_si256(mod1q, v0);
    v1 = _mm256_and_si256(mod1q, v1);

    u0 = _mm256_sub_epi32(u0, v0);
    u1 = _mm256_sub_epi32(u1, v1);

    _mm256_store_si256(bufs+6, u0);
    _mm256_store_si256(bufs+7, u1);

    memcpy(output, bufs, OUTPUTSIZE * 4);
}
