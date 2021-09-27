#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
#include "xof_shake.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

using namespace std;

void Shake::init()
{
    for (int i = 0; i < 25; i++)
    {
        states_[i] = 0;
    }
    pos_ = 0;
}

void Shake::reset()
{
    for (int i = 0; i < 25; i++)
    {
        states_[i] = 0;
    }
    pos_ = 0;
}

uint64_t load64(const uint8_t x[8])
{
    unsigned int i;
    uint64_t r = 0;

    for(i = 0; i < 8; i++)
    {
        r |= (uint64_t)x[i] << 8*i;
    }

    return r;
}

void Shake::absorb_once(const uint8_t *in, size_t inlen)
{
    unsigned int i;
    while (inlen >= RATE_IN_BYTE)
    {
        for (i = 0; i < RATE_IN_BYTE/8; i++)
        {
            states_[i] ^= load64(in+8*i);
        }
        in += RATE_IN_BYTE;
        inlen -= RATE_IN_BYTE;
        KeccakP1600_Permute_24rounds((void*)states_);
    }

    for (i = 0; i < inlen; i++)
    {
        states_[i/8] ^= (uint64_t)in[i] << 8 * (i % 8);
    }
    states_[i/8] ^= 0x1fULL << 8*(i % 8);
    states_[(RATE_IN_BYTE-1)/8] ^= 1ULL << 63;

    pos_ = RATE_IN_BYTE;
}

void Shake::squeeze(uint8_t *out, size_t outlen)
{
    unsigned int i;
    while (outlen) {
        if (pos_ == RATE_IN_BYTE) {
            KeccakP1600_Permute_24rounds((void*)states_);
            pos_ = 0;
        }

        unsigned int squeeze_size = MIN(RATE_IN_BYTE - pos_, outlen);
        KeccakP1600_ExtractBytes((void*)states_, out, pos_, squeeze_size);
        outlen -= squeeze_size;
        out += squeeze_size;
        pos_ += squeeze_size;
    }
}
