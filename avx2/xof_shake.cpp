#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
#include "xof_shake.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

using namespace std;

void Shake::init()
{
    Keccak_HashInitialize_SHAKE256(&hash);
}

void Shake::reset()
{
    Keccak_HashInitialize_SHAKE256(&hash);
}

void Shake::absorb_once(const uint8_t *in, size_t inlen)
{
    Keccak_HashUpdate(&hash, in, inlen * 8);
    Keccak_HashFinal(&hash, NULL);
}

void Shake::squeeze(uint8_t *out, size_t outlen)
{
    Keccak_HashSqueeze(&hash, out, outlen * 8);
}
