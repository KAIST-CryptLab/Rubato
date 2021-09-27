#ifndef __XOF_SHAKE_H__
#define __XOF_SHAKE_H__

#include <stdint.h>
#include <stdlib.h>
extern "C"
{
#include "KeccakP-1600-SnP.h"
}
#include "parms.h"

using namespace std;

class Shake
{
    public:
        void init();
        void reset();
        void absorb_once(const uint8_t *in, size_t inlen);
        void squeeze(uint8_t *out, size_t outlen);

    private:
        alignas(32) uint64_t states_[25];
        unsigned int pos_;
};

#endif