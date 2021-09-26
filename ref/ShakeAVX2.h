#ifndef __SHAKE_AVX2_H__
#define __SHAKE_AVX2_H__

#include <stdint.h>
#include <stdlib.h>
extern "C"
{
#include "KeccakP-1600-SnP.h"
}
#include "parms.h"

#define RATE 1088
#define RATE_IN_BYTE 136

using namespace std;

class ShakeAVX2
{
    public:
        ~ShakeAVX2()
        {
            free(states_);
        };

        void init();
        void update(uint64_t nonce, uint64_t counter);
        void squeeze(uint8_t *output, unsigned int length);
        void print_state();

    private:
        uint8_t *states_;
        uint64_t nonce_;
        uint64_t counter_;
        int rounds_;
        int instance_index_;
        int instance_offset_;
};

#endif
