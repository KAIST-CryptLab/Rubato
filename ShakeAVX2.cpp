#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdio.h>
#include "ShakeAVX2.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

using namespace std;

void ShakeAVX2::init()
{
    states_ = static_cast<uint8_t *>(aligned_alloc(32, sizeof(uint8_t) * 200));
    rounds_ = ROUNDS;
}

void ShakeAVX2::update(uint64_t nonce, uint64_t counter)
{
    nonce_ = nonce;
    counter_ = counter;

    KeccakP1600_Initialize((void*)states_);
    instance_offset_ = 0;


    uint8_t data[48];
    memset(data, 0, sizeof(data));
    for (int i = 0; i < 8; i++)
    {
        data[i] = (rounds_ >> (8 * i)) & 0xff;
        data[i + 24] = (nonce >> (8 * i)) & 0xff;
        data[i + 40] = (counter >> (8 * i)) & 0xff;
        data[32] = 0x1f;
    }
    KeccakP1600_AddBytes((void*)states_, data, 48, 0);
    KeccakP1600_AddByte((void*)states_, 0x80, RATE_IN_BYTE);

    // Permute 
    KeccakP1600_Permute_24rounds((void*)states_);
}

void ShakeAVX2::squeeze(uint8_t *output, unsigned int length)
{
    unsigned offset = 0;
    while (length > 0)
    {
        if (instance_offset_ == RATE_IN_BYTE)
        {
            KeccakP1600_Permute_24rounds((void*)states_);
            instance_offset_ = 0;
        }

        unsigned int squeeze_size = MIN(length, RATE_IN_BYTE - instance_offset_);
        KeccakP1600_ExtractBytes((void*)states_, output + offset, instance_offset_, squeeze_size);
        offset += squeeze_size;
        length -= squeeze_size;
        instance_offset_ += squeeze_size;
    }
}

void ShakeAVX2::print_state()
{
    cout << "State: " << flush;
    for (int i = 0; i < 200; i++)
    {
        uint8_t data = 0;
        KeccakP1600_ExtractBytes((void *)states_, &data, i, 1);
        cout << hex << setw(2) << setfill('0') << (unsigned int)data << dec << " " << flush;
    }
    cout << endl;
}
