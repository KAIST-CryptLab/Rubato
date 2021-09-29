#include <iostream>
#include "parms.h"
#include "Rubato.h"
extern "C" {
#include "util.h"
}

using namespace std;

int main()
{
    uint32_t key[BLOCKSIZE];
    uint64_t nonce = 0x01234566789abcdef;
    uint64_t counter = 0;
    uint32_t keystream[OUTPUTSIZE];
    for (int i = 0; i < BLOCKSIZE; i++)
    {
        key[i] = i + 1;
    }
    Rubato cipher(key);
    MEASURE(cipher.init(nonce, counter);cipher.crypt(keystream););

    switch (PARAM_ID) {
        case 0:
            cout << "Rubato_80S_";
            break;
        case 1:
            cout << "Rubato_80M_";
            break;
        case 2:
            cout << "Rubato_80L_";
            break;
        case 3:
            cout << "Rubato_128S_";
            break;
        case 4:
            cout << "Rubato_128M_";
            break;
        case 5:
            cout << "Rubato_128L_";
            break;
    }

    switch (XOF_TYPE) {
        case 0:
            cout << "AES : ";
            break;
        case 1:
            cout << "SHAKE128 : ";
            break;
        case 2:
            cout << "SHAKE256 : ";
            break;
    }

    cout << CYCLES_PER_ITER << " (CPI)" << endl;
    return 0;
}
