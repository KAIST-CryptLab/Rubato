#include <iostream>
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

	cout << "Key: " << flush;
	for (int i = 0; i < BLOCKSIZE; i++)
	{
		cout << hex << key[i] << " ";
	}
	cout << dec << endl;

	Rubato cipher(key);
	cout << "Bench Start" << endl;
    MEASURE(cipher.init(nonce, counter);cipher.crypt(keystream););
    cout << "Cycles Per Iter" << CYCLES_PER_ITER << endl;
	return 0;
}
