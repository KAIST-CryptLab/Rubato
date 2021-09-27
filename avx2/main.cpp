#include <iostream>
#include <iomanip>
#include <string>

#include "Rubato.h"
#include "parms.h"
#include "xof_shake.h"
#include "xof_aes.h"

using namespace std;

int main(int argc, char *argv[])
{
	uint32_t key[BLOCKSIZE];
	uint64_t nonce;
	uint64_t counter;
	uint32_t keystream[OUTPUTSIZE];
	uint8_t *arr = (uint8_t *) keystream;

	for (int i = 0; i < BLOCKSIZE; i++)
	{
		key[i] = i + 1;
	}

	nonce = stoi(argv[1]);
	counter = stoi(argv[2]);

	Rubato cipher(key);
	cipher.init(nonce, counter);
	cipher.crypt(keystream);

	cout << setfill('0');
	for (int i = 0; i < sizeof(keystream); i++)
    	cout << hex << setw(2) << (int)arr[i];
	cout << endl;
}
