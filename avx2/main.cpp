#include <iostream>
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
	uint32_t coeffs[XOF_ELEMENT_COUNT];
	uint32_t keystream[OUTPUTSIZE];

	for (int i = 0; i < BLOCKSIZE; i++)
	{
		key[i] = i + 1;
	}

	nonce = stoi(argv[1]);
	counter = stoi(argv[2]);

	Rubato cipher(key);
	cipher.init(nonce, counter);
	cipher.get_coeffs(coeffs);
	for (int i = 0; i < XOF_ELEMENT_COUNT; i++)
		cout << coeffs[i] << " ";
	cout << endl;

	cipher.crypt(keystream);
	for (int i = 0; i < OUTPUTSIZE; i++)
		cout << coeffs[i] << " ";
	cout << endl;
}
