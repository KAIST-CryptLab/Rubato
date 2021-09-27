#include <iostream>
#include "Rubato.h"

using namespace std;

int main()
{
	uint32_t key[BLOCKSIZE];
	uint64_t nonce = 0x01234566789abcdef;
	uint64_t counter = 0;
	uint32_t coeffs[XOF_ELEMENT_COUNT];
	uint32_t round_keys[XOF_ELEMENT_COUNT];
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

	// Print coeffs
	cout << "Get coeffs" << endl;
	cipher.init(nonce, counter);
	cipher.get_coeffs(coeffs);

	for (int r = 0; r <= ROUNDS; r++)
	{
		cout << "  Round " << r << "  : " << hex << flush;
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			cout << coeffs[r * BLOCKSIZE + i] << " ";
		}
		cout << dec << endl;
	}

	// Print round keys
	cout << "Get round keys" << endl;

	cipher.get_round_keys(round_keys);
	for (int r = 0; r <= ROUNDS; r++)
	{
		cout << "  Round " << r << "  : " << hex << flush;
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			cout << round_keys[r * BLOCKSIZE + i]<< " ";
		}
		cout << dec << endl;
	}

	cout << "Keystream" << hex << endl;
	cipher.crypt(keystream);
	cout << hex;
	for (int i = 0; i < OUTPUTSIZE; i++)
	{
		cout << keystream[i] << " ";
	}
	cout << dec << endl;

	return 0;
}
