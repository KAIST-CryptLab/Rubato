#include <iostream>
#include "Rubato.h"

using namespace std;

int main()
{
	uint64_t key[BLOCKSIZE];
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

	uint64_t nonce = 0x121132412345;
	uint64_t counter = 0;

	Rubato cipher(key);
	cout << "Rubato init" << endl;
	cipher.init(nonce, counter);

	// Check validity of rand vector
	cout << "Check rand vectors" << endl;
	uint64_t buf[XOF_ELEMENT_COUNT];
	for (size_t i = 0; i < (1ULL << 16); i++)
	{
		cipher.update(nonce, i);
		cipher.get_rand_vectors(buf);

		for (int j = 0; j < XOF_ELEMENT_COUNT; j++)
		{
			if (buf[j] >= MODULUS)
			{
				cout << "Counter_" << i << "[" << j << "]: " << buf[j] << endl;
				return -1;
			}
		}
	}

	// Print random vectors
	cout << "Get random vectors" << endl;
	cipher.update(nonce, counter);
	uint64_t rand_vectors[XOF_ELEMENT_COUNT];
	cipher.get_rand_vectors(rand_vectors);

	for (int r = 0; r < ROUNDS; r++)
	{
		cout << "  Round " << r << "  : " << hex << flush;
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			cout << rand_vectors[r * BLOCKSIZE + i] << " ";
		}
		cout << dec << endl;
	}
	cout << "  Final    : " << hex << flush;
	for (int i = 0; i < BLOCKSIZE; i++)
	{
		cout << rand_vectors[ROUNDS * BLOCKSIZE + i] << " ";
	}
	cout << dec << endl;

	// Print round keys
	cout << "Get round keys" << endl;
	uint64_t round_keys[XOF_ELEMENT_COUNT];
	cipher.get_round_keys(round_keys);

	for (int r = 0; r < ROUNDS; r++)
	{
		cout << "  Round " << r << "  : " << hex << flush;
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			cout << round_keys[r * BLOCKSIZE + i] << " ";
		}
		cout << dec << endl;
	}
	cout << "  Final    : " << hex << flush;
	for (int i = 0; i < BLOCKSIZE; i++)
	{
		cout << round_keys[ROUNDS * BLOCKSIZE + i] << " ";
	}
	cout << dec << endl;

	// Print encryption result
	cout << "Keystream" << hex << endl;
	block_t keystream_naive = block_init(BLOCKSIZE);
	cipher.compute_keystream_naive(keystream_naive);
	cout << hex;
	for (int i = 0; i < BLOCKSIZE; i++)
	{
		cout << keystream_naive[i] << " ";
	}
	cout << dec << endl;

	return 0;
}
