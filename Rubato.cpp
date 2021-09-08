#include "Rubato.h"
#include <cstring>
#include <cassert>
#ifndef NDBUG
#include <iostream>
#endif

block_t block_init(size_t sz)
{
    block_t block = static_cast<uint64_t*>
        (aligned_alloc(32, sizeof(uint64_t) * sz));
    memset(block, 0, sizeof(uint64_t) * sz);
    return block;
}

// Rubato public functions
void Rubato::set_key(uint64_t *key)
{
	for (int i = 0; i < BLOCKSIZE; i++)
	{
		key_[i] = key[i] % MODULUS;
	}
}

void Rubato::init(uint64_t nonce, uint64_t counter)
{
	shake_ = new ShakeAVX2();
	shake_->init();
	is_shake_init_ = true;
	keyschedule();
}

void Rubato::update(uint64_t nonce, uint64_t counter)
{
	shake_->update(nonce, counter);
	keyschedule();
}

void Rubato::get_rand_vectors(uint64_t *output)
{
	memcpy(output, rand_vectors_, sizeof(uint64_t) * XOF_ELEMENT_COUNT);
}

void Rubato::keyschedule()
{
	for (int r = 0; r <= ROUNDS; r++)
	{
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			rand_vectors_[r * BLOCKSIZE + i] = (i + 1) % MODULUS;
			round_keys_[r * BLOCKSIZE + i] = rand_vectors_[r * BLOCKSIZE + i] * key_[i];
			round_keys_[r * BLOCKSIZE + i] %= MODULUS;
		}
	}
}

void Rubato::compute_keystream_naive(block_t out)
{
	assert(BLOCKSIZE == 16 || BLOCKSIZE == 36 || BLOCKSIZE == 64);
	uint64_t buf[BLOCKSIZE];
	uint64_t tmp[BLOCKSIZE];

	for (int i = 0; i < BLOCKSIZE; i++)
	{
		tmp[i] = i + 1;
	}

	uint64_t round_key[BLOCKSIZE];
	for (int r = 0; r < ROUNDS; r++)
	{
		// AddRoundKey
		memcpy(round_key, round_keys_ + r * BLOCKSIZE, sizeof(uint64_t) * BLOCKSIZE);
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			tmp[i] += round_key[i];
			tmp[i] %= MODULUS;
		}

		// Linear layer
		linear_layer(tmp, buf);

		// Nonlinear
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			buf[i] = tmp[i];
		}

		for (int i = 1; i < BLOCKSIZE; i++)
		{
			tmp[i] = (buf[i] + buf[i-1]*buf[i-1]) % MODULUS;
		}

#ifndef NDEBUG
		cout << "  Round " << r << ": " << hex << flush;
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			cout << tmp[i] << " ";
		}
		cout << dec << endl;
#endif
	}

	// Finalization
	{
		// Linear layer
		linear_layer(tmp, buf);

		// AddRoundKey
		memcpy(round_key, round_keys_ + ROUNDS * BLOCKSIZE, sizeof(uint64_t) * BLOCKSIZE);
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			tmp[i] += round_key[i];
			tmp[i] %= MODULUS;
		}
	}

	// Truncation
	memcpy(out, tmp, sizeof(uint64_t) * OUTPUTSIZE);
}

// Rubato private functions
void Rubato::linear_layer(block_t state, block_t buf)
{
#if BLOCKSIZE == 16
	// MixColumns
	for (int row = 0; row < 4; row++)
	{
		for (int col = 0; col < 4; col++)
		{
			buf[row * 4 + col] = 2 * state[row * 4 + col];
			buf[row * 4 + col] += 3 * state[((row + 1) % 4) * 4 + col];
			buf[row * 4 + col] += state[((row + 2) % 4) * 4 + col];
			buf[row * 4 + col] += state[((row + 3) % 4) * 4 + col];
			buf[row * 4 + col] %= MODULUS;
		}
	}

	// MixRows
	for (int row = 0; row < 4; row++)
	{
		for (int col = 0; col < 4; col++)
		{
			state[row * 4 + col] = 2 * buf[row * 4 + col];
			state[row * 4 + col] += 3 * buf[row * 4 + (col + 1) % 4];
			state[row * 4 + col] += buf[row * 4 + (col + 2) % 4];
			state[row * 4 + col] += buf[row * 4 + (col + 3) % 4];
			state[row * 4 + col] %= MODULUS;
		}
	}

#elif BLOCKSIZE == 36
	// MixColumns
	for (int row = 0; row < 6; row++)
	{
		for (int col = 0; col < 6; col++)
		{
			buf[row * 6 + col] = 4 * state[row * 6 + col];
			buf[row * 6 + col] += 2 * state[((row + 1) % 6) * 6 + col];
			buf[row * 6 + col] += 4 * state[((row + 2) % 6) * 6 + col];
			buf[row * 6 + col] += 3 * state[((row + 3) % 6) * 6 + col];
			buf[row * 6 + col] += state[((row + 4) % 6) * 6 + col];
			buf[row * 6 + col] += state[((row + 5) % 6) * 6 + col];
			buf[row * 6 + col] %= MODULUS;
		}
	}

	// MixRows
	for (int row = 0; row < 6; row++)
	{
		for (int col = 0; col < 6; col++)
		{
			state[row * 6 + col] = 4 * buf[row * 6 + col];
			state[row * 6 + col] += 2 * buf[row * 6 + (col + 1) % 6];
			state[row * 6 + col] += 4 * buf[row * 6 + (col + 2) % 6];
			state[row * 6 + col] += 3 * buf[row * 6 + (col + 3) % 6];
			state[row * 6 + col] += buf[row * 6 + (col + 4) % 6];
			state[row * 6 + col] += buf[row * 6 + (col + 5) % 6];
			state[row * 6 + col] %= MODULUS;
		}
	}

#else // BLOCKSIZE = 64
	// MixColumns
	for (int row = 0; row < 8; row++)
	{
		for (int col = 0; col < 8; col++)
		{
			buf[row * 8 + col] = 5 * state[row * 8 + col];
			buf[row * 8 + col] += 3 * state[((row + 1) % 8) * 8 + col];
			buf[row * 8 + col] += 4 * state[((row + 2) % 8) * 8 + col];
			buf[row * 8 + col] += 3 * state[((row + 3) % 8) * 8 + col];
			buf[row * 8 + col] += 6 * state[((row + 4) % 8) * 8 + col];
			buf[row * 8 + col] += 2 * state[((row + 5) % 8) * 8 + col];
			buf[row * 8 + col] += state[((row + 6) % 8) * 8 + col];
			buf[row * 8 + col] += state[((row + 7) % 8) * 8 + col];
			buf[row * 8 + col] %= MODULUS;
		}
	}

	// MixRows
	for (int row = 0; row < 8; row++)
	{
		for (int col = 0; col < 8; col++)
		{
			state[row * 8 + col] = 5 * buf[row * 8 + col];
			state[row * 8 + col] += 3 * buf[row * 8 + (col + 1) % 8];
			state[row * 8 + col] += 4 * buf[row * 8 + (col + 2) % 8];
			state[row * 8 + col] += 3 * buf[row * 8 + (col + 3) % 8];
			state[row * 8 + col] += 6 * buf[row * 8 + (col + 4) % 8];
			state[row * 8 + col] += 2 * buf[row * 8 + (col + 5) % 8];
			state[row * 8 + col] += buf[row * 8 + (col + 6) % 8];
			state[row * 8 + col] += buf[row * 8 + (col + 7) % 8];
			state[row * 8 + col] %= MODULUS;
		}
	}
#endif
}
