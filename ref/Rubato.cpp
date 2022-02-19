#include <cstring>
#include <cassert>
#include <x86intrin.h>
#include "Rubato.h"
#include <iostream>

void dump_state36(uint64_t *tmp)
{
	for (int row = 0; row < 6; row++)
	{
		uint32_t ptr[6];
		for (int col = 0; col < 6; col++)
		{
			ptr[col] = tmp[6 * row + col] * R % Q;
		}
		printf("[%08x %08x %08x %08x %08x %08x]\n", ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5]);
	}
	puts("");
}

void dump_state64(uint64_t *tmp)
{
	for (int row = 0; row < 8; row++)
	{
		uint32_t ptr[8];
		for (int col = 0; col < 8; col++)
		{
			ptr[col] = tmp[8 * row + col] * R % Q;
		}
		printf("[%08x %08x %08x %08x %08x %08x %08x %08x]\n", ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
	}
	puts("");
}

void linear_layer(uint64_t* state, uint64_t* buf)
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
			buf[row * 4 + col] %= Q;
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
			state[row * 4 + col] %= Q;
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
			buf[row * 6 + col] %= Q;
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
			state[row * 6 + col] %= Q;
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
			buf[row * 8 + col] %= Q;
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
			state[row * 8 + col] %= Q;
		}
	}
#endif
}

void Rubato::init(uint64_t nonce, uint64_t counter)
{
	uint8_t buf[32];
	*(uint64_t *)buf = nonce;
	*(uint64_t *)(buf + 8) = counter;
	*(uint64_t *)(buf + 16) = seed_;
	*(uint64_t *)(buf + 24) = 0;
	xof_coeff_->absorb_once(buf, 16);
	xof_noise_->absorb_once(buf, 32);

	keyschedule();
}

void Rubato::reset()
{
	xof_coeff_->reset();
	xof_noise_->reset();
}

void Rubato::get_coeffs(uint64_t *output)
{
	memcpy(output, coeffs_, sizeof(uint64_t) * XOF_ELEMENT_COUNT);
}

void Rubato::get_round_keys(uint64_t *output)
{
	memcpy(output, round_keys_, sizeof(uint64_t) * XOF_ELEMENT_COUNT);
}

void Rubato::keyschedule()
{
	uint8_t buf[RATE_IN_BYTE];
	unsigned int offset = 0;
	unsigned int squeeze_byte = 0;

	xof_coeff_->squeeze(buf, RATE_IN_BYTE);

	int ctr = RATE_IN_BYTE / 4;
	while (offset < XOF_ELEMENT_COUNT)
	{
		if (ctr == 0)
		{
			xof_coeff_->squeeze(buf, RATE_IN_BYTE);
			squeeze_byte = 0;
			ctr = RATE_IN_BYTE / 4;
		}

		uint64_t elem;
		memcpy(&elem, buf + squeeze_byte, 4);
		elem &= Q_BIT_MASK;
		squeeze_byte += 4;

		if (elem < Q)
		{
			coeffs_[offset] = elem;
			offset++;
		}
		ctr--;
	}

	for (int r = 0; r <= ROUNDS; r++)
	{
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			round_keys_[r * BLOCKSIZE + i] = coeffs_[r * BLOCKSIZE + i] * key_[i];
			round_keys_[r * BLOCKSIZE + i] %= Q;
		}
	}
}

void Rubato::crypt(uint32_t output[OUTPUTSIZE])
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
			tmp[i] %= Q;
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
			tmp[i] = (buf[i] + buf[i-1]*buf[i-1]) % Q;
		}
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
			tmp[i] %= Q;
		}
	}

	for (int i = 0; i < OUTPUTSIZE; i++)
	{
		output[i] = tmp[i];
	}
}
