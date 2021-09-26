#include <cstring>
#include <cassert>
#include <x86intrin.h>
#include "Rubato.h"
#ifndef NDBUG
#include <iostream>
#endif

constexpr int unpack_order[8] = {2, 3, 6, 7, 0, 1, 4, 5};

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

void Rubato::get_round_keys(uint64_t *output)
{
	memcpy(output, round_keys_, sizeof(uint64_t) * XOF_ELEMENT_COUNT);
}

void Rubato::keyschedule()
{
	#if 1 // Naive rejection sampling
	uint8_t buf[RATE_IN_BYTE];
	unsigned int offset = 0;
	unsigned int squeeze_byte = 0;

	shake_->squeeze(buf, RATE_IN_BYTE);
	int ctr = RATE_IN_BYTE / 4;
	while (offset < XOF_ELEMENT_COUNT)
	{
		if (ctr == 0)
		{
			shake_->squeeze(buf, RATE_IN_BYTE);
			squeeze_byte = 0;
			ctr = RATE_IN_BYTE / 4;
		}

		uint64_t elem;
		memcpy(&elem, buf + squeeze_byte, 4);
		elem &= ((1ULL << MOD_BIT_COUNT) - 1);
		squeeze_byte += 4;

		if (elem < MODULUS)
		{
			rand_vectors_[offset] = elem;
			offset++;
		}
		ctr--;
	}

	for (int r = 0; r <= ROUNDS; r++)
	{
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			round_keys_[r * BLOCKSIZE + i] = rand_vectors_[r * BLOCKSIZE + i] * key_[i];
			round_keys_[r * BLOCKSIZE + i] %= MODULUS;
		}
	}
	#else // Rejection sampleing using AVX2
	// Set random vector from SHAKE256
	const __m256i zero = _mm256_setzero_si256();
	const __m256i modulus32 = _mm256_set1_epi32(MODULUS);
	const __m256i all_one_mask32 = _mm256_set1_epi32((1ULL << MOD_BIT_COUNT) - 1);
	__m256i u0, u1, u2, u3, v0, v1, v2, v3, idx;

	__attribute__((aligned(32))) uint8_t buf[160];
	memset(buf, 0xff, 160);

	unsigned int offset = 0;
	unsigned int squeeze_byte = 0;
	__attribute__((aligned(32))) uint32_t idx_arr[8] = {0};

	shake_->squeeze(buf, RATE_IN_BYTE);
	int ctr = RATE_IN_BYTE / 32;
	while (offset < XOF_ELEMENT_COUNT)
	{
		if (squeeze_byte == RATE_IN_BYTE)
		{
			shake_->squeeze(buf, RATE_IN_BYTE);
			squeeze_byte = 0;
			ctr = RATE_IN_BYTE / 32;
		}

		if (ctr > 0)
		{
			u0 = _mm256_loadu_si256((__m256i *)(buf + squeeze_byte));
			u0 = _mm256_and_si256(u0, all_one_mask32);
			u1 = _mm256_cmpgt_epi32(modulus32, u0);

			int good_idx = _mm256_movemask_ps((__m256)u1);
			int good_cnt = 0;
			int bad_cnt = 0;
			for (int i = 0; i < 8; i++)
			{
				if (good_idx & (1 << i))
				{
					idx_arr[unpack_order[good_cnt]] = i;
					good_cnt++;
				}
				else
				{
					idx_arr[unpack_order[7 - bad_cnt]] = i;
					bad_cnt++;
				}
			}
			idx = _mm256_load_si256((__m256i *)idx_arr);
			u2 = _mm256_permutevar8x32_epi32(u0, idx);

			u0 = _mm256_unpackhi_epi32(u2, zero);
			u1 = _mm256_unpacklo_epi32(u2, zero);
			_mm256_storeu_si256((__m256i *)(rand_vectors_ + offset), u0);
			_mm256_storeu_si256((__m256i *)(rand_vectors_ + offset + 4), u1);

			offset += good_cnt;
			squeeze_byte += 32;
			ctr--;
		}
		else // Last 8 bytes in the squeezed bytes
		{
			u0 = _mm256_loadu_si256((__m256i *) (buf + squeeze_byte));
			u0 = _mm256_and_si256(u0, all_one_mask32);
			u1 = _mm256_cmpgt_epi32(modulus32, u0);

			int good_idx = _mm256_movemask_ps((__m256)u1);
			int good_cnt = 0;
			int bad_cnt = 0;
			for (int i = 0; i < 8; i++)
			{
				if (good_idx & (1 << i))
				{
					idx_arr[unpack_order[good_cnt]] = i;
					good_cnt++;
				}
				else
				{
					idx_arr[unpack_order[7 - bad_cnt]] = i;
					bad_cnt++;
				}
			}
			idx = _mm256_load_si256((__m256i *)idx_arr);
			u2 = _mm256_permutevar8x32_epi32(u0, idx);

			u0 = _mm256_unpackhi_epi32(u2, zero);
			u1 = _mm256_unpacklo_epi32(u2, zero);
			_mm256_storeu_si256((__m256i *) (rand_vectors_ + offset), u0);

			offset += good_cnt;
			squeeze_byte += 8;
		}
	}

	// Compute round keys
	// round_keys[i] = (key_[i] * rand_vectors_[i]) % MODULUS;
	u0 = _mm256_load_si256((__m256i *) key_);
	u1 = _mm256_load_si256((__m256i *) (key_ + 4));
	u2 = _mm256_load_si256((__m256i *) (key_ + 8));
	u3 = _mm256_load_si256((__m256i *) (key_ + 12));

	for (int  r = 0; r <= ROUNDS; r++)
	{
		v0 = _mm256_load_si256((__m256i *) (rand_vectors_ + BLOCKSIZE * r));
		v1 = _mm256_load_si256((__m256i *) (rand_vectors_ + BLOCKSIZE * r + 4));
		v2 = _mm256_load_si256((__m256i *) (rand_vectors_ + BLOCKSIZE * r + 8));
		v3 = _mm256_load_si256((__m256i *) (rand_vectors_ + BLOCKSIZE * r + 12));

		v0 = _mm256_mul_epu32(u0, v0);
		v1 = _mm256_mul_epu32(u1, v1);
		v2 = _mm256_mul_epu32(u2, v2);
		v3 = _mm256_mul_epu32(u3, v3);

		_mm256_store_si256((__m256i *) (round_keys_ + BLOCKSIZE * r), v0);
		_mm256_store_si256((__m256i *) (round_keys_ + BLOCKSIZE * r + 4), v1);
		_mm256_store_si256((__m256i *) (round_keys_ + BLOCKSIZE * r + 8), v2);
		_mm256_store_si256((__m256i *) (round_keys_ + BLOCKSIZE * r + 12), v3);
	}

	for (int i = 0; i < (ROUNDS + 1) * BLOCKSIZE; i++)
	{
		round_keys_[i] %= MODULUS;
	}

	#endif
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
		cout << "[Round " << r << "]" << hex << endl;
		for (int i = 0; i < BLOCKSIZE; i++)
		{
			cout << (uint32_t)(tmp[i] * 0x100000000ULL % MODULUS) << " ";
		}
		cout << endl;
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
