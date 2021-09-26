#ifndef __RUBATO_H__
#define __RUBATO_H__

#include "parms.h"
#include "ShakeAVX2.h"

using namespace std;

typedef uint64_t *block_t;

block_t block_init(size_t sz);

class Rubato
{
	public:
		Rubato(uint64_t *key)
		{
			key_ = block_init(BLOCKSIZE);

			for (int i = 0; i < BLOCKSIZE; i++)
			{
				key_[i] = key[i] % MODULUS;
			}

			rand_vectors_ = block_init(XOF_ELEMENT_COUNT + 8);
			round_keys_ = block_init(XOF_ELEMENT_COUNT);
			is_shake_init_ = false;
		}

		// Destruct a Rubato instance
		~Rubato()
		{
			free(key_);
			free(rand_vectors_);
			if (is_shake_init_)
			{
				delete shake_;
			}
		}

		// Re-keying function
		void set_key(uint64_t *key);

        /*
        Both init and update function compute round keys from
        extendable output function. The difference is, the init
        function creates a new ShakeAVX2 object while the update
        function does not.

        @param[in] nonce Distinct nonce
        @param[in] counter Counter, but may be used as an integrated nonce
         */
		void init(uint64_t nonce, uint64_t counter);
		void update(uint64_t nonce, uint64_t counter);

		void compute_keystream_naive(block_t out);

		// Copy outputs of XOF
		void get_rand_vectors(uint64_t *output);
		void get_round_keys(uint64_t *output);

	private:
		// Secret key
		block_t key_;

		// Round constants
		block_t rand_vectors_;

		// Key multiplied by round constant
		block_t round_keys_;

		// Shake object
		ShakeAVX2 *shake_;

		bool is_shake_init_;

		// The inner key schedule function in init and update
		void keyschedule();

		// Linear Layer
		inline void linear_layer(block_t state, block_t buf);
};

#endif
