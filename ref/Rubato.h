#ifndef __RUBATO_H__
#define __RUBATO_H__

#include <sys/random.h>
#include "parms.h"
#include "xof.h"

extern "C" {
#include "util.h"
}

#define MAX_BLOCKSIZE 64

using namespace std;

class Rubato
{
	public:
		Rubato(uint32_t key[BLOCKSIZE])
		{
			for (int i = 0; i < BLOCKSIZE; i++)
			{
				key_[i] = key[i];
				state_[i] = i+1;
			}
			xof_coeff_ = new XOF();
			xof_noise_ = new XOF();
			xof_coeff_->init();
			xof_noise_->init();
			do {
				rand_bytes(&seed_, 8);
			} while (seed_ == 0);
		}

		// Destruct a Rubato instance
		~Rubato()
		{
			delete xof_coeff_;
			delete xof_noise_;
		}

		// Re-keying function
		void set_key(uint32_t key[BLOCKSIZE])
		{
			for (int i = 0; i < BLOCKSIZE; i++)
			{
				key_[i] = key[i];
				state_[i] = i+1;
			}
			xof_coeff_->reset();
			xof_noise_->reset();
			do {
				rand_bytes(&seed_, 8);
			} while (seed_ != 0);
		}

        /*
       Init function compute round keys from extendable output function.

        @param[in] nonce Distinct nonce
        @param[in] counter Counter, but may be used as an integrated nonce
        */
		void init(uint64_t nonce, uint64_t counter);
		void reset();
		void crypt(uint32_t output[BLOCKSIZE]);

		// for debug
		void get_coeffs(uint64_t *output);
		void get_round_keys(uint64_t *output);

	private:
		// Secret key
		alignas(32) uint64_t key_[MAX_BLOCKSIZE];

		// Round constants
		alignas(32) uint64_t coeffs_[(ROUNDS + 1) * MAX_BLOCKSIZE + 8];//

		// Key multiplied by round constant
		alignas(32) uint64_t round_keys_[(ROUNDS+1) * MAX_BLOCKSIZE + 8];

		// Internal state
		alignas(32) uint64_t state_[MAX_BLOCKSIZE];

		// XOF objects
		XOF *xof_coeff_;
		XOF *xof_noise_;

		uint64_t seed_;

		// The inner key schedule function in init and update
		void gen_coeffs();
		void keyschedule();
};

#endif
