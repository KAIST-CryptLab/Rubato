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

      if (argc != 3)
      {
          cout << "Usage : " << argv[0] << " [nonce] [counter]" << endl;
          exit(0);
      }

	for (int i = 0; i < BLOCKSIZE; i++)
	{
		key[i] = i + 1;
	}

	nonce = stoull(argv[1]);
	counter = stoull(argv[2]);

	Rubato cipher(key);
	cipher.init(nonce, counter);
	cipher.get_coeffs(coeffs);
	for (int i = 0; i < XOF_ELEMENT_COUNT; i++)
		cout << coeffs[i] << " ";
	cout << endl;

	cipher.crypt(keystream);
	for (int i = 0; i < OUTPUTSIZE; i++)
		cout << keystream[i] << " ";
	cout << endl;
}
