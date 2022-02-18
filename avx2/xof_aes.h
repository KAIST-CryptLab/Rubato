#ifndef __XOF_AES_H__
#define __XOF_AES_H__

#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>

#include <stdint.h>
#include <stdlib.h>
#include "parms.h"

using namespace std;

class XoAES
{
    public:
        void init();
        void reset();
        void absorb_once(const uint8_t *in, size_t inlen);
        void squeeze(uint8_t *out, size_t outlen);

    private:
        EVP_CIPHER_CTX *ctx_;
        alignas(32) uint8_t zero_[128];
        alignas(32) uint8_t buf_[128];
        unsigned int pos_;
};

#endif
