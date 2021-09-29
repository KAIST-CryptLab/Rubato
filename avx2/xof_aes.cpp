#include <string.h>

#include <openssl/conf.h>
#include <openssl/evp.h>

#include "xof_aes.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

using namespace std;

void handleErrors(void)
{
    ERR_print_errors_fp(stderr);
    abort();
}

void XoAES::init()
{
    ctx_ = EVP_CIPHER_CTX_new();
    memset(zero_, 0, RATE_IN_BYTE);
    pos_ = 0;
}

void XoAES::reset()
{
    EVP_CIPHER_CTX_reset(ctx_);
    pos_ = 0;
}

void XoAES::absorb_once(const uint8_t *in, size_t inlen)
{
    alignas(32) uint8_t iv[16] = {0,};
    if (inlen != 16 && inlen != 32)
        abort();

    memcpy(iv, in+16, inlen-16);

    EVP_EncryptInit_ex(ctx_, EVP_aes_128_ctr(), NULL, in, iv);
    EVP_CIPHER_CTX_set_padding(ctx_, 0);
    pos_ = RATE_IN_BYTE;
}

void XoAES::squeeze(uint8_t *out, size_t outlen)
{
    int sz;
    while (outlen) {
        if (pos_ == RATE_IN_BYTE) {
            EVP_EncryptUpdate(ctx_, buf_, &sz, zero_, RATE_IN_BYTE);
            pos_ = 0;
        }

        unsigned int squeeze_size = MIN(RATE_IN_BYTE - pos_, outlen);
        memcpy(out, buf_+pos_, squeeze_size);
        outlen -= squeeze_size;
        out += squeeze_size;
        pos_ += squeeze_size;
    }
}
