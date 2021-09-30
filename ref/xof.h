#ifndef __XOF_H__
#define __XOF_H__

#include "parms.h"

#if XOF_TYPE == XOF_SHAKE128
    #include "xof_shake.h"
    #define RATE_IN_BYTE 168
    #define XOF Shake
#elif XOF_TYPE == XOF_SHAKE256
    #include "xof_shake.h"
    #define RATE_IN_BYTE 136
    #define XOF Shake
#elif XOF_TYPE == XOF_AES
    #include "xof_aes.h"
    #define XOF XoAES
#endif

#endif
