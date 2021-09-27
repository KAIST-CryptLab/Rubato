#ifndef __XOF_H__
#define __XOF_H__

#include "parms.h"

#if XOF_TYPE == XOF_SHAKE
    #include "xof_shake.h"
    #define XOF Shake
#elif XOF_TYPE == XOF_AES
    #include "xof_aes.h"
    #define XOF XoAES
#endif

#endif
