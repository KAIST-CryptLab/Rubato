#ifndef __PARMS_H__
#define __PARMS_H__

#define PARAM_80_S 0
#define PARAM_80_M 1
#define PARAM_80_L 2
#define PARAM_128_S 3
#define PARAM_128_M 4
#define PARAM_128_L 5

#define XOF_AES 0
#define XOF_SHAKE 1

#define PARAM_ID PARAM_80_M
#define XOF_TYPE XOF_SHAKE

#if PARAM_ID == PARAM_80_S
    #define BLOCKSIZE 16
    #define OUTPUTSIZE 12
    #define ROUNDS 2
    #define Q 0xffa0001ULL
    #define Q_BIT_MASK 0xfffffffULL
    #define XOF_ELEMENT_COUNT ((ROUNDS + 1) * BLOCKSIZE)

    #define R 0x100000000ULL
    #define R2_MOD_Q 0x907c029ULL
    #define Qbar_MOD_R 0xff9ffffULL
    #define RINV_MOD_Q 0xff4024ULL
#elif PARAM_ID == PARAM_80_M
    #define BLOCKSIZE 36
    #define OUTPUTSIZE 32
    #define ROUNDS 2
    #define Q 0x3ee0001ULL
    #define Q_BIT_MASK 0x3ffffffULL
    #define XOF_ELEMENT_COUNT ((ROUNDS + 1) * BLOCKSIZE)

    #define R 0x100000000ULL
    #define R2_MOD_Q 0x2b2e03bULL
    #define Qbar_MOD_R 0x3edffffULL
    #define RINV_MOD_Q 0xf7144ULL
#elif PARAM_ID == PARAM_80_L
    #define BLOCKSIZE 64
    #define OUTPUTSIZE 60
    #define ROUNDS 2
    #define Q 0x1fc0001ULL
    #define Q_BIT_MASK 0x1ffffffULL
    #define XOF_ELEMENT_COUNT ((ROUNDS + 1) * BLOCKSIZE)

    #define R 0x100000000ULL
    #define R2_MOD_Q 0x3038f3ULL
    #define Qbar_MOD_R 0x1fbffffULL
    #define RINV_MOD_Q 0x3f010ULL
#elif PARAM_ID == PARAM_128_S
    #define BLOCKSIZE 16
    #define OUTPUTSIZE 12
    #define ROUNDS 5
    #define Q 0xffa0001ULL
    #define Q_BIT_MASK 0xfffffffULL
    #define XOF_ELEMENT_COUNT ((ROUNDS + 1) * BLOCKSIZE)

    #define R 0x100000000ULL
    #define R2_MOD_Q 0x907c029ULL
    #define Qbar_MOD_R 0xff9ffffULL
    #define RINV_MOD_Q 0xff4024ULL
#elif PARAM_ID == PARAM_128_M
    #define BLOCKSIZE 36
    #define OUTPUTSIZE 32
    #define ROUNDS 3
    #define Q 0x3ee0001ULL
    #define Q_BIT_MASK 0x3ffffffULL
    #define XOF_ELEMENT_COUNT ((ROUNDS + 1) * BLOCKSIZE)

    #define R 0x100000000ULL
    #define R2_MOD_Q 0x2b2e03bULL
    #define Qbar_MOD_R 0x3edffffULL
    #define RINV_MOD_Q 0xf7144ULL
#elif PARAM_ID == PARAM_128_L
    #define BLOCKSIZE 64
    #define OUTPUTSIZE 60
    #define ROUNDS 2
    #define Q 0x3ee0001ULL
    #define Q_BIT_MASK 0x3ffffffULL
    #define XOF_ELEMENT_COUNT ((ROUNDS + 1) * BLOCKSIZE)

    #define R 0x100000000ULL
    #define R2_MOD_Q 0x2b2e03bULL
    #define Qbar_MOD_R 0x3edffffULL
    #define RINV_MOD_Q 0xf7144ULL
#endif

#endif