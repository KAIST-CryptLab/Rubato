#ifndef __PARMS_H__
#define __PARMS_H__

// Parameter
#define BLOCKSIZE 16
#define OUTPUTSIZE 12
#define ROUNDS 4
#define MODULUS 0xffa0001ULL
#define MOD_BIT_COUNT 28

#define XOF_ELEMENT_COUNT ((ROUNDS + 1) * BLOCKSIZE)
#endif
