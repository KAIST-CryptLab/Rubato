# Rubato
This is an implementation of [Rubato](https://eprint.iacr.org/2022/537) cipher

## Contents
We implement:
- reference Rubato implementation in [ref](./ref) with test code for debugging;
- fast Rubato implementation in [avx2](./avx2) with test code for debugging and benchmark scripts.

## How to use
Each directory has Makefile with following commands:
- ```bench```(only in [avx2](./avx2)): create benchmark code;
- ```test```: create test code for debugging - it prints all internal variables.
