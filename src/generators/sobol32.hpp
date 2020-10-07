#define SOBOL32_VECTORSIZ   (32)

/**
State of Sobol32 RNG.
*/
typedef struct{
  uint vector[SOBOL32_VECTORSIZ];
  uint d;
  uint i;
} sobol32_state;

const char * sobol32_prng_kernel = R"EOK(
/**
@file

Implements Sobol's quasirondom sequence genertor.

S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's quasirandom
sequence generator, 2003
http://doi.acm.org/10.1145/641876.641879
*/

#pragma once

#define RNG32
#define SOBOL32_VECTORSIZ   (32)

/**
State of Sobol32 RNG.
*/
typedef struct{
  uint vector[SOBOL32_VECTORSIZ];
  uint d;
  uint i;
} sobol32_state;

uint rightmost_zero_bit(uint x){
    if(x == 0){
        return 0;
    }
    uint y = x;
    uint z = 1;
    while(y & 1){
        y >>= 1;
        z++;
    }
    return z - 1;
}

/**
Internal function. Advances state of Sobol32 RNG.

@param state State of the RNG to advance.
*/
void discard_state(sobol32_state* state){
    state.d ^= state.vectors[rightmost_zero_bit(state.i)];
    state.i++;
}

/**
Generates a random 32-bit unsigned integer using Sobol RNG.

@param state State of the RNG to use.
*/
#define sobol32_uint(state) _sobol32_uint(&state)
uint _sobol32_uint(sobol32_state* state){
    uint p = state.d;
    state.d ^= state.vectors[rightmost_zero_bit(state.i)];
    state.i++;
    return p;
}

/**
Seeds Sobol32 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value unused. Only there to preserve expected function format. Seeding is to be done
            on the host side.
*/
void sobol32_seed(sobol32_state* state, ulong j){
}

/**
Generates a random 64-bit unsigned integer using Sobol RNG.

@param state State of the RNG to use.
*/
#define sobol32_ulong(state) ((((ulong)sobol32_uint(state)) << 32) | sobol32_uint(state))

/**
Generates a random float using Sobol RNG.

@param state State of the RNG to use.
*/
#define sobol32_float(state) (sobol32_uint(state)*SOBOL32_FLOAT_MULTI)

/**
Generates a random double using Sobol32 RNG.

@param state State of the RNG to use.
*/
#define sobol32_double(state) (sobol32_ulong(state)*SOBOL32_DOUBLE_MULTI)

/**
Generates a random double using Sobol RNG. Generated using only 32 random bits.

@param state State of the RNG to use.
*/
#define sobol32_double2(state) (sobol32_uint(state)*SOBOL32_DOUBLE2_MULTI)
)EOK";
