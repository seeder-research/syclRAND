typedef struct{
    uint x[5]; // Xorshift values (160 bits)
    uint d;    // Weyl sequence value
} xorwow_state;

const char * xorwow_prng_kernel = R"EOK(
/**
@file

Implements a 64-bit xorwow* generator that returns 32-bit values.

// G. Marsaglia, Xorshift RNGs, 2003
// http://www.jstatsoft.org/v08/i14/paper
*/
#pragma once
#define RNG32

#define XORWOW_FLOAT_MULTI 2.3283064365386963e-10f
#define XORWOW_DOUBLE2_MULTI 2.3283064365386963e-10
#define XORWOW_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of xorwow RNG.
*/
typedef struct{
    uint x[5]; // Xorshift values (160 bits)
    uint d;    // Weyl sequence value
} xorwow_state;

/**
Generates a random 32-bit unsigned integer using xorwow RNG.

@param state State of the RNG to use.
*/
#define xorwow_uint(state) _xorwow_uint(&state)
uint _xorwow_uint(xorwow_state* restrict state){
        const uint t = state->x[0] ^ (state->x[0] >> 2);
        state->x[0] = state->x[1];
        state->x[1] = state->x[2];
        state->x[2] = state->x[3];
        state->x[3] = state->x[4];
        state->x[4] = (state->x[4] ^ (state->x[4] << 4)) ^ (t ^ (t << 1));

        state->d += 362437;

        return state->d + state->x[4];
}

/**
Seeds xorwow RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void xorwow_seed(xorwow_state* state, unsigned long j){
        state->x[0] = 123456789U;
        state->x[1] = 362436069U;
        state->x[2] = 521288629U;
        state->x[3] = 88675123U;
        state->x[4] = 5783321U;

        state->d = 6615241U;

        // Constants are arbitrary prime numbers
        const uint s0 = (uint)(j) ^ 0x2c7f967fU;
        const uint s1 = (uint)(j >> 32) ^ 0xa03697cbU;
        const uint t0 = 1228688033U * s0;
        const uint t1 = 2073658381U * s1;
        state->x[0] += t0;
        state->x[1] ^= t0;
        state->x[2] += t1;
        state->x[3] ^= t1;
        state->x[4] += t0;
        state->d += t1 + t0;

}

/**
Generates a random 64-bit unsigned integer using xorwow RNG.

@param state State of the RNG to use.
*/
#define xorwow_ulong(state) ((((ulong)xorwow_uint(state)) << 32) | xorwow_uint(state))

/**
Generates a random float using xorwow RNG.

@param state State of the RNG to use.
*/
#define xorwow_float(state) (xorwow_uint(state)*XORWOW_FLOAT_MULTI)

/**
Generates a random double using xorwow RNG.

@param state State of the RNG to use.
*/
#define xorwow_double(state) (xorwow_ulong(state)*XORWOW_DOUBLE_MULTI)

/**
Generates a random double using xorwow RNG. Generated using only 32 random bits.

@param state State of the RNG to use.
*/
#define xorwow_double2(state) (xorwow_uint(state)*XORWOW_DOUBLE2_MULTI)
)EOK";
