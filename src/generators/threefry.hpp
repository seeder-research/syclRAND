#define C240 0x1BD11BDAA9FC1A22
#define N_ROUNDS 20
#define DOUBLE_MULT 5.421010862427522e-20

/**
State of threefry RNG.
*/
typedef struct{
	uint2 counter;
	uint2 result;
	uint2 key;
	uint tracker;
} threefry_state;

const char * threefry_prng_kernel = R"EOK(
/**
@file

Implements threefry RNG.

/*******************************************************
 * Modified version of Random123 library:
 * https://www.deshawresearch.com/downloads/download_random123.cgi/
 * The original copyright can be seen here:
 *
 * RANDOM123 LICENSE AGREEMENT
 *
 * Copyright 2010-2011, D. E. Shaw Research. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions, and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions, and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * Neither the name of D. E. Shaw Research nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************/

#pragma once

#define ELEMENTS_PER_BLOCK 256
#define SKEIN_KS_PARITY 0x1BD11BDAA9FC1A22
#define DOUBLE_MULT 5.421010862427522e-20

static const int ROTATION[] = {16, 42, 12, 31, 16, 32, 24, 21};

/**
State of threefry RNG.
*/
typedef struct{
	ulong2 counter;
	ulong2 result;
	ulong2 key;
	uint tracker;
} threefry_state;

inline static
ulong rotL(ulong x, uint N){
  return ((x << N) | (x >> (64 - N)));
}

inline
void threefry_round(threefry_state* state){
    uint ks[3];

    ks[2] = SKEIN_KS_PARITY;
    ks[0] = state.key.x;
    state.result.x  = state.counter.x;
    ks[2] ^= state.key.x;
    ks[1] = state.key.y;
    state.result.y  = state.counter.y;
    ks[2] ^= state.key.y;

    state.result.x += ks[0];
    state.result.y += ks[1];

    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R0);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R1);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R2);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R3);
    state.result.y ^= state.result.x;

    /* InjectKey(r=1) */
    state.result.x += ks[1];
    state.result.y += ks[2];
    state.result.y += 1; /* X[2-1] += r  */

    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R4);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R5);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R6);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R7);
    state.result.y ^= state.result.x;

    /* InjectKey(r=2) */
    state.result.x += ks[2];
    state.result.y += ks[0];
    state.result.y += 2;

    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R0);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R1);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R2);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R3);
    state.result.y ^= state.result.x;

    /* InjectKey(r=3) */
    state.result.x += ks[0];
    state.result.y += ks[1];
    state.result.y += 3;

    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R4);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R5);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R6);
    state.result.y ^= state.result.x;
    state.result.x += state.result.y;
    state.result.y = rotL(state.result.y, R7);
    state.result.y ^= state.result.x;

    /* InjectKey(r=4) */
    state.result.x += ks[1];
    state.result.y += ks[2];
    state.result.y += 4;
}

/**
Seeds threefry RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void threefry_seed(threefry_state *state, ulong j){
    state->key->x = (uint)(j);
    state->key->y = (uint)(j >> 32);
}

/**
Generates a random 64-bit unsigned long using threefry RNG.

@param state State of the RNG to use.
*/
#define threefry_ulong(state) (ulong)_threefry_uint(state)

/**
Generates a random 32-bit unsigned integer using threefry RNG.

@param state State of the RNG to use.
*/
#define threefry_uint(state) (uint)_threefry_ulong(state)
uint _threefry_uint(threefry_state *state){
    index = get_group_id(0) * ELEMENTS_PER_BLOCK + get_local_id(0);
    if (state.tracker == 1) {
        uint tmp = state.result.y;
        state.counter.x += index;
        state.counter.y += (state.counter.y < index);
        threefry_round(state);
        state.tracker = 0;
        return tmp;
    } else {
        state->tracker++;
        return state.result.x;
    }
}

/**
Generates a random float using threefry RNG.

@param state State of the RNG to use.
*/
#define threefry_float(state) (threefry_uint(state)*THREEFRY_FLOAT_MULTI)

/**
Generates a random double using threefry RNG.

@param state State of the RNG to use.
*/
#define threefry_double(state) (threefry_uint(state)*THREEFRY_DOUBLE_MULTI)

/**
Generates a random double using threefry RNG. Since threefry returns 64-bit numbers this is equivalent to threefry_double.

@param state State of the RNG to use.
*/
#define threefry_double2(state) threefry_double(state)
)EOK";
