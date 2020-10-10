#ifndef __THREEFRY_RNG__
#define __THREEFRY_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

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

// Kernel function
// Seed RNG by single ulong
class threefry_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<threefry_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		threefry_seed_by_value_kernel(ulong val,
			state_accessor statePtr)
		: seedVal(val),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id(0);
            ulong seed = (ulong)(gid);
            seed <<= 1;
            seed += seedVal;
            if (seed == 0) {
                seed += 1;
            }
            threefry_state state;
            threefry_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class threefry_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<threefry_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		threefry_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            threefry_state state;
            threefry_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class threefry_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<threefry_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		threefry_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            threefry_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= threefry_uint(state);
            }
            stateBuf[gid] = state;
		}

	private:
		int             num;
		state_accessor  stateBuf;
		output_accessor res;
};

// Class function
// Launch kernel to seed RNG by single ulong
void THREEFRY_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 threefry_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void THREEFRY_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 threefry_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void THREEFRY_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 threefry_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __THREEFRY_RNG__
