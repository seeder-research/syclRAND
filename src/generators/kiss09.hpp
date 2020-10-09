#ifndef __KISS09_RNG__
#define __KISS09_RNG__

#ifndef __SYCLRAND_CLASS
#include "common/syclrand_def.hpp"
#endif __SYCLRAND_CLASS

/**
@file

Implements KISS (Keep It Simple, Stupid) generator, proposed in 2009.

G. Marsaglia, 64-bit kiss rngs, https://www.thecodingforums.com/threads/64-bit-kiss-rngs.673657.
*/
#pragma once

#define KISS09_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define KISS09_DOUBLE_MULTI 5.4210108624275221700372640e-20

//https://www.thecodingforums.com/threads/64-bit-kiss-rngs.673657/

/**
State of kiss09 RNG.
*/
typedef struct {
	ulong x,c,y,z;
} kiss09_state;

/* KISS09 class */
class KISS09_PRNG : _SyCLRAND {
	public:
	    using state_accessor = 
		      sycl::accessor<kiss09_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using output_accessor = 
			  sycl::accessor<uint, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using seed_accessor = 
		      sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		void seed_by_value(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
		void seed_by_array(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
		void generate_uint(cl::sycl::queue funcQueue, int count, cl::sycl::buffer &dst, size_t gsize, size_t lsize);
	private:
	    state_accessor      stateBuf;
		kiss09_state        *stateArr;
};

/**
Generates a random 64-bit unsigned integer using kiss09 RNG.

@param state State of the RNG to use.
*/
#define kiss09_ulong(state) (\
	/*multiply with carry*/ \
	state.c = state.x >> 6, \
	state.x += (state.x << 58) + state.c, \
	state.c += state.x < (state.x << 58) + state.c, \
	/*xorshift*/ \
	state.y ^= state.y << 13, \
	state.y ^= state.y >> 17, \
	state.y ^= state.y << 43, \
	/*linear congruential*/ \
	state.z = 6906969069UL * state.z + 1234567UL, \
	state.x + state.y + state.z \
	)

@param state State of the RNG to use.
*/
#define kiss09_uint(state) ((uint)kiss09_ulong(state))

/**
Generates a random 64-bit unsigned integer using kiss09 RNG.

This is alternative implementation of kiss09 RNG as a function.

@param state State of the RNG to use.
*/
#define kiss09_func_ulong(state) _kiss09_func_ulong(&state)
ulong _kiss09_func_ulong(kiss09_state* state){
	//multiply with carry
	ulong t = (state->x << 58) + state->c;
	state->c = state-> x >>6;
	state->x += t;
	state->c += state->x < t;
	//xorshift
	state->y ^= state->y << 13;
	state->y ^= state->y >> 17;
	state->y ^= state->y << 43;
	//linear congruential
	state->z = 6906969069UL * state->z + 1234567UL;
	return state->x + state->y + state->z;
}

/**
Seeds kiss09 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void kiss09_seed(kiss09_state* state, ulong j){
	state->x = 1234567890987654321UL ^ j;
	state->c = 123456123456123456UL ^ j;
	state->y = 362436362436362436UL ^ j;
	if(state->y==0){
		state->y=1;
	}
	state->z = 1066149217761810UL ^ j;
}

// Kernel function
// Seed RNG by single ulong
class kiss09_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<kiss09_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		kiss09_seed_by_value_kernel(ulong val,
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
            kiss09_state state;
            kiss09_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class kiss09_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<kiss09_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		kiss09_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            kiss09_state state;
            kiss09_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class kiss09_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<kiss09_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		kiss09_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            kiss09_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= kiss09_uint(state);
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
void KISS09_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 kiss09_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void KISS09_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 kiss09_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void KISS09_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 kiss09_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif __KISS09_RNG__
