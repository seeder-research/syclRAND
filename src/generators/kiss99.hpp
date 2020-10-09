#ifndef __KISS99_RNG__
#define __KISS99_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS


/**
@file

Implements KISS (Keep It Simple, Stupid) generator, proposed in 1999.

G. Marsaglia, Random numbers for c: End, at last?, http://www.cse.yorku.ca/~oz/marsaglia-rng.html.
*/
#pragma once
#define RNG32

#define KISS99_FLOAT_MULTI 2.3283064365386963e-10f
#define KISS99_DOUBLE2_MULTI 2.3283064365386963e-10
#define KISS99_DOUBLE_MULTI 5.4210108624275221700372640e-20

//http://www.cse.yorku.ca/~oz/marsaglia-rng.html

/**
State of kiss99 RNG.
*/
typedef struct {
	uint z, w, jsr, jcong;
} kiss99_state;

/**
Generates a random 32-bit unsigned integer using kiss99 RNG.

This is alternative, macro implementation of kiss99 RNG.

@param state State of the RNG to use.
*/
#define kiss99_macro_uint(state) (\
	/*multiply with carry*/ \
	state.z = 36969 * (state.z & 65535) + (state.z >> 16), \
	state.w = 18000 * (state.w & 65535) + (state.w >> 16), \
	/*xorshift*/ \
	state.jsr ^= state.jsr << 17, \
	state.jsr ^= state.jsr >> 13, \
	state.jsr ^= state.jsr << 5, \
	/*linear congruential*/ \
	state.jcong = 69069 * state.jcong + 1234567, \
	\
	(((state.z << 16) + state.w) ^ state.jcong) + state.jsr \
	)

/**
Generates a random 32-bit unsigned integer using kiss99 RNG.

@param state State of the RNG to use.
*/
#define kiss99_uint(state) _kiss99_uint(&state)
uint _kiss99_uint(kiss99_state* state){
	//multiply with carry
	state->z = 36969 * (state->z & 65535) + (state->z >> 16);
	state->w = 18000 * (state->w & 65535) + (state->w >> 16);

	//xorshift
	state->jsr ^= state->jsr << 17;
	state->jsr ^= state->jsr >> 13;
	state->jsr ^= state->jsr << 5;

	//linear congruential
	state->jcong = 69069 * state->jcong + 1234567;

	return (((state->z << 16) + state->w) ^ state->jcong) + state->jsr;
}

/**
Seeds kiss99 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void kiss99_seed(kiss99_state* state, ulong j){
	state->z=362436069 ^ (uint)j;
	if(state->z==0){
		state->z=1;
	}
	state->w=521288629 ^ (uint)(j >> 32);
	if(state->w==0){
		state->w=1;
	}
	state->jsr=123456789 ^ (uint)j;
	if(state->jsr==0){
		state->jsr=1;
	}
	state->jcong=380116160 ^ (uint)(j >> 32);
}

/**
Generates a random 64-bit unsigned integer using kiss99 RNG.

@param state State of the RNG to use.
*/
#define kiss99_ulong(state) ((((ulong)kiss99_uint(state)) << 32) | kiss99_uint(state))

// Kernel function
// Seed RNG by single ulong
class kiss99_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<kiss99_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		kiss99_seed_by_value_kernel(ulong val,
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
            kiss99_state state;
            kiss99_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class kiss99_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<kiss99_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		kiss99_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            kiss99_state state;
            kiss99_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class kiss99_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<kiss99_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		kiss99_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            kiss99_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= kiss99_uint(state);
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
void KISS99_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 kiss99_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void KISS99_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 kiss99_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void KISS99_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 kiss99_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif __KISS99_RNG__