#ifndef __PHILOX2X32_10_RNG__
#define __PHILOX2X32_10_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements philox2x32-10 RNG.

J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, Parallel random numbers: as easy as 1, 2, 3, in: High Performance Computing, Networking, Storage and Analysis (SC), 2011 International Conference for, IEEE, 2011, pp. 1–12.
*/
#pragma once

#define PHILOX2X32_10_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define PHILOX2X32_10_DOUBLE_MULTI 5.4210108624275221700372640e-20

#define PHILOX2X32_10_MULTIPLIER 0xd256d193
#define PHILOX2X32_10_KEY_INC 0x9E3779B9
//#define PHILOX2X64_10_MULTIPLIER 0xD2B74407B1CE6E93
//#define PHILOX2X64_10_KEY_INC 0x9E3779B97F4A7C15 //golden ratio

/**
State of philox2x32_10 RNG.
*/
typedef union{
	ulong LR;
	struct{
		uint L, R;
	};
} philox2x32_10_state;

/**
Internal function. calculates philox2x32-10 random number from state and key.

@param state State of the RNG to use.
@param key Key to use.
*/
ulong philox2x32_10(philox2x32_10_state state, uint key){
	uint tmp, L0 = state.L, R0 = state.R;
	for(uint i=0;i<10;i++){
		uint tmp = R0 * PHILOX2X32_10_MULTIPLIER;
		R0 = mul_hi(R0,PHILOX2X32_10_MULTIPLIER) ^ L0 ^ key;
		L0 = tmp;
		key += PHILOX2X32_10_KEY_INC;
	}
	state.L = L0;
	state.R = R0;
	return state.LR;
}

/**
Generates a random 64-bit unsigned integer using philox2x32_10 RNG.

@param state State of the RNG to use.
*/
#define philox2x32_10_ulong(state) _philox2x32_10_ulong(&state)
ulong _philox2x32_10_ulong(philox2x32_10_state *state){
	state->LR++;
	return philox2x32_10(*state, 12345);
}

/**
Seeds philox2x32_10 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void philox2x32_10_seed(philox2x32_10_state *state, ulong j){
	state->LR = j;
}

/**
Generates a random 32-bit unsigned integer using philox2x32_10 RNG.

@param state State of the RNG to use.
*/
#define philox2x32_10_uint(state) ((uint)philox2x32_10_ulong(state))

// Kernel function
// Seed RNG by single ulong
class philox2x32_10_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<philox2x32_10_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		philox2x32_10_seed_by_value_kernel(ulong val,
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
            philox2x32_10_state state;
            philox2x32_10_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class philox2x32_10_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<philox2x32_10_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		philox2x32_10_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            philox2x32_10_state state;
            philox2x32_10_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class philox2x32_10_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<philox2x32_10_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		philox2x32_10_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            philox2x32_10_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= philox2x32_10_uint(state);
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
void PHILOX2X32_10_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 philox2x32_10_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void PHILOX2X32_10_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 philox2x32_10_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void PHILOX2X32_10_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 philox2x32_10_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __PHILOX2X32_10_RNG__
