#ifndef __TYCHE_RNG__
#define __TYCHE_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a 512-bit tyche (Well-Equidistributed Long-period Linear) RNG.

S. Neves, F. Araujo, Fast and small nonlinear pseudorandom number generators for computer simulation, in: International Conference on Parallel Processing and Applied Mathematics, Springer, 2011, pp. 92–101.
*/
#pragma once

#define TYCHE_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of tyche RNG.
*/
typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_state;

#define TYCHE_ROT(a,b) (((a) << (b)) | ((a) >> (32 - (b))))

/**
Generates a random 64-bit unsigned integer using tyche RNG.

This is alternative, macro implementation of tyche RNG.

@param state State of the RNG to use.
*/
#define tyche_macro_ulong(state) (tyche_macro_advance(state), state.res)
#define tyche_macro_advance(state) ( \
	state.a += state.b, \
	state.d = TYCHE_ROT(state.d ^ state.a, 16), \
	state.c += state.d, \
	state.b = TYCHE_ROT(state.b ^ state.c, 12), \
	state.a += state.b, \
	state.d = TYCHE_ROT(state.d ^ state.a, 8), \
	state.c += state.d, \
	state.b = TYCHE_ROT(state.b ^ state.c, 7) \
)

/**
Generates a random 64-bit unsigned integer using tyche RNG.

@param state State of the RNG to use.
*/
#define tyche_ulong(state) (tyche_advance(&state), state.res)
void tyche_advance(tyche_state* state){
	state->a += state->b;
	state->d = TYCHE_ROT(state->d ^ state->a, 16);
	state->c += state->d;
	state->b = TYCHE_ROT(state->b ^ state->c, 12);
	state->a += state->b;
	state->d = TYCHE_ROT(state->d ^ state->a, 8);
	state->c += state->d;
	state->b = TYCHE_ROT(state->b ^ state->c, 7);
}

/**
Seeds tyche RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tyche_seed(tyche_state* state, ulong seed){
	state->a = seed >> 32;
	state->b = seed;
	state->c = 2654435769;
	state->d = 1367130551 ^ (get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2)));
	for(uint i=0;i<20;i++){
		tyche_advance(state);
	}
}

/**
Generates a random 32-bit unsigned integer using tyche RNG.

@param state State of the RNG to use.
*/
#define tyche_uint(state) ((uint)tyche_ulong(state))

// Kernel function
// Seed RNG by single ulong
class tyche_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<tyche_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		tyche_seed_by_value_kernel(ulong val,
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
            tyche_state state;
            tyche_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class tyche_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<tyche_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		tyche_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            tyche_state state;
            tyche_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class tyche_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<tyche_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		tyche_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            tyche_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= tyche_uint(state);
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
void TYCHE_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 tyche_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void TYCHE_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 tyche_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void TYCHE_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 tyche_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __TYCHE_RNG__
