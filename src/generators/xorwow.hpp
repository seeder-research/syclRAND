#ifndef __XORWOW_RNG__
#define __XORWOW_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

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

// Kernel function
// Seed RNG by single ulong
class xorwow_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<xorwow_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		xorwow_seed_by_value_kernel(ulong val,
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
            xorwow_state state;
            xorwow_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class xorwow_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<xorwow_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		xorwow_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            xorwow_state state;
            xorwow_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class xorwow_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<xorwow_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		xorwow_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            xorwow_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= xorwow_uint(state);
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
void XORWOW_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 xorwow_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void XORWOW_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 xorwow_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void XORWOW_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 xorwow_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __XORWOW_RNG__
