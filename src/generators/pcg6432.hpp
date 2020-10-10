#ifndef __PCG6432_RNG__
#define __PCG6432_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a 64-bit Permutated Congruential generator (PCG-XSH-RR).

M. E. O’Neill, Pcg: A family of simple fast space-efficient statistically good algorithms for random number generation, ACM Transactions on Mathematical Software.
*/

#pragma once

#define RNG32

#define PCG6432_FLOAT_MULTI 2.3283064365386963e-10f
#define PCG6432_DOUBLE2_MULTI 2.3283064365386963e-10
#define PCG6432_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of pcg6432 RNG.
*/
typedef unsigned long pcg6432_state;

#define PCG6432_XORSHIFTED(s) ((uint)((((s) >> 18u) ^ (s)) >> 27u))
#define PCG6432_ROT(s) ((s) >> 59u)

#define pcg6432_macro_uint(state) ( \
	state = state * 6364136223846793005UL + 0xda3e39cb94b95bdbUL, \
	(PCG6432_XORSHIFTED(state) >> PCG6432_ROT(state)) | (PCG6432_XORSHIFTED(state) << ((-PCG6432_ROT(state)) & 31)) \
)

/**
Generates a random 32-bit unsigned integer using pcg6432 RNG.

@param state State of the RNG to use.
*/
#define pcg6432_uint(state) _pcg6432_uint(&state)
unsigned int _pcg6432_uint(pcg6432_state* state){
    ulong oldstate = *state;
	*state = oldstate * 6364136223846793005UL + 0xda3e39cb94b95bdbUL;
	uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
Seeds pcg6432 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void pcg6432_seed(pcg6432_state* state, unsigned long j){
	*state=j;
}

// Kernel function
// Seed RNG by single ulong
class pcg6432_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<pcg6432_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		pcg6432_seed_by_value_kernel(ulong val,
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
            pcg6432_state state;
            pcg6432_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class pcg6432_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<pcg6432_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		pcg6432_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            pcg6432_state state;
            pcg6432_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class pcg6432_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<pcg6432_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		pcg6432_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            pcg6432_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= pcg6432_uint(state);
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
void PCG6432_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 pcg6432_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void PCG6432_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 pcg6432_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void PCG6432_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 pcg6432_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __PCG6432_RNG__
