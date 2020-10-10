#ifndef __SOBOL32_RNG__
#define __SOBOL32_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

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

// Kernel function
// Seed RNG by single ulong
class sobol32_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<sobol32_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		sobol32_seed_by_value_kernel(ulong val,
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
            sobol32_state state;
            sobol32_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class sobol32_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<sobol32_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		sobol32_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            sobol32_state state;
            sobol32_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class sobol32_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<sobol32_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		sobol32_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            sobol32_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= sobol32_uint(state);
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
void SOBOL32_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 sobol32_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void SOBOL32_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 sobol32_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void SOBOL32_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 sobol32_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __SOBOL32_RNG__
