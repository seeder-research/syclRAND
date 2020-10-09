#ifndef __LCG6432_RNG__
#define __LCG6432_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a 64-bit Linear Congruential Generator, that returns 32-bit numbers (lcg6432). Not recomended for serious use, as it does not pass BigCrush test.

P. L’ecuyer, Tables of linear congruential generators of different sizes and good lattice structure, Mathematics of Computation of the American Mathematical Society 68 (225) (1999) 249–260.
*/
#pragma once

#define RNG32

#define LCG6432_FLOAT_MULTI 2.3283064365386963e-10f
#define LCG6432_DOUBLE2_MULTI 2.3283064365386963e-10
#define LCG6432_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of lcg6432 RNG.
*/
typedef unsigned long lcg6432_state;

/* LCG6432 class */
class LCG6432_PRNG : _SyCLRAND {
	public:
	    using state_accessor = 
		      sycl::accessor<lcg6432_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using output_accessor = 
			  sycl::accessor<uint, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using seed_accessor = 
		      sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		void seed_by_value(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
		void seed_by_array(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
		void generate_uint(cl::sycl::queue funcQueue, int count, cl::sycl::buffer &dst, size_t gsize, size_t lsize);
	private:
	    state_accessor      stateBuf;
		lcg6432_state        *stateArr;
};

/**
Generates a random 32-bit unsigned integer using lcg6432 RNG.

This is alternative, macro implementation of lcg6432 RNG.

@param state State of the RNG to use.
*/
#define lcg6432_macro_uint(state) ( \
	state = state * 6364136223846793005UL + 0xda3e39cb94b95bdbUL, \
	state>>32 \
)

/**
Generates a random 32-bit unsigned integer using lcg6432 RNG.

@param state State of the RNG to use.
*/
#define lcg6432_uint(state) _lcg6432_uint(&state)
unsigned int _lcg6432_uint(lcg6432_state* state){
	*state = *state * 6364136223846793005UL + 0xda3e39cb94b95bdbUL;
	return *state>>32;
}

/**
Seeds lcg6432 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void lcg6432_seed(lcg6432_state* state, unsigned long j){
	*state=j;
}

// Kernel function
// Seed RNG by single ulong
class lcg6432_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<lcg6432_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		lcg6432_seed_by_value_kernel(ulong val,
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
            lcg6432_state state;
            lcg6432_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class lcg6432_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<lcg6432_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		lcg6432_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            lcg6432_state state;
            lcg6432_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class lcg6432_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<lcg6432_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		lcg6432_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            lcg6432_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= lcg6432_uint(state);
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
void LCG6432_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lcg6432_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void LCG6432_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lcg6432_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void LCG6432_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lcg6432_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __LCG6432_RNG__