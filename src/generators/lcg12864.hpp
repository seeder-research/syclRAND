#ifndef __LCG12864_RNG__
#define __LCG12864_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a 128-bit Linear Congruential Generator. Returns 64-bit numbers.

P. L’ecuyer, Tables of linear congruential generators of different sizes and good lattice structure, Mathematics of Computation of the American Mathematical Society 68 (225) (1999) 249–260.
*/
#pragma once

#define LCG12864_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define LCG12864_DOUBLE_MULTI 5.4210108624275221700372640e-20

#define LCG12864_MULTI_HIGH 2549297995355413924UL
#define LCG12864_MULTI_LOW 4865540595714422341UL
#define LCG12864_INC_HIGH 6364136223846793005UL
#define LCG12864_INC_LOW 1442695040888963407UL

/**
State of lcg12864 RNG.
*/
typedef struct{
	ulong low, high;
} lcg12864_state;

/**
Generates a random 64-bit unsigned integer using lcg12864 RNG.

@param state State of the RNG to use.
*/
#define lcg12864_ulong(state) ( \
	state.high = state.high * LCG12864_MULTI_LOW + state.low * LCG12864_MULTI_HIGH + mul_hi(state.low, LCG12864_MULTI_LOW), \
	state.low = state.low * LCG12864_MULTI_LOW, \
	state.low += LCG12864_INC_LOW, \
	state.high += state.low < LCG12864_INC_LOW, \
	state.high += LCG12864_INC_HIGH, \
	state.high \
)

/**
Generates a random 64-bit unsigned integer using lcg12864 RNG.

This is alternative implementation of lcg12864 RNG as a function.

@param state State of the RNG to use.
*/
#define lcg12864_func_ulong(state) _lcg12864_func_ulong(&state)
ulong _lcg12864_func_ulong(lcg12864_state* state){
	state->high = state->high * LCG12864_MULTI_LOW + state->low * LCG12864_MULTI_HIGH + mul_hi(state->low, LCG12864_MULTI_LOW);
	state->low = state->low * LCG12864_MULTI_LOW;

	state->low += LCG12864_INC_LOW;
	state->high += state->low < LCG12864_INC_LOW;
	state->high += LCG12864_INC_HIGH;
	return state->high;
}

/**
Seeds lcg12864 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void lcg12864_seed(lcg12864_state* state, ulong j){
	state->low=j;
	state->high=j^0xda3e39cb94b95bdbUL;
}

/**
Generates a random 32-bit unsigned integer using lcg12864 RNG.

@param state State of the RNG to use.
*/
#define lcg12864_uint(state) ((uint)lcg12864_ulong(state))

// Kernel function
// Seed RNG by single ulong
class lcg12864_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<lcg12864_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		lcg12864_seed_by_value_kernel(ulong val,
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
            lcg12864_state state;
            lcg12864_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class lcg12864_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<lcg12864_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		lcg12864_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            lcg12864_state state;
            lcg12864_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class lcg12864_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<lcg12864_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		lcg12864_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            lcg12864_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= lcg12864_uint(state);
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
void LCG12864_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lcg12864_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void LCG12864_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lcg12864_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void LCG12864_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lcg12864_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __LCG12864_RNG__
