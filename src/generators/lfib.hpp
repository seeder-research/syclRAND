#ifndef __LFIB_RNG__
#define __LFIB_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a Multiplicative Lagged Fibbonaci generator. Returns 64-bit random numbers, but the lowest bit is always 1.

G. Marsaglia, L.-H. Tsay, Matrices and the structure of random number sequences, Linear algebra and its applications 67 (1985) 147–156.
*/
#pragma once

#define LFIB_LAG1 17
#define LFIB_LAG2 5

/**
State of lfib RNG.
*/
typedef struct{
	ulong s[LFIB_LAG1];
	char p1,p2;
}lfib_state;

/**
Generates a random 64-bit unsigned integer using lfib RNG.

This is alternative, macro implementation of lfib RNG using ternary operators instead of if statements.

@param state State of the RNG to use.
*/
#define lfib_macro_ulong(state) ( \
	state.p1 = --state.p1 >= 0 ? state.p1 : LFIB_LAG1 - 1, \
	state.p2 = --state.p2 >= 0 ? state.p2 : LFIB_LAG1 - 1, \
	state.s[state.p1]*=state.s[state.p2], \
	state.s[state.p1] \
)

/**
Generates a random 64-bit unsigned integer using lfib RNG.

This is alternative implementation of lfib RNG using ternary operators instead of if statements.

@param state State of the RNG to use.
*/
#define lfib_ternary_ulong(state) _lfib_ternary_ulong(&state)
ulong _lfib_ternary_ulong(lfib_state* state){
	/*state->p1++;
	state->p1%=LFIB_LAG1;
	state->p2++;
	state->p2%=LFIB_LAG2;*/
	state->p1 = --state->p1 >= 0 ? state->p1 : LFIB_LAG1 - 1;
	state->p2 = --state->p2 >= 0 ? state->p2 : LFIB_LAG1 - 1;
	state->s[state->p1] *= state->s[state->p2];
	return state->s[state->p1];
}

/**
Generates a random 64-bit unsigned integer using lfib RNG.

@param state State of the RNG to use.
*/
#define lfib_ulong(state) _lfib_ulong(&state)
ulong _lfib_ulong(lfib_state* state){
	/*state->p1++;
	state->p1%=LFIB_LAG1;
	state->p2++;
	state->p2%=LFIB_LAG2;*/
	state->p1--;
	if(state->p1<0) state->p1=LFIB_LAG1-1;
	state->p2--;
	if(state->p2<0) state->p2=LFIB_LAG1-1;
	state->s[state->p1]*=state->s[state->p2];
	return state->s[state->p1];
}

/**
Generates a random 64-bit unsigned integer using lfib RNG.

This is alternative implementation of lfib RNG using modulo instead of conditionals.

@param state State of the RNG to use.
*/
#define lfib_inc_ulong(state) _lfib_inc_ulong(&state)
ulong _lfib_inc_ulong(lfib_state* state){
	state->p1++;
	state->p1%=LFIB_LAG1;
	state->p2++;
	state->p2%=LFIB_LAG2;
	state->s[state->p1]*=state->s[state->p2];
	return state->s[state->p1];
}
/**
Generates a random 64-bit unsigned integer using lfib RNG.

This is alternative, macro implementation of lfib RNG using modulo instead of conditionals.

@param state State of the RNG to use.
*/
#define lfib_inc_macro_ulong(state) ( \
	state.p1++, \
	state.p1%=LFIB_LAG1, \
	state.p2++, \
	state.p2%=LFIB_LAG2, \
	state.s[state.p1]*=state.s[state.p2], \
	state.s[state.p1] \
)

/**
Seeds lfib RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void lfib_seed(lfib_state* state, ulong j){
	state->p1=LFIB_LAG1;
	state->p2=LFIB_LAG2;
	//if(get_global_id(0)==0) printf("seed %d\n",state->p1);
    for (int i = 0; i < LFIB_LAG1; i++){
		j=6906969069UL * j + 1234567UL; //LCG
		state->s[i] = j | 1; // values must be odd
	}
}

/**
Generates a random 32-bit unsigned integer using lfib RNG.

@param state State of the RNG to use.
*/
#define lfib_uint(state) ((uint)(lfib_ulong(state)>>1))

// Kernel function
// Seed RNG by single ulong
class lfib_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<lfib_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		lfib_seed_by_value_kernel(ulong val,
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
            lfib_state state;
            lfib_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class lfib_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<lfib_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		lfib_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            lfib_state state;
            lfib_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class lfib_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<lfib_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		lfib_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            lfib_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= lfib_uint(state);
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
void LFIB_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lfib_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void LFIB_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lfib_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void LFIB_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 lfib_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __LFIB_RNG__
