#ifndef __RAN2_RNG__
#define __RAN2_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a ran2 RNG.

W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, Numerical recipes in c: The art of scientific computing (; cambridge (1992).
*/
#pragma once

#define RNG32

#define RAN2_FLOAT_MULTI 4.6566128752457969230960e-10f
#define RAN2_DOUBLE2_MULTI 4.6566128752457969230960e-10
#define RAN2_DOUBLE_MULTI 2.1684043469904927853807e-19

#define   IM1 2147483563
#define   IM2 2147483399
#define   AM (1.0/IM1)
#define   IMM1  (IM1-1)
#define   IA1 40014
#define   IA2 40692
#define   IQ1 53668
#define   IQ2 52774
#define   IR1 12211
#define   IR2 3791
#define   NTAB 32
#define   NDIV (1+IMM1/NTAB)
#define   EPS 1.2e-7
#define   RNMX (1.0-EPS)

/**
State of ran2 RNG.
*/
typedef struct{
	int idum;
	int idum2;
	int iy;
	int iv[NTAB];
} ran2_state;

/**
Generates a random 32-bit unsigned integer using ran2 RNG. The lowest bit is always 0.

@param state State of the RNG to use.
*/
#define ran2_uint(state) (_ran2_uint(&state)<<1)
ulong _ran2_uint(ran2_state* state){

	int k = state->idum / IQ1;
	state->idum = IA1 * (state->idum - k*IQ1) - k*IR1;
	if(state->idum < 0){
		state->idum += IM1;
	}

	k = state->idum2 / IQ2;
	state->idum2 = IA2 * (state->idum2 - k*IQ2) - k*IR2;
	if(state->idum2 < 0){
		state->idum2 += IM2;
	}

	short j = state->iy / NDIV;
	state->iy = state->iv[j] - state->idum2;
	state->iv[j] = state->idum;
	if(state->iy < 1){
		state->iy += IMM1;
	}
	return state->iy;
	/*float temp = AM * state->iy;
	if(temp > RNMX){
		return RNMX;
	}
	else {
		return temp;
	}*/
}

/**
Generates a random 64-bit unsigned integer using ran2 RNG. The highest bit is always 0.

@param state State of the RNG to use.
*/
#define ran2_unshifted_uint(state) _ran2_uint(&state)

/**
Seeds ran2 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void ran2_seed(ran2_state* state, ulong seed){
	if(seed == 0){
		seed = 1;
	}
	state->idum = seed;
	state->idum2 = seed>>32;
	for(int j = NTAB + 7; j >= 0; j--){
		short k = state->idum / IQ1;
		state->idum = IA1 * (state->idum - k*IQ1) - k*IR1;
		if(state->idum < 0){
			state->idum += IM1;
		}
		if(j < NTAB){
			state->iv[j] = state->idum;
		}
	}
	state->iy = state->iv[0];
}

// Kernel function
// Seed RNG by single ulong
class ran2_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<ran2_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		ran2_seed_by_value_kernel(ulong val,
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
            ran2_state state;
            ran2_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class ran2_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<ran2_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		ran2_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            ran2_state state;
            ran2_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class ran2_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<ran2_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		ran2_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            ran2_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= ran2_uint(state);
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
void RAN2_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 ran2_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void RAN2_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 ran2_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void RAN2_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 ran2_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __RAN2_RNG__
