#ifndef __XORSHIFT6432STAR_RNG__
#define __XORSHIFT6432STAR_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a 64-bit xorshift* generator that returns 32-bit values.

S. Vigna, An experimental exploration of marsaglia’s xorshift generators, scrambled, ACM Transactions on Mathematical Software (TOMS) 42 (4) (2016) 30.
*/
#pragma once
#define RNG32

#define XORSHIFT6432STAR_FLOAT_MULTI 2.3283064365386963e-10f
#define XORSHIFT6432STAR_DOUBLE2_MULTI 2.3283064365386963e-10
#define XORSHIFT6432STAR_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of xorshift6432star RNG.
*/
typedef unsigned long xorshift6432star_state;

#define xorshift6432star_macro_uint(state) (\
	state ^= state >> 12, \
	state ^= state << 25, \
	state ^= state >> 27, \
	(uint)((state*0x2545F4914F6CDD1D)>>32) \
	)

/**
Generates a random 32-bit unsigned integer using xorshift6432star RNG.

@param state State of the RNG to use.
*/
#define xorshift6432star_uint(state) _xorshift6432star_uint(&state)
unsigned int _xorshift6432star_uint(xorshift6432star_state* restrict state){
	*state ^= *state >> 12; // a
	*state ^= *state << 25; // b
	*state ^= *state >> 27; // c
	return (uint)((*state*0x2545F4914F6CDD1D)>>32);
}

/**
Seeds xorshift6432star RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void xorshift6432star_seed(xorshift6432star_state* state, unsigned long j){
	if(j==0){
		j++;
	}
	*state=j;
}

// Kernel function
// Seed RNG by single ulong
class xorshift6432star_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<xorshift6432star_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		xorshift6432star_seed_by_value_kernel(ulong val,
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
            xorshift6432star_state state;
            xorshift6432star_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class xorshift6432star_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<xorshift6432star_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		xorshift6432star_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            xorshift6432star_state state;
            xorshift6432star_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class xorshift6432star_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<xorshift6432star_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		xorshift6432star_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            xorshift6432star_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= xorshift6432star_uint(state);
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
void XORSHIFT6432STAR_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 xorshift6432star_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void XORSHIFT6432STAR_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 xorshift6432star_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void XORSHIFT6432STAR_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 xorshift6432star_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __XORSHIFT6432STAR_RNG__
