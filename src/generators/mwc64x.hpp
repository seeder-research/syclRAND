#ifndef __MWC64X_RNG__
#define __MWC64X_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a 64-bit Multiply With Carry generator that returns 32-bit numbers that are xor of lower and upper 32-bit numbers.

http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
*/
#pragma once

#define RNG32

#define mwc64x_FLOAT_MULTI 2.3283064365386963e-10f
#define mwc64x_DOUBLE2_MULTI 2.3283064365386963e-10
#define mwc64x_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of mwc64x RNG.
*/
typedef union {
	ulong xc;
	struct{
		uint x;
		uint c;
	};
} mwc64x_state;

/**
Generates a random 32-bit unsigned integer using mwc64x RNG.

@param state State of the RNG to use.
*/
#define mwc64x_uint(state) _mwc64x_uint(&state)
uint _mwc64x_uint(mwc64x_state *s)
{
	uint res = s->x ^ s->c;
	uint X = s->x;
	uint C = s->c;

	uint Xn=4294883355U*X+C;
	uint carry=(uint)(Xn<C);
	uint Cn=mad_hi(4294883355U,X,carry);

	s->x=Xn;
	s->c=Cn;
	return res;
}

/**
Seeds mwc64x RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void mwc64x_seed(mwc64x_state* state, unsigned long j){
	state->xc=j;
}

// Kernel function
// Seed RNG by single ulong
class mwc64x_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<mwc64x_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		mwc64x_seed_by_value_kernel(ulong val,
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
            mwc64x_state state;
            mwc64x_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class mwc64x_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<mwc64x_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		mwc64x_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            mwc64x_state state;
            mwc64x_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class mwc64x_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<mwc64x_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		mwc64x_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            mwc64x_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= mwc64x_uint(state);
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
void MWC64X_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mwc64x_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void MWC64X_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mwc64x_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void MWC64X_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mwc64x_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __MWC64X_RNG__
