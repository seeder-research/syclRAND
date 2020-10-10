#ifndef __MSWS_RNG__
#define __MSWS_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements msws (Middle Square Weyl Sequence) RNG.

B. Widynski, Middle square weyl sequence rng, arXiv preprint arXiv:1704.00358. https://arxiv.org/abs/1704.00358
*/
#pragma once

#define MSWS_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define MSWS_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of msws RNG.
*/
typedef struct{
	union{
		ulong x;
		uint2 x2;
	};
	ulong w;
}msws_state;

/**
Generates a random 64-bit unsigned integer using msws RNG.

This is alternative, macro implementation of msws RNG.

@param state State of the RNG to use.
*/
#define msws_macro_ulong(state) (\
	state.x *= state.x, \
	state.x += (state.w += 0xb5ad4eceda1ce2a9), \
	state.x = (state.x>>32) | (state.x<<32) \
	)

/**
Generates a random 64-bit unsigned integer using msws RNG.

@param state State of the RNG to use.
*/
#define msws_ulong(state) _msws_ulong(&state)
ulong _msws_ulong(msws_state* state){
	state->x *= state->x;
	state->x += (state->w += 0xb5ad4eceda1ce2a9);
	return state->x = (state->x>>32) | (state->x<<32);
}

/**
Generates a random 64-bit unsigned integer using msws RNG.

This is alternative implementation of msws RNG, that swaps values instead of using shifts.

@param state State of the RNG to use.
*/
#define msws_swap_ulong(state) _msws_swap_ulong(&state)
ulong _msws_swap_ulong(msws_state* state){
	state->x *= state->x;
	state->x += (state->w += 0xb5ad4eceda1ce2a9);
	/*uint tmp = state->xl;
	state->xl = state->xh;
	state->xh = tmp;*/
	state->x2 = state->x2.yx;
	return state->x;
}
/**
Generates a random 64-bit unsigned integer using msws RNG.

This is second alternative implementation of msws RNG, that swaps values instead of using shifts.

@param state State of the RNG to use.
*/
#define msws_swap2_ulong(state) _msws_swap2_ulong(&state)
ulong _msws_swap2_ulong(msws_state* state){
	state->x *= state->x;
	state->x += (state->w += 0xb5ad4eceda1ce2a9);
	uint tmp = state->x2.x;
	state->x2.x = state->x2.y;
	state->x2.y = tmp;
	return state->x;
}

/**
Seeds msws RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void msws_seed(msws_state* state, ulong j){
	state->x = j;
	state->w = j;
}

/**
Generates a random 32-bit unsigned integer using msws RNG.

@param state State of the RNG to use.
*/
#define msws_uint(state) ((uint)msws_ulong(state))

// Kernel function
// Seed RNG by single ulong
class msws_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<msws_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		msws_seed_by_value_kernel(ulong val,
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
            msws_state state;
            msws_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class msws_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<msws_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		msws_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            msws_state state;
            msws_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class msws_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<msws_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		msws_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            msws_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= msws_uint(state);
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
void MSWS_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 msws_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void MSWS_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 msws_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void MSWS_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 msws_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __MSWS_RNG__
