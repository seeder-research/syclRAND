#ifndef __MRG31K3P_RNG__
#define __MRG31K3P_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements mrg31k3p RNG.

P. L’Ecuyer, R. Touzin, Fast combined multiple recursive generators with multipliers of the form a=+-2 q+-2 r, in: Proceedings of the 32nd conference on Winter simulation, Society for Computer Simulation International, 2000, pp. 683–689.
*/
#pragma once

#define RNG32

#define MRG31K3P_FLOAT_MULTI 4.6566128752457969230960e-10f
#define MRG31K3P_DOUBLE2_MULTI 4.6566128752457969230960e-10
#define MRG31K3P_DOUBLE_MULTI 2.1684043469904927853807e-19

#define MRG31K3P_M1 2147483647
#define MRG31K3P_M2 2147462579
#define MRG31K3P_MASK12 511
#define MRG31K3P_MASK13 16777215
#define MRG31K3P_MRG31K3P_MASK13 65535

/**
State of mrg31k3p RNG.
*/
typedef struct{
	uint x10, x11, x12, x20, x21, x22;
} mrg31k3p_state;

/**
Generates a random 32-bit unsigned integer using mrg31k3p RNG.

@param state State of the RNG to use.
*/
#define mrg31k3p_uint(state) _mrg31k3p_uint(&state)
inline
uint _mrg31k3p_uint(mrg31k3p_state* state){
	uint y1, y2;
	//first component
	y1 = (((state->x11 & MRG31K3P_MASK12) << 22) + (state->x11 >> 9)) + (((state->x12 & MRG31K3P_MASK13) << 7) + (state->x12 >> 24));
	if (y1 > MRG31K3P_M1){
		y1 -= MRG31K3P_M1;
	}
	y1 += state->x12;
	if (y1 > MRG31K3P_M1){
		y1 -= MRG31K3P_M1;
	}
	state->x12 = state->x11;
	state->x11 = state->x10;
	state->x10 = y1;
	//second component
	y1 = ((state->x20 & MRG31K3P_MRG31K3P_MASK13) << 15) + 21069 * (state->x20 >> 16);
	if (y1 > MRG31K3P_M2){
		y1 -= MRG31K3P_M2;
	}
	y2 = ((state->x22 & MRG31K3P_MRG31K3P_MASK13) << 15) + 21069 * (state->x22 >> 16);
	if (y2 > MRG31K3P_M2){
		y2 -= MRG31K3P_M2;
	}
	y2 += state->x22;
	if (y2 > MRG31K3P_M2){
		y2 -= MRG31K3P_M2;
	}
	y2 += y1;
	if (y2 > MRG31K3P_M2){
		y2 -= MRG31K3P_M2;
	}
	state->x22 = state->x21;
	state->x21 = state->x20;
	state->x20 = y2;
	//combining the result
	if (state->x10 <= state->x20){
		return state->x10 - state->x20 + MRG31K3P_M1;
	}
	else{
		return state->x10 - state->x20;
	}
}

/**
Seeds mrg31k3p RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
inline
void mrg31k3p_seed(mrg31k3p_state* state, ulong j){
	state->x10 = (uint)j;
	state->x11 = (uint)(j >> 5);
	state->x12 = (uint)(j >> 11);
	state->x20 = (uint)(j >> 22);
	state->x21 = (uint)(j >> 30);
	state->x22 = (uint)(j >> 33);
	if(j == 0){
		state->x10++;
		state->x21++;
	}
	if (state->x10 > MRG31K3P_M1) state->x10 -= MRG31K3P_M1;
	if (state->x11 > MRG31K3P_M1) state->x11 -= MRG31K3P_M1;
	if (state->x12 > MRG31K3P_M1) state->x12 -= MRG31K3P_M1;

	if (state->x20 > MRG31K3P_M2) state->x20 -= MRG31K3P_M2;
	if (state->x21 > MRG31K3P_M2) state->x21 -= MRG31K3P_M2;
	if (state->x22 > MRG31K3P_M2) state->x22 -= MRG31K3P_M2;

	if (state->x10 > MRG31K3P_M1) state->x10 -= MRG31K3P_M1;
	if (state->x11 > MRG31K3P_M1) state->x11 -= MRG31K3P_M1;
	if (state->x12 > MRG31K3P_M1) state->x12 -= MRG31K3P_M1;

	if (state->x20 > MRG31K3P_M2) state->x20 -= MRG31K3P_M2;
	if (state->x21 > MRG31K3P_M2) state->x21 -= MRG31K3P_M2;
	if (state->x22 > MRG31K3P_M2) state->x22 -= MRG31K3P_M2;
}

/**
Generates a random 64-bit unsigned integer using mrg31k3p RNG.

@param state State of the RNG to use.
*/
#define mrg31k3p_ulong(state) ((((ulong)mrg31k3p_uint(state)) << 32) | mrg31k3p_uint(state))

// Kernel function
// Seed RNG by single ulong
class mrg31k3p_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<mrg31k3p_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		mrg31k3p_seed_by_value_kernel(ulong val,
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
            mrg31k3p_state state;
            mrg31k3p_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class mrg31k3p_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<mrg31k3p_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		mrg31k3p_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            mrg31k3p_state state;
            mrg31k3p_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class mrg31k3p_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<mrg31k3p_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		mrg31k3p_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            mrg31k3p_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= mrg31k3p_uint(state);
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
void MRG31K3P_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mrg31k3p_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void MRG31K3P_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mrg31k3p_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void MRG31K3P_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mrg31k3p_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __MRG31K3P_RNG__
