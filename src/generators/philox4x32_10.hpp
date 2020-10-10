#ifndef __PHILOX4X32_10_RNG__
#define __PHILOX4X32_10_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements philox4x32-10 RNG.

J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, Parallel random numbers: as easy as 1, 2, 3, in: High Performance Computing, Networking, Storage and Analysis (SC), 2011 International Conference for, IEEE, 2011, pp. 1–12.
*/
#pragma once

#define PHILOX4X32_10_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define PHILOX4X32_10_DOUBLE_MULTI 5.4210108624275221700372640e-20

#define PHILOXM4x32_0  0xD2511F53U
#define PHILOXM4x32_1  0xCD9E8D57U
#define PHILOX_W32_0   0x9E3779B9U
#define PHILOX_W32_1   0xBB67AE85U

/**
State of philox4x32_10 RNG.
*/
typedef struct{
	uint4 counter;
	uint4 result;
	uint2 key;
	uint substrate;
} philox4x32_10_state;

inline static
unsigned int mulhilo32(unsigned int x, unsigned int y, unsigned int& z)
{
    unsigned long long xy = mul_hi(x, y);
    z = x * y;
    return (unsigned int)(xy);
}

inline static
uint4 single_round(uint4 counter, uint2 key)
{
    // Source: Random123
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(PHILOXM4x32_0, counter.x, hi0);
    unsigned int lo1 = mulhilo32(PHILOXM4x32_1, counter.z, hi1);
    return uint4 {
        hi1 ^ counter.y ^ key.x,
        lo1,
        hi0 ^ counter.w ^ key.y,
        lo0
    };
}

inline static
uint2 bumpkey(uint2 key)
{
    key.x += PHILOX_W32_0;
    key.y += PHILOX_W32_1;
    return key;
}

inline static
void discard_state(unsigned long long offset){
    unsigned int lo = (unsigned int)(offset);
    unsigned int hi = (unsigned int)(offset >> 32);

    uint4 temp = state.counter;
    state.counter.x += lo;
    state.counter.y += hi + (state.counter.x < temp.x ? 1 : 0);
    state.counter.z += (state.counter.y < temp.y ? 1 : 0);
    state.counter.w += (state.counter.z < temp.z ? 1 : 0);
}

inline static
void discard_impl(unsigned long long offset){
    // Adjust offset for subset
    state.substate += offset & 3;
    offset += state.substate < 4 ? 0 : 4;
    state.substate += state.substate < 4 ? 0 : -4;
    // Discard states
    discard_state(offset / 4);
}

inline static
void discard_subsequence_impl(unsigned long long subsequence){
    unsigned int lo = (unsigned int)(subsequence);
    unsigned int hi = (unsigned int)(subsequence >> 32);

    unsigned int temp = state.counter.z;
    state.counter.z += lo;
    state.counter.w += hi + (state.counter.z < temp ? 1 : 0);
}

inline static
uint4 ten_rounds(uint4 counter, uint2 key)
{
    counter = single_round(counter, key); key = bumpkey(key); // 1
    counter = single_round(counter, key); key = bumpkey(key); // 2
    counter = single_round(counter, key); key = bumpkey(key); // 3
    counter = single_round(counter, key); key = bumpkey(key); // 4
    counter = single_round(counter, key); key = bumpkey(key); // 5
    counter = single_round(counter, key); key = bumpkey(key); // 6
    counter = single_round(counter, key); key = bumpkey(key); // 7
    counter = single_round(counter, key); key = bumpkey(key); // 8
    counter = single_round(counter, key); key = bumpkey(key); // 9
    return single_round(counter, key);                        // 10
}

inline static
void restart(const unsigned long long subsequence,
             const unsigned long long offset) {
    state.counter = {0, 0, 0, 0};
    state.result  = {0, 0, 0, 0};
    state.substate = 0;
    discard_subsequence_impl(subsequence);
    discard_impl(offset);
    state.result = ten_rounds(state.counter, state.key);
}
/**
Seeds philox4x32_10 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void philox4x32_10_seed(philox4x32_10_state *state, ulong j){
    state->key->x = (unsigned int)(j);
    state->key->y = (unsigned int)(j >> 32);
    restart(0, 0);
}

/**
Generates a random 32-bit unsigned integer using philox4x32_10 RNG.

@param state State of the RNG to use.
*/
#define philox4x32_10_uint(state) _philox4x32_10_uint(state)
uint _philox4x32_10_uint(philox4x32_10_state *state){
    unsigned int ret = (&state.result.x)[state.substrate];
    state.substate++;
    if(state.substate == 4) {
        state.substate = 0;
        discard_state();
        state.result = ten_rounds(state.counter, state.key);
    }
    return (uint)(ret);
}

// Kernel function
// Seed RNG by single ulong
class philox4x32_10_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<philox4x32_10_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		philox4x32_10_seed_by_value_kernel(ulong val,
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
            philox4x32_10_state state;
            philox4x32_10_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class philox4x32_10_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<philox4x32_10_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		philox4x32_10_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            philox4x32_10_state state;
            philox4x32_10_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class philox4x32_10_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<philox4x32_10_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		philox4x32_10_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            philox4x32_10_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= philox4x32_10_uint(state);
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
void PHILOX4X32_10_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 philox4x32_10_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void PHILOX4X32_10_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 philox4x32_10_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void PHILOX4X32_10_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 philox4x32_10_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __PHILOX4X32_10_RNG__
