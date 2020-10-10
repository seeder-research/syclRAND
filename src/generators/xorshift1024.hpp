#ifndef __XORSHIFT1024_RNG__
#define __XORSHIFT1024_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements 1024-bit xorshift generator. State is shared between 32 threads. As it uses barriers,
all threads of a work group must call the generator at the same time, even if they do not require the
result. In `localRNGs.h` header is the function `RNGLocal::xorshift1024_local_mem` that calculates required
state size given local size. See "examplePrintLocal".

M. Manssen, M. Weigel, A. K. Hartmann, Random number generators for massively parallel simulations on GPU, The European Physical Journal-Special Topics 210 (1) (2012) 53–71.
*/

#pragma once
#define RNG_LOCAL

#define XORSHIFT1024_FLOAT_MULTI 2.3283064365386962890625e-10f
#define XORSHIFT1024_DOUBLE2_MULTI 2.3283064365386962890625e-10
#define XORSHIFT1024_DOUBLE_MULTI 5.4210108624275221700372640e-20

#define XORSHIFT1024_WARPSIZE 32
#define XORSHIFT1024_WORD 32
#define XORSHIFT1024_WORDSHIFT 10
#define XORSHIFT1024_RAND_A 9
#define XORSHIFT1024_RAND_B 27
#define XORSHIFT1024_RAND_C 24

/**
State of xorshift1024 RNG.
*/
typedef uint xorshift1024_state;

/**
generates a random 32-bit unsigned integer using xorshift1024 RNG.

@param stateblock pointer to buffer in local memory, that holds state of the generator.
*/
uint xorshift1024_uint(local xorshift1024_state* stateblock){
	/* Indices. */
	int tid = get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2));
	int wid = tid / XORSHIFT1024_WARPSIZE; // Warp index in block
	int lid = tid % XORSHIFT1024_WARPSIZE; // Thread index in warp
	int woff = wid * (XORSHIFT1024_WARPSIZE + XORSHIFT1024_WORDSHIFT + 1) + XORSHIFT1024_WORDSHIFT + 1;
	// warp offset
	/* Shifted indices. */
	int lp = lid + XORSHIFT1024_WORDSHIFT; // Left word shift
	int lm = lid - XORSHIFT1024_WORDSHIFT; // Right word shift

	uint state;

	/* << A. */
	state = stateblock[woff + lid]; // Read states
	state ^= stateblock[woff + lp] << XORSHIFT1024_RAND_A; // Left part
	state ^= stateblock[woff + lp + 1] >> (XORSHIFT1024_WORD - XORSHIFT1024_RAND_A); // Right part
	barrier(CLK_LOCAL_MEM_FENCE);

	/* >> B. */
	stateblock[woff + lid] = state; // Share states
	barrier(CLK_LOCAL_MEM_FENCE);
	state ^= stateblock[woff + lm - 1] << (XORSHIFT1024_WORD - XORSHIFT1024_RAND_B); // Left part
	state ^= stateblock[woff + lm] >> XORSHIFT1024_RAND_B; // Right part
	barrier(CLK_LOCAL_MEM_FENCE);

	/* << C. */
	stateblock[woff + lid] = state; // Share states
	barrier(CLK_LOCAL_MEM_FENCE);
	state ^= stateblock[woff + lp] << XORSHIFT1024_RAND_C; // Left part
	state ^= stateblock[woff + lp + 1] >> (XORSHIFT1024_WORD - XORSHIFT1024_RAND_C); // Right part
	barrier(CLK_LOCAL_MEM_FENCE);

	stateblock[woff + lid] = state; // Share states
	barrier(CLK_LOCAL_MEM_FENCE);

	return state;
}

/**
Seeds xorshift1024 RNG

@param stateblock Buffer in local memory, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void xorshift1024_seed(local xorshift1024_state* stateblock, ulong seed){
	int tid = get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2));
	int wid = tid / XORSHIFT1024_WARPSIZE; // Warp index in block
	int lid = tid % XORSHIFT1024_WARPSIZE; // Thread index in warp
	int woff = wid * (XORSHIFT1024_WARPSIZE + XORSHIFT1024_WORDSHIFT + 1) + XORSHIFT1024_WORDSHIFT + 1;
	//printf("tid: %d, lid %d, wid %d, woff %d \n", tid, (uint)get_local_id(0), wid, woff);

	uint mem = (XORSHIFT1024_WARPSIZE + XORSHIFT1024_WORDSHIFT + 1) * (get_local_size(0) * get_local_size(1) * get_local_size(2) / XORSHIFT1024_WARPSIZE) + XORSHIFT1024_WORDSHIFT + 1;

	if(lid==13 && (uint)seed==0){ //shouldnt be seeded with all zeroes in wrap, but such check is simpler
		seed=1;
	}

	if(lid<XORSHIFT1024_WORDSHIFT + 1){
		//printf("%d setting %d to 0\n",(uint)get_global_id(0), woff - XORSHIFT1024_WORDSHIFT - 1 + lid);
		stateblock[woff - XORSHIFT1024_WORDSHIFT - 1 + lid] = 0;
	}
	if(tid<XORSHIFT1024_WORDSHIFT + 1){
		//printf("%d setting2 %d to 0\n",(uint)get_global_id(0), mem - 1 - tid);
		stateblock[mem - 1 - tid] = 0;
	}
	stateblock[woff + lid] = (uint)seed;
	//printf("%d seed set\n",(uint)get_local_id(0));
	barrier(CLK_LOCAL_MEM_FENCE);
	//printf("%d after barrier\n",(uint)get_local_id(0));
}

// Kernel function
// Seed RNG by single ulong
class xorshift1024_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<xorshift1024_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		xorshift1024_seed_by_value_kernel(ulong val,
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
            xorshift1024_state state;
            xorshift1024_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class xorshift1024_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<xorshift1024_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		xorshift1024_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            xorshift1024_state state;
            xorshift1024_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class xorshift1024_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<xorshift1024_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		xorshift1024_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            xorshift1024_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= xorshift1024_uint(state);
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
		                 xorshift1024_seed_by_value_kernel(seedVal, state_acc));
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
		                 xorshift1024_seed_by_array_kernel(seed_acc, state_acc));
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
		                 xorshift1024_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __XORSHIFT1024_RNG__
