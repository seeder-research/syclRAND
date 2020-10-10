#ifndef __MRG32K3A_RNG__
#define __MRG32K3A_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Thomas Bradley, Parallelisation Techniques for Random Number Generators
// https://www.nag.co.uk/IndustryArticles/gpu_gems_article.pdf

#define MRG32K3A_M1 4294967087
#define MRG32K3A_M2 4294944443
#define MRG32K3A_NORM_DOUBLE (2.3283065498378288e-10) // 1/MRG32K3A_M1
#define MRG32K3A_UINT_NORM (1.000000048661607) // (MRG32K3A_POW32 - 1)/(MRG32K3A_M1 - 1)

/**
State of mrg32k3a RNG.
*/
typedef struct {
   ulong g1[3];
   ulong g2[3];
} mrg32k3a_state;

/**
Generates a random 32-bit unsigned integer using mrg32k3a RNG.

@param state State of the RNG to use.
*/
#define mrg32k3a_uint(state) (uint)_mrg32k3a_ulong(&state)
#define mrg32k3a_ulong(state) _mrg32k3a_ulong(&state)
ulong _mrg32k3a_ulong(mrg32k3a_state* state){

    ulong* g1 = state->g1;
    ulong* g2 = state->g2;
    long p0, p1;
    
    /* component 1 */
    p0 = 1403580 * state->g1[1] - 810728 * state->g1[0];
    p0 %= MRG32K3A_M1;
    if (p0 < 0)
        p0 += MRG32K3A_M1;
    g1[0] = g1[1];
    g1[1] = g1[2];
    g1[2] = p0;

    /* component 2 */
    p1 = 527612 * g2[2] - 1370589 * g2[0];
    p1 %= MRG32K3A_M2;
    if (p1 < 0)
        p1 += MRG32K3A_M2;
    g2[0] = g2[1];
    g2[1] = g2[2];
    g2[2] = p1;

    return (p0 - p1) + (p0 <= p1 ? MRG32K3A_M1 : 0);
}

/**
Seeds mrg32k3a RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void mrg32k3a_seed(mrg32k3a_state* state, ulong j){
    ulong* g1 = state->g1;
    ulong* g2 = state->g2;
    g1[0] = j % MRG32K3A_M1;
    g1[1] = 1;
    g1[2] = 1;
    g2[0] = 1;
    g2[1] = 1;
    g2[2] = 1;
}

// Kernel function
// Seed RNG by single ulong
class mrg32k3a_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<mrg32k3a_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		mrg32k3a_seed_by_value_kernel(ulong val,
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
            mrg32k3a_state state;
            mrg32k3a_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class mrg32k3a_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<mrg32k3a_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		mrg32k3a_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            mrg32k3a_state state;
            mrg32k3a_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class mrg32k3a_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<mrg32k3a_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		mrg32k3a_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            mrg32k3a_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= mrg32k3a_uint(state);
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
void MRG32K3A_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mrg32k3a_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void MRG32K3A_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mrg32k3a_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void MRG32K3A_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 mrg32k3a_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __MRG32K3A_RNG__
