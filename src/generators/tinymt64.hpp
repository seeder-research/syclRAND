#ifndef __TINYMT64_RNG__
#define __TINYMT64_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements RandomCL interface to tinymt64 RNG.

Tiny mersenne twister, http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/index.html.
*/
#pragma once

#define TINYMT64_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TINYMT64_DOUBLE_MULTI 5.4210108624275221700372640e-20

#define KERNEL_PROGRAM
#ifndef TINYMT64_CLH
#define TINYMT64_CLH
/**
 * @file tinymt64.clh
 *
 * @brief Sample Program for openCL 1.2
 *
 * tinymt64
 * This program generates 64-bit unsigned integers.
 * The period of generated integers is 2<sup>127</sup>-1.
 *
 * This also generates single precision floating point numbers
 * uniformly distributed in the range [1, 2). (float r; 1.0 <= r < 2.0)
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2013 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#ifndef TINYMT_CLH
#define TINYMT_CLH
/**
 * @file tinymt.clh
 *
 * @brief Common functions for tinymt on kernel program in openCL 1.2.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2013 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */

/**
 * return unique id in a device.
 * This function may not work correctly in multiple devices.
 * @return unique id in a device
 */
inline static size_t
tinymt_get_sequential_id()
{
    return get_global_id(2) - get_global_offset(2)
        + get_global_size(2) * (get_global_id(1) - get_global_offset(1))
        + get_global_size(1) * get_global_size(2)
        * (get_global_id(0) - get_global_offset(0));
}

/**
 * return number of unique ids in a device.
 * This function may not work correctly in multiple devices.
 * @return number of unique ids in a device
 */
inline static size_t
tinymt_get_sequential_size()
{
    return get_global_size(0) * get_global_size(1) * get_global_size(2);
}

#endif
#ifndef TINYMT64DEF_H
#define TINYMT64DEF_H
/**
 * @file tinymt64def.h
 *
 * @brief Common definitions in host and kernel for 64-bit tinymt.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2013 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */

#if defined(KERNEL_PROGRAM)
#if !defined(cl_uint)
#define cl_uint uint
#endif
#if !defined(cl_ulong)
#define cl_ulong ulong
#endif
#if !defined(UINT64_X)
#define UINT64_C(x) (x ## UL)
#endif
#endif

/**
 * TinyMT32 structure with parameters
 */
typedef struct TINYMT64WP_T {
    cl_ulong s0;
    cl_ulong s1;
    cl_uint mat1;
    cl_uint mat2;
    cl_ulong tmat;
} tinymt64wp_t;

/**
 * TinyMT32 structure for jump without parameters
 */
typedef struct TINYMT64J_T {
    cl_ulong s0;
    cl_ulong s1;
} tinymt64j_t;

#define TINYMT64J_MAT1 0xfa051f40U
#define TINYMT64J_MAT2 0xffd0fff4U;
#define TINYMT64J_TMAT UINT64_C(0x58d02ffeffbfffbc)

#endif

#define TINYMT64_SHIFT0 12
#define TINYMT64_SHIFT1 11
#define TINYMT64_MIN_LOOP 8

__constant ulong tinymt64_mask = 0x7fffffffffffffffUL;
__constant ulong tinymt64_double_mask = 0x3ff0000000000000UL;

#if defined(HAVE_DOUBLE)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 */
inline static void
tinymt64_next_state(tinymt64wp_t * tiny)
{
    ulong x;

    tiny->s0 &= tinymt64_mask;
    x = tiny->s0 ^ tiny->s1;
    x ^= x << TINYMT64_SHIFT0;
    x ^= x >> 32;
    x ^= x << 32;
    x ^= x << TINYMT64_SHIFT1;
    tiny->s0 = tiny->s1;
    tiny->s1 = x;
    if (x & 1) {
        tiny->s0 ^= tiny->mat1;
        tiny->s1 ^= (ulong)tiny->mat2 << 32;
    }
}

/**
 * tempering output function
 *@param tiny internal state of tinymt with parameter
 *@return tempered output
 */
inline static ulong
tinymt64_temper(tinymt64wp_t * tiny)
{
    ulong x;
    x = tiny->s0 + tiny->s1;
    x ^= tiny->s0 >> 8;
    if (x & 1) {
        x ^= tiny->tmat;
    }
    return x;
}
/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 *@return 32-bit random integer
 */
inline static ulong
tinymt64_uint64(tinymt64wp_t * tiny)
{
    tinymt64_next_state(tiny);
    return tinymt64_temper(tiny);
}

#if defined(HAVE_DOUBLE)
/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 *@return random number uniformly distributes in the range [1, 2)
 */
inline static double
tinymt64_double12(tinymt64wp_t * tiny)
{
    ulong x = tinymt64_uint64(tiny);
    x = (x >> 12) ^ tinymt64_double_mask;
    return as_double(x);
}

/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 *@return random number uniformly distributes in the range [0, 1)
 */
inline static double
tinymt64_double01(tinymt64wp_t * tiny)
{
    return tinymt64_double12(tiny) - 1.0;
}
#endif

/**
 * Internal function
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 64-bit integer
 * @return 64-bit integer
 */
inline static ulong
tinymt64_ini_func1(ulong x)
{
    return (x ^ (x >> 59)) * 2173292883993UL;
}

/**
 * Internal function
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 64-bit integer
 * @return 64-bit integer
 */
inline static ulong
tinymt64_ini_func2(ulong x)
{
    return (x ^ (x >> 59)) * 58885565329898161UL;
}

/**
 * Internal function.
 * This function certificate the period of 2^127-1.
 * @param tiny tinymt state vector.
 */
inline static void
tinymt64_period_certification(tinymt64wp_t * tiny)
{
    if ((tiny->s0 & tinymt64_mask) == 0 &&
        tiny->s1 == 0) {
        tiny->s0 = 'T';
        tiny->s1 = 'M';
    }
}

/**
 * This function initializes the internal state array with a 64-bit
 * unsigned integer seed.
 * @param tiny tinymt state vector.
 * @param seed a 64-bit unsigned integer used as a seed.
 */
inline static void
tinymt64_init(tinymt64wp_t * tiny, ulong seed)
{
    ulong status[2];
    status[0] = seed ^ ((ulong)tiny->mat1 << 32);
    status[1] = tiny->mat2 ^ tiny->tmat;
    for (int i = 1; i < TINYMT64_MIN_LOOP; i++) {
        status[i & 1] ^= i + 6364136223846793005UL
            * (status[(i - 1) & 1] ^ (status[(i - 1) & 1] >> 62));
    }
    tiny->s0 = status[0];
    tiny->s1 = status[1];
    tinymt64_period_certification(tiny);
}

/**
 * This function initializes the internal state array,
 * with an array of 64-bit unsigned integers used as seeds
 * @param tiny tinymt state vector.
 * @param init_key the array of 64-bit integers, used as a seed.
 * @param key_length the length of init_key.
 */
inline static void
tinymt64_init_by_array(tinymt64wp_t * tiny,
                       ulong init_key[],
                       int key_length)
{
    const int lag = 1;
    const int mid = 1;
    const int size = 4;
    int i, j;
    int count;
    ulong r;
    ulong st[4];

    st[0] = 0;
    st[1] = tiny->mat1;
    st[2] = tiny->mat2;
    st[3] = tiny->tmat;
    if (key_length + 1 > TINYMT64_MIN_LOOP) {
        count = key_length + 1;
    } else {
        count = TINYMT64_MIN_LOOP;
    }
    r = tinymt64_ini_func1(st[0] ^ st[mid % size]
                           ^ st[(size - 1) % size]);
    st[mid % size] += r;
    r += key_length;
    st[(mid + lag) % size] += r;
    st[0] = r;
    count--;
    for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
        r = tinymt64_ini_func1(st[i % size]
                               ^ st[(i + mid) % size]
                               ^ st[(i + size - 1) % size]);
        st[(i + mid) % size] += r;
        r += init_key[j] + i;
        st[(i + mid + lag) % size] += r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    for (; j < count; j++) {
        r = tinymt64_ini_func1(st[i % size]
                      ^ st[(i + mid) % size]
                      ^ st[(i + size - 1) % size]);
        st[(i + mid) % size] += r;
        r += i;
        st[(i + mid + lag) % size] += r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
        r = tinymt64_ini_func2(st[i % size]
                               + st[(i + mid) % size]
                               + st[(i + size - 1) % size]);
        st[(i + mid) % size] ^= r;
        r -= i;
        st[(i + mid + lag) % size] ^= r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    tiny->s0 = st[0] ^ st[1];
    tiny->s1 = st[2] ^ st[3];
    tinymt64_period_certification(tiny);
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 */
inline static void
tinymt64_status_read(tinymt64wp_t * tiny,
                     __global tinymt64wp_t * g_status)
{
    const size_t id = tinymt_get_sequential_id();
    tiny->s0 = g_status[id].s0;
    tiny->s1 = g_status[id].s1;
    tiny->mat1 = g_status[id].mat1;
    tiny->mat2 = g_status[id].mat2;
    tiny->tmat = g_status[id].tmat;
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 */
inline static void
tinymt64_status_write(__global tinymt64wp_t * g_status,
                      tinymt64wp_t * tiny)
{
    const size_t id = tinymt_get_sequential_id();
    g_status[id].s0 = tiny->s0;
    g_status[id].s1 = tiny->s1;
#if defined(DEBUG)
    g_status[id].mat1 = tiny->mat1;
    g_status[id].mat2 = tiny->mat2;
    g_status[id].tmat = tiny->tmat;
#endif
}

#undef TINYMT64_SHIFT0
#undef TINYMT64_SHIFT1
#undef TINYMT64_MIN_LOOP

#endif
#undef KERNEL_PROGRAM

/**
State of tinymt64 RNG.
*/
typedef tinymt64wp_t tinymt64_state;


/**
Generates a random 64-bit unsigned integer using tinymt64 RNG.

@param state State of the RNG to use.
*/
#define tinymt64_ulong(state) tinymt64_uint64(&state)

//#define tinymt64_seed(state, seed) tinymt64_init(state, seed)

/**
Seeds tinymt64 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tinymt64_seed(tinymt64_state* state, ulong seed){
	state->mat1=TINYMT64J_MAT1;
	state->mat2=TINYMT64J_MAT2;
	state->tmat=TINYMT64J_TMAT;
	tinymt64_init(state, seed);
}

/**
Generates a random 32-bit unsigned integer using tinymt64 RNG.

@param state State of the RNG to use.
*/
#define tinymt64_uint(state) ((uint)tinymt64_ulong(state))

// Kernel function
// Seed RNG by single ulong
class tinymt64_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<tinymt64_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		tinymt64_seed_by_value_kernel(ulong val,
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
            tinymt64_state state;
            tinymt64_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class tinymt64_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<tinymt64_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		tinymt64_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            tinymt64_state state;
            tinymt64_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class tinymt64_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<tinymt64_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		tinymt64_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            tinymt64_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= tinymt64_uint(state);
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
void TINYMT64_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 tinymt64_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void TINYMT64_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 tinymt64_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void TINYMT64_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 tinymt64_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __TINYMT64_RNG__
