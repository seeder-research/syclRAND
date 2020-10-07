typedef struct TINYMT32WP_T {
    cl_uint s0;
    cl_uint s1;
    cl_uint s2;
    cl_uint s3;
    cl_uint mat1;
    cl_uint mat2;
    cl_uint tmat;
} tinymt32wp_t;

typedef tinymt32wp_t tinymt32_state;

const char * tinymt32_prng_kernel = R"EOK(
/**
@file

Implements RandomCL interface to tinymt32 RNG.

Tiny mersenne twister, http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/index.html.
*/
#pragma once

#define RNG32

#define TINYMT32_FLOAT_MULTI 2.3283064365386963e-10f
#define TINYMT32_DOUBLE2_MULTI 2.3283064365386963e-10
#define TINYMT32_DOUBLE_MULTI 5.4210108624275221700372640e-20

#define KERNEL_PROGRAM
#ifndef TINYMT32_CLH
#define TINYMT32_CLH
/**
 * @file tinymt32.clh
 *
 * @brief Sample Program for openCL 1.2
 *
 * tinymt32
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>127</sup>-1.
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
#ifndef TINYMT32DEF_H
#define TINYMT32DEF_H
/**
 * @file tinymt32def.h
 *
 * @brief Common definitions in host and kernel for 32-bit tinymt.
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
#endif

/**
 * TinyMT32 structure with parameters
 */
typedef struct TINYMT32WP_T {
    cl_uint s0;
    cl_uint s1;
    cl_uint s2;
    cl_uint s3;
    cl_uint mat1;
    cl_uint mat2;
    cl_uint tmat;
} tinymt32wp_t;

/**
 * TinyMT32 structure for jump without parameters
 */
typedef struct TINYMT32J_T {
    cl_uint s0;
    cl_uint s1;
    cl_uint s2;
    cl_uint s3;
} tinymt32j_t;

#define TINYMT32J_MAT1 0x8f7011eeU
#define TINYMT32J_MAT2 0xfc78ff1fU
#define TINYMT32J_TMAT 0x3793fdffU

#endif

#define TINYMT32_SHIFT0 1
#define TINYMT32_SHIFT1 10
#define TINYMT32_SHIFT8 8
#define TINYMT32_MIN_LOOP 8
#define TINYMT32_PRE_LOOP 8

__constant uint tinymt32_mask = 0x7fffffff;
__constant uint tinymt32_float_mask = 0x3f800000;

/**
 * This function generates unsigned 32-bit integers.
 *@param tiny tinymt internal state with parameters
 *@return unsigned 32-bit integer
 */
inline static void
tinymt32_next_state(tinymt32wp_t * tiny)
{
    uint x = (tiny->s0 & tinymt32_mask) ^ tiny->s1 ^ tiny->s2;
    uint y = tiny->s3;
    uint t0, t1;
    x ^= x << TINYMT32_SHIFT0;
    y ^= (y >> TINYMT32_SHIFT0) ^ x;
    tiny->s0 = tiny->s1;
    tiny->s1 = tiny->s2;
    tiny->s2 = x ^ (y << TINYMT32_SHIFT1);
    tiny->s3 = y;
    if (y & 1) {
	tiny->s1 ^= tiny->mat1;
	tiny->s2 ^= tiny->mat2;
    }
}

/**
 * This function generates unsigned 32-bit integers.
 *@param tiny tinymt internal state with parameters
 *@return unsigned 32-bit integer
 */
inline static uint
tinymt32_temper(tinymt32wp_t * tiny)
{
    uint t0, t1;
    t0 = tiny->s3;
    t1 = tiny->s0 + (tiny->s2 >> TINYMT32_SHIFT8);
    t0 ^= t1;
    if (t1 & 1) {
	t0 ^= tiny->tmat;
    }
    return t0;
}

/**
 * This function generates single floating point numbers uniformly
 * distribute in the range [1, 2).
 *@param tiny tinymt internal state with parameters
 *@return single floating point number
 */
inline static float
tinymt32_temper12(tinymt32wp_t * tiny)
{
    uint t0, t1;
    t0 = tiny->s3;
    t1 = tiny->s0 + (tiny->s2 >> TINYMT32_SHIFT8);
    t0 ^= t1;
    if (t1 & 1) {
	t0 = t0 ^ tiny->tmat;
    }
    t0 = (t0 >> 9) | tinymt32_float_mask;
    return as_float(t0);
}

/**
 * This function generates unsigned 32-bit integers.
 *@param tiny tinymt internal state with parameters
 *@return unsigned 32-bit integer
 */
inline static uint
tinymt32_uint32(tinymt32wp_t * tiny)
{
    tinymt32_next_state(tiny);
    return tinymt32_temper(tiny);
}

/**
 * This function generates unsigned 32-bit integers.
 *@param tiny tinymt internal state with parameters
 *@return unsigned 32-bit integer
 */
inline static float
tinymt32_single12(tinymt32wp_t * tiny)
{
    tinymt32_next_state(tiny);
    return tinymt32_temper12(tiny);
}

/**
 * This function generates single floating point numbers uniformly
 * distribute in the range [0, 1).
 *@param tiny tinymt internal state with parameters
 *@return single floating point number
 */
inline static float
tinymt32_single01(tinymt32wp_t * tiny)
{
    return tinymt32_single12(tiny) - 1.0f;
}

/**
 * Internal function
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
inline static uint
tinymt32_ini_func1(uint x)
{
    return (x ^ (x >> 27)) * 1664525U;
}

/**
 * Internal function
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
inline static uint
tinymt32_ini_func2(uint x)
{
    return (x ^ (x >> 27)) * 1566083941U;
}

/**
 * Internal function.
 * This function certificate the period of 2^127-1.
 * @param tiny tinymt state vector.
 */
inline static void
tinymt32_period_certification(tinymt32wp_t * tiny)
{
    if ((tiny->s0 & tinymt32_mask) == 0 &&
        tiny->s1 == 0 &&
        tiny->s2 == 0 &&
        tiny->s3 == 0) {
        tiny->s0 = 'T';
        tiny->s1 = 'I';
        tiny->s2 = 'N';
        tiny->s3 = 'Y';
    }
}

/**
 * This function initializes the internal state array with a 32-bit
 * unsigned integer seed.
 * @param tiny tinymt state vector.
 * @param seed a 32-bit unsigned integer used as a seed.
 */
inline static void
tinymt32_init(tinymt32wp_t * tiny, uint seed)
{
    uint status[4];
    status[0] = seed;
    status[1] = tiny->mat1;
    status[2] = tiny->mat2;
    status[3] = tiny->tmat;
    for (int i = 1; i < TINYMT32_MIN_LOOP; i++) {
        status[i & 3] ^= i + 1812433253U
            * (status[(i - 1) & 3]
               ^ (status[(i - 1) & 3] >> 30));
    }
    tiny->s0 = status[0];
    tiny->s1 = status[1];
    tiny->s2 = status[2];
    tiny->s3 = status[3];
    tinymt32_period_certification(tiny);
    for (int i = 0; i < TINYMT32_PRE_LOOP; i++) {
        tinymt32_uint32(tiny);
    }
}

/**
 * This function initializes the internal state array,
 * with an array of 32-bit unsigned integers used as seeds
 * @param tiny tinymt state vector.
 * @param init_key the array of 32-bit integers, used as a seed.
 * @param key_length the length of init_key.
 */
inline static void
tinymt32_init_by_array(tinymt32wp_t * tiny,
		       uint init_key[],
		       int key_length)
{
    const int lag = 1;
    const int mid = 1;
    const int size = 4;
    int i, j;
    int count;
    uint r;
    uint st[4];

    st[0] = 0;
    st[1] = tiny->mat1;
    st[2] = tiny->mat2;
    st[3] = tiny->tmat;
    if (key_length + 1 > TINYMT32_MIN_LOOP) {
	count = key_length + 1;
    } else {
	count = TINYMT32_MIN_LOOP;
    }
    r = tinymt32_ini_func1(st[0] ^ st[mid % size]
			   ^ st[(size - 1) % size]);
    st[mid % size] += r;
    r += key_length;
    st[(mid + lag) % size] += r;
    st[0] = r;
    count--;
    for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
	r = tinymt32_ini_func1(st[i % size]
			       ^ st[(i + mid) % size]
			       ^ st[(i + size - 1) % size]);
	st[(i + mid) % size] += r;
	r += init_key[j] + i;
	st[(i + mid + lag) % size] += r;
	st[i % size] = r;
	i = (i + 1) % size;
    }
    for (; j < count; j++) {
	r = tinymt32_ini_func1(st[i % size]
		      ^ st[(i + mid) % size]
		      ^ st[(i + size - 1) % size]);
	st[(i + mid) % size] += r;
	r += i;
	st[(i + mid + lag) % size] += r;
	st[i % size] = r;
	i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
	r = tinymt32_ini_func2(st[i % size]
			       + st[(i + mid) % size]
			       + st[(i + size - 1) % size]);
	st[(i + mid) % size] ^= r;
	r -= i;
	st[(i + mid + lag) % size] ^= r;
	st[i % size] = r;
	i = (i + 1) % size;
    }
    tiny->s0 = st[0];
    tiny->s1 = st[1];
    tiny->s2 = st[2];
    tiny->s3 = st[3];
    tinymt32_period_certification(tiny);
    for (i = 0; i < TINYMT32_PRE_LOOP; i++) {
	tinymt32_uint32(tiny);
    }
}

/**
 * Read the internal state vector from kernel I/O data
 * @param tiny tinymt internal state with parameters
 * @param g_status state vectors in global memory
 */
inline static void
tinymt32_status_read(tinymt32wp_t * tiny,
		     __global tinymt32wp_t * g_status)
{
    const size_t id = tinymt_get_sequential_id();
    tiny->s0 = g_status[id].s0;
    tiny->s1 = g_status[id].s1;
    tiny->s2 = g_status[id].s2;
    tiny->s3 = g_status[id].s3;
    tiny->mat1 = g_status[id].mat1;
    tiny->mat2 = g_status[id].mat2;
    tiny->tmat = g_status[id].tmat;
}

/**
 * Write the internal state vector to global memory.
 * @param g_status state vectors in global memory
 * @param tiny tinymt internal state with parameters.
 */
inline static void
tinymt32_status_write(__global tinymt32wp_t * g_status,
		      tinymt32wp_t * tiny)
{
    const size_t id = tinymt_get_sequential_id();
    g_status[id].s0 = tiny->s0;
    g_status[id].s1 = tiny->s1;
    g_status[id].s2 = tiny->s2;
    g_status[id].s3 = tiny->s3;
#if defined(DEBUG)
    g_status[id].mat1 = tiny->mat1;
    g_status[id].mat2 = tiny->mat2;
    g_status[id].tmat = tiny->tmat;
#endif
}

#undef TINYMT32_SHIFT0
#undef TINYMT32_SHIFT1
#undef TINYMT32_MIN_LOOP
#undef TINYMT32_PRE_LOOP

#endif
#undef KERNEL_PROGRAM

/**
State of tinymt32 RNG.
*/
typedef tinymt32wp_t tinymt32_state;

/**
Generates a random 32-bit unsigned integer using tinymt32 RNG.

@param state State of the RNG to use.
*/
#define tinymt32_uint(state) tinymt32_uint32(&state)

//#define tinymt32_seed(state, seed) tinymt32_init(state, seed)


/**
Seeds tinymt32 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tinymt32_seed(tinymt32_state* state, ulong seed){
	state->mat1=TINYMT32J_MAT1;
	state->mat2=TINYMT32J_MAT2;
	state->tmat=TINYMT32J_TMAT;
	tinymt32_init(state, seed);
}


/**
Generates a random 64-bit unsigned integer using tinymt32 RNG.

@param state State of the RNG to use.
*/
#define tinymt32_ulong(state) ((((ulong)tinymt32_uint(state)) << 32) | tinymt32_uint(state))

/**
Generates a random float using tinymt32 RNG.

@param state State of the RNG to use.
*/
#define tinymt32_float(state) (tinymt32_uint(state)*TINYMT32_FLOAT_MULTI)

/**
Generates a random double using tinymt32 RNG.

@param state State of the RNG to use.
*/
#define tinymt32_double(state) (tinymt32_ulong(state)*TINYMT32_DOUBLE_MULTI)

/**
Generates a random double using tinymt32 RNG. Generated using only 32 random bits.

@param state State of the RNG to use.
*/
#define tinymt32_double2(state) (tinymt32_uint(state)*TINYMT32_DOUBLE2_MULTI)
)EOK";
