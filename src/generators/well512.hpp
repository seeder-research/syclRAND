#ifndef __WELL512_RNG__
#define __WELL512_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements a 512-bit WELL (Well-Equidistributed Long-period Linear) RNG.

F. Panneton, P. L’ecuyer, M. Matsumoto, Improved long-period generators based on linear recurrences modulo 2, ACM Transactions on Mathematical Software (TOMS) 32 (1) (2006) 1–16.
*/

/* ind(mm,x) is bits 2..9 of x, or (floor(x/4) mod 256)*4 */
#pragma once
#define RNG32

#define WELL512_FLOAT_MULTI 2.3283064365386963e-10f
#define WELL512_DOUBLE2_MULTI 2.3283064365386963e-10
#define WELL512_DOUBLE_MULTI 5.4210108624275221700372640e-20

#define W 32
#define R 16
#define P 0
#define M1 13
#define M2 9
#define M3 5

/**
State of WELL RNG.
*/
typedef struct{
	unsigned int s[R];
	unsigned int i;
}well512_state;

#define MAT0POS(t,v) (v^(v>>t))
#define MAT0NEG(t,v) (v^(v<<(-(t))))
#define MAT3NEG(t,v) (v<<(-(t)))
#define MAT4NEG(t,b,v) (v ^ ((v<<(-(t))) & b))

#define V0_(state)            state.s[state.i                   ]
#define VM1_(state)           state.s[(state.i+M1) & 0x0000000fU]
#define VM2_(state)           state.s[(state.i+M2) & 0x0000000fU]
#define VM3_(state)           state.s[(state.i+M3) & 0x0000000fU]
#define VRm1_(state)          state.s[(state.i+15) & 0x0000000fU]
#define VRm2_(state)          state.s[(state.i+14) & 0x0000000fU]
#define newV0_(state)         state.s[(state.i+15) & 0x0000000fU]
#define newV1_(state)         state.s[state.i                   ]
#define newVRm1_(state)       state.s[(state.i+14) & 0x0000000fU]

#define WELL512MACRO_z0(state) VRm1_(state)
#define WELL512MACRO_z1(state) (MAT0NEG(-16,V0_(state)) ^ MAT0NEG(-15, VM1_(state)))
#define WELL512MACRO_z2(state) (MAT0POS(11, VM2_(state)))
/**
Generates a random 32-bit unsigned integer using WELL RNG.

This is alternative, macro implementation of WELL RNG.

@param state State of the RNG to use.
*/
#define well512_macro_uint(state) (\
	newV1_(state) = WELL512MACRO_z1(state) ^ WELL512MACRO_z2(state), \
	newV0_(state) = MAT0NEG(-2,WELL512MACRO_z0(state)) ^ MAT0NEG(-18,WELL512MACRO_z1(state)) ^ MAT3NEG(-28,WELL512MACRO_z2(state)) ^ MAT4NEG(-5,0xda442d24U,newV1_(state)), \
	state.i = (state.i + 15) & 0x0000000fU, \
	state.s[state.i] \
)

#define V0            state->s[state->i                   ]
#define VM1           state->s[(state->i+M1) & 0x0000000fU]
#define VM2           state->s[(state->i+M2) & 0x0000000fU]
#define VM3           state->s[(state->i+M3) & 0x0000000fU]
#define VRm1          state->s[(state->i+15) & 0x0000000fU]
#define VRm2          state->s[(state->i+14) & 0x0000000fU]
#define newV0         state->s[(state->i+15) & 0x0000000fU]
#define newV1         state->s[state->i                   ]
#define newVRm1       state->s[(state->i+14) & 0x0000000fU]

/**
Generates a random 32-bit unsigned integer using WELL RNG.

@param state State of the RNG to use.
*/
#define well512_uint(state) _well512_uint(&state)
uint _well512_uint(well512_state* state){
	unsigned int z0, z1, z2;
	z0    = VRm1;
	z1    = MAT0NEG (-16,V0)    ^ MAT0NEG (-15, VM1);
	z2    = MAT0POS (11, VM2)  ;
	newV1 = z1                  ^ z2; 
	newV0 = MAT0NEG (-2,z0)     ^ MAT0NEG(-18,z1)    ^ MAT3NEG(-28,z2) ^ MAT4NEG(-5,0xda442d24U,newV1) ;
	state->i = (state->i + 15) & 0x0000000fU;
	return state->s[state->i];
}

/**
Seeds WELL RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void well512_seed(well512_state* state, unsigned long j){
    state->i = 0;
    for (int i = 0; i < R; i+=2){
		j=6906969069UL * j + 1234567UL; //LCG
		state->s[i    ] = j;
		state->s[i + 1] = j>>32;
	}
}

// Kernel function
// Seed RNG by single ulong
class well512_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<well512_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		well512_seed_by_value_kernel(ulong val,
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
            well512_state state;
            well512_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class well512_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<well512_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		well512_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            well512_state state;
            well512_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class well512_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<well512_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		well512_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            well512_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= well512_uint(state);
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
void WELL512_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 well512_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void WELL512_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 well512_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void WELL512_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 well512_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __WELL512_RNG__
