#ifndef __ISAAC_RNG__
#define __ISAAC_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "common/syclrand_def.hpp"
#endif // __SYCLRAND_BASE_CLASS

/**
@file

Implements ISAAC (Indirection, Shift, Accumulate, Add, and Count). Does not work on graphics cards, as it requires unaligned accesses to memory.

R. J. Jenkins, Isaac, in: International Workshop on Fast Software Encryption, Springer, 1996, pp. 41–49.
*/

/* ind(mm,x) is bits 2..9 of x, or (floor(x/4) mod 256)*4 */

#define ind(mm,x) (*(uint *)((uchar *)(mm) + ((x) & (255 << 2))))
//#define ind(mm,x) (*(uint *)((uint *)(mm) + (((x) >> 2) & 255)))
#define rngstep(mix,a,b,mm,m,m2,r,x) \
{\
	x = *m; \
	a = (a ^ (mix)) + *(m2++); \
	*(m++) = y = ind(mm, x) + a + b; \
	*(r++) = b = ind(mm, y >> 8) + x; \
}

#define ISAAC_RANDSIZL   (8)
#define ISAAC_RANDSIZ    (1<<ISAAC_RANDSIZL)

/**
State of ISAAC RNG.
*/
typedef struct{
  uint rr[ISAAC_RANDSIZ];
  uint mm[ISAAC_RANDSIZ];
  uint aa;
  uint bb;
  uint cc;
  uint idx;
} isaac_state;

/* Isaac class */
class ISAAC_PRNG : _SyCLRAND {
	public:
	    using state_accessor = 
		      sycl::accessor<isaac_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using output_accessor = 
			  sycl::accessor<uint, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using seed_accessor = 
		      sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		void seed_by_value(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
		void seed_by_array(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
		void generate_uint(cl::sycl::queue funcQueue, int count, cl::sycl::buffer &dst, size_t gsize, size_t lsize);
	private:
	    state_accessor      stateBuf;
		isaac_state         *stateArr;
};

// Functions for ISAAC RNG
void isaac_advance(isaac_state* state){
	uint a, b, x, y, *m, *m2, *r, *mend;
	m = state->mm;
	r = state->rr;
	a = state->aa;
	b = state->bb + (++state->cc);
	for (m = state->mm, mend = m2 = m+(ISAAC_RANDSIZ/2); m < mend; ){
		rngstep(a << 13, a, b, state->mm, m, m2, r, x);
		rngstep(a >> 6 , a, b, state->mm, m, m2, r, x);
		rngstep(a << 2 , a, b, state->mm, m, m2, r, x);
		rngstep(a >> 16, a, b, state->mm, m, m2, r, x);
	}
	for (m2 = state->mm; m2 < mend; ){
		rngstep(a << 13, a, b, state->mm, m, m2, r, x);
		rngstep(a >> 6 , a, b, state->mm, m, m2, r, x);
		rngstep(a << 2 , a, b, state->mm, m, m2, r, x);
		rngstep(a >> 16, a, b, state->mm, m, m2, r, x);
	}
	state->bb = b;
	state->aa = a;
}

/**
Generates a random 32-bit unsigned integer using ISAAC RNG.

@param state State of the RNG to use.
*/
#define isaac_uint(state) _isaac_uint(&state)
uint _isaac_uint(isaac_state* state){
	//printf("%d\n", get_global_id(0));
	if(state->idx == ISAAC_RANDSIZ){
		isaac_advance(state);
		state->idx=0;
	}
	return state->rr[state->idx++];
}

/**
Seeds ISAAC RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
inline void isaac_seed(isaac_state* state, ulong j){
	state->aa = j;
	state->bb = j ^ 123456789;
	state->cc = j + 123456789;
	state->idx = ISAAC_RANDSIZ;
	for(int i=0;i<ISAAC_RANDSIZ;i++){
		j=6906969069UL * j + 1234567UL; //LCG
		state->mm[i]=j;
		//isaac_advance(state);
	}
}

// Kernel function
// Seed RNG by single ulong
class isaac_seed_by_value_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<isaac_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		isaac_seed_by_value_kernel(ulong val,
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
            isaac_state state;
            isaac_seed(&state, seed);
            stateBuf[gid] = state;
		}

	private:
		ulong           seedVal;
		state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class isaac_seed_by_array_kernel {
	public:
	    using state_accessor =
		    sycl::accessor<isaac_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using input_accessor =
		    sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		isaac_seed_by_array_kernel(input_accessor seedArr,
			state_accessor statePtr)
		: seedArr(seedArr),
		  stateBuf(statePtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_id(0);
            ulong seed = seedArr[gid];
            isaac_state state;
            isaac_seed(&state,seed);
            stateBuf[gid] = state;
		}

	private:
		state_accessor  stateBuf;
		input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class isaac_rng_kernel{
	public:
	    using state_accessor =
		    sycl::accessor<isaac_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using output_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		isaac_rng_kernel(int count,
			state_accessor statePtr,
			output_accessor dstPtr)
		: num(count),
		  stateBuf(statePtr),
		  res(dstPtr) {}
		void operator()(sycl::nd_item<1> item) {
            uint gid=get_global_linear_id();
            uint gsize=get_num_range(0);
            isaac_state state;
            state = stateBuf[gid];
            for(uint i=gid;i<num;i+=gsize) {
                res[i]= isaac_uint(state);
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
void ISAAC_PRNG::seed_by_value(sycl::queue funcQueue,
				               size_t gsize,
				               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		// If seed by value, we will use the first element in seedArr as the value
		seedVal = this->seedArr.data()[0];

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 isaac_seed_by_value_kernel(seedVal, state_acc));
    });
}

// Class function
// Launch kernel to seed RNG by an array of ulong
void ISAAC_PRNG::seed_by_array(sycl::queue funcQueue,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = this->stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
		this->seedBuf = cl::sycl::buffer<ulong, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = this->seedBuf->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 isaac_seed_by_array_kernel(seed_acc, state_acc));
    });
}

void ISAAC_PRNG::generate_uint(sycl::queue funcQueue,
				int count,
                sycl::buffer<uint, 1> &dst,
				size_t gsize,
				size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = stateBuf->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 isaac_rng_kernel(count, state_acc, dst_acc));
    });
}

#endif // __ISAAC_RNG__
