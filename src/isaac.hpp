#ifndef __ISAAC_RNG__
#define __ISAAC_RNG__

#ifndef __SYCLRAND_BASE_CLASS
#include "syclrand_def.hpp"
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

typedef sycl::accessor<isaac_state, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer> state_accessor;
typedef sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer> input_accessor;
typedef sycl::accessor<uint, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer> output_accessor;

/* Isaac class */
class ISAAC_PRNG : _SyCLRAND {
    public:
        void seed_by_value(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
        void seed_by_array(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
        void generate_uint(cl::sycl::queue funcQueue, int count, cl::sycl::buffer<uint, 1> &dst, size_t gsize, size_t lsize);
    private:
        sycl::buffer<isaac_state, 1>    stateBuf;
        isaac_state                     *stateArr;
};

// Functions for ISAAC RNG
void isaac_advance(isaac_state* state);

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
inline void isaac_seed(isaac_state* state, ulong j);

// Kernel function
// Seed RNG by single ulong
class isaac_seed_by_value_kernel {
    public:
        isaac_seed_by_value_kernel(ulong val,
                                   state_accessor statePtr)
            : seedVal(val),
              stateBuf(statePtr) {}
        void operator()(sycl::nd_item<1> item);

    private:
        ulong           seedVal;
        state_accessor  stateBuf;
};

// Kernel function
// Seed RNG by array of ulong
class isaac_seed_by_array_kernel {
    public:
        isaac_seed_by_array_kernel(input_accessor seedArr,
                                   state_accessor statePtr)
        : seedArr(seedArr),
	  stateBuf(statePtr) {}

        void operator()(sycl::nd_item<1> item);

    private:
        state_accessor  stateBuf;
        input_accessor  seedArr;
};

// Kernel function
// Generate random uint
class isaac_rng_kernel{
    public:
        isaac_rng_kernel(int count,
                         state_accessor statePtr,
                         output_accessor dstPtr)
            : num(count),
              stateBuf(statePtr),
              res(dstPtr) {}
        void operator()(sycl::nd_item<1> item);

    private:
        int             num;
        state_accessor  stateBuf;
        output_accessor res;
};

#endif // __ISAAC_RNG__
