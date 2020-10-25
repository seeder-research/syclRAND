#include "isaac.hpp"

/**
@file

Implements ISAAC (Indirection, Shift, Accumulate, Add, and Count). Does not work on graphics cards, as it requires unaligned accesses to memory.

R. J. Jenkins, Isaac, in: International Workshop on Fast Software Encryption, Springer, 1996, pp. 41–49.
*/

// Functions for ISAAC RNG
void isaac_advance(isaac_state* state) {
    uint a, b, x, y, *m, *m2, *r, *mend;
    m = state->mm;
    r = state->rr;
    a = state->aa;
    b = state->bb + (++state->cc);
    for (m = state->mm, mend = m2 = m+(ISAAC_RANDSIZ/2); m < mend; ) {
        rngstep(a << 13, a, b, state->mm, m, m2, r, x);
        rngstep(a >> 6 , a, b, state->mm, m, m2, r, x);
        rngstep(a << 2 , a, b, state->mm, m, m2, r, x);
        rngstep(a >> 16, a, b, state->mm, m, m2, r, x);
    }
    for (m2 = state->mm; m2 < mend; ) {
        rngstep(a << 13, a, b, state->mm, m, m2, r, x);
        rngstep(a >> 6 , a, b, state->mm, m, m2, r, x);
        rngstep(a << 2 , a, b, state->mm, m, m2, r, x);
        rngstep(a >> 16, a, b, state->mm, m, m2, r, x);
    }
    state->bb = b;
    state->aa = a;
}

/**
Seeds ISAAC RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
inline void isaac_seed(isaac_state* state, ulong j) {
    state->aa = j;
    state->bb = j ^ 123456789;
    state->cc = j + 123456789;
    state->idx = ISAAC_RANDSIZ;
    for (int i=0;i<ISAAC_RANDSIZ;i++) {
        j = 6906969069UL * j + 1234567UL; //LCG
        state->mm[i] = j;
        //isaac_advance(state);
    }
}

void isaac_seed_by_value_kernel::operator()(sycl::nd_item<1> item) {
    uint gid = item.get_global_id(0);
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


// Kernel function
// Seed RNG by array of ulong
void isaac_seed_by_array_kernel::operator()(sycl::nd_item<1> item) {
    uint gid = item.get_global_id(0);
    ulong seed = seedArr[gid];
    isaac_state state;
    isaac_seed(&state,seed);
    stateBuf[gid] = state;
}

// Kernel function
// Generate random uint
void isaac_rng_kernel::operator()(sycl::nd_item<1> item) {
    uint gid = item.get_global_linear_id();
    uint gsize = item.get_local_range(0) * item.get_group_range(0);
    isaac_state state;
    state = stateBuf[gid];
    for (uint i=gid; i<num; i+=gsize) {
        res[i]= isaac_uint(state);
    }
    stateBuf[gid] = state;
}

// Class function
// Launch kernel to seed RNG by single ulong
void ISAAC_PRNG::seed_by_value(sycl::queue funcQueue,
                               size_t gsize,
                               size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto state_acc = (&this->stateBuf)->template get_access<sycl::access::mode::read_write>(cgh);

        // If seed by value, we will use the first element in seedArr as the value
        auto seedVal = this->seedArr.data()[0];

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
        auto state_acc = (&this->stateBuf)->template get_access<sycl::access::mode::read_write>(cgh);
        auto seedBuf = sycl::buffer<std::vector<ulong>, 1>(&this->seedArr, cl::sycl::range<1>(this->seedArr.size()));
        auto seed_acc = (&this->seedBuf)->template get_access<sycl::access::mode::read>(cgh);

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
        auto state_acc = (&this->stateBuf)->template get_access<sycl::access::mode::read_write>(cgh);
        auto dst_acc = dst.template get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
                                           sycl::range<1>(lsize)),
                                           isaac_rng_kernel(count, state_acc, dst_acc));
    });
}
