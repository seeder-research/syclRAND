/*
This file defines the SyCLRAND container class.
The underlying PRNG object is defined in the constructor
*/

#ifndef __SYCLRAND_BASE_CLASS
#define __SYCLRAND_BASE_CLASS
#ifndef __SYCLRAND_CLASS
#include "CL/sycl.hpp"
#include <vector>

typedef unsigned char uchar;
typedef sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer> seed_accessor;

#endif // __SYCLRAND_CLASS

class _SyCLRAND {
    public:

        // Constructor
        _SyCLRAND(sycl::buffer<ulong, 1> inBuf) :
            seedArr(NULL),
            seedBuf(inBuf) {}

        // For updating the private seedArr variable
        void setSeed(std::vector<ulong> seed) { this->seedArr = seed; }

        // For seeding the PRNG by a single ulong
        virtual void seed_by_value(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);

        // For seeding the PRNG by an array of ulong
        virtual void seed_by_array(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
        // Use the PRNG to generate <count> number of random numbers
        // and fill into the buffer <dst>
        virtual void generate_uint(cl::sycl::queue funcQueue, int count, cl::sycl::buffer<uint, 1> &dst, size_t gsize, size_t lsize);

        std::vector<ulong>        seedArr;
        sycl::buffer<ulong, 1>    seedBuf;
};

#endif // __SYCLRAND_BASE_CLASS
