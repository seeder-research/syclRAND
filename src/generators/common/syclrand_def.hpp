#include "CL/sycl.hpp"
#include <vector>

#ifndef __SYCLRAND_CLASS
#define __SYCLRAND_CLASS

class _SyCLRAND {
	public:
		// Constructor
		_SyCLRAND() :
		    seedArr(NULL) {}
		// For updating the protected seedArr variable
		void setSeed(std::vector<ulong> seed) { this->seedArr = seed };
		// For seeding the PRNG by a single ulong
		virtual void seed_by_value(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
		// For seeding the PRNG by an array of ulong
		virtual void seed_by_array(cl::sycl::queue funcQueue, size_t gsize, size_t lsize);
		// Use the PRNG to generate <count> number of random numbers
		// and fill into the buffer <dst>
		virtual void generate_uint(cl::sycl::queue funcQueue, int count, cl::sycl::buffer &dst, size_t gsize, size_t lsize);
	protected:
		using seed_accessor = 
		      sycl::accessor<ulong, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    std::vector<ulong>    seedArr;
		seed_accessor         seedBuf;
};

#endif