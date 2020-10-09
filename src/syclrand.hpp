#ifndef __SYCLRAND_CLASS
#include "CL/sycl.hpp"
#include <vector>
#define __SYCLRAND_CLASS

template<class T>
class SyCLRAND {
	private:
	    T PRNGObj;
	public:
		// Constructor
		SyCLRAND();
		// For updating the protected seedArr variable
		void setSeed(std::vector<ulong> seed) { this->PRNGObj.setSeed(seed); }
		// For seeding the PRNG by a single ulong
		void seed_by_value(cl::sycl::queue funcQueue, size_t gsize, size_t lsize) { this->PRNGObj.seed_by_value(funcQueue, gsize, lsize); }
		// For seeding the PRNG by an array of ulong
		void seed_by_array(cl::sycl::queue funcQueue, size_t gsize, size_t lsize) { this->PRNGObj.seed_by_array(funcQueue, gsize, lsize); }
		// Use the PRNG to generate <count> number of random numbers
		// and fill into the buffer <dst>
		void generate_uint(cl::sycl::queue funcQueue, int count, cl::sycl::buffer &dst, size_t gsize, size_t lsize) { this->PRNGObj.generate_uint(funcQueue, count, dst, gsize, lsize); }
};

#endif // __SYCLRAND_CLASS
