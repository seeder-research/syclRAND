#include "CL/sycl.hpp"
#include <vector>

#ifndef __SYCLRAND_CLASS
#define __SYCLRAND_CLASS

class _SyCLRAND {
	public:
		SyCLRAND() :
		    seedArr(NULL) {}
		void setSeed(std::vector<ulong> seed) { this->seedArr = seed };
		virtual void seed_by_value();
		virtual void seed_by_array();
		virtual void generate_uint(int count, cl::sycl::buffer &dst);
	private:
	    std::vector<ulong>    seedArr;
};

#endif