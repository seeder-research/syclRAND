const char * dbl_kernel_util_defs = R"EOK(
////////////////////////////////////////////////////////////////////////////////////
#define double_inv_max_uint 0x3df0000000100000
#define double_inv_nmax_uint 0x3df0000000000000

#define double_inv_max_ulong 0x3bf0000000000000
#define double_inv_nmax_ulong 0x3bf0000000000000
                              
#define double_inv_nmax_uint_adj 0x3de0000000000000
#define double_inv_nmax_ulong_adj 0x3be0000000000000

#define CLRAND_SQRT2 1.4142135623730951

inline double dsimple_cc_01_uint(uint x) {
    double tmp = (double)(x);
	return tmp*as_double(double_inv_max_uint);
} // dsimple_cc_01_uint

inline double dsimple_cc_01_ulong(ulong x) {
    double tmp = (double)(x);
	return tmp*as_double(double_inv_max_ulong);
} // dsimple_cc_01_ulong

inline double dsimple_co_01_uint(uint x) {
    double tmp = (double)(x);
	return tmp*as_double(double_inv_nmax_uint);
} // 

inline double dsimple_co_01_ulong(ulong x) {
    double tmp = (double)(x);
	return tmp*as_double(double_inv_nmax_ulong);
} // dsimple_co_01_ulong

inline double dsimple_oc_01_uint(uint x) {
    double tmp = (double)(x);
	return (tmp+1.0)*as_double(double_inv_nmax_uint);
} // dsimple_oc_01_uint

inline double dsimple_oc_01_ulong(ulong x) {
    double tmp = (double)(x);
	return (tmp+1.0)*as_double(double_inv_nmax_ulong);
} // dsimple_oc_01_ulong

inline double dsimple_oo_01_uint(uint x) {
    double tmp = (double)(x);
	return tmp*as_double(double_inv_nmax_uint) + as_double(double_inv_nmax_uint_adj);
} // dsimple_oo_01_uint

inline double dsimple_oo_01_ulong(ulong x) {
    double tmp = (double)(x);
	return tmp*as_double(double_inv_nmax_ulong) + as_double(double_inv_nmax_ulong_adj);
} // dsimple_oo_01_ulong
////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////
/*
Define function to calculate inverse CDF of standard Gaussian distribution.
This function was taken from clProbDist from Univ. Montreal
*/
////////////////////////////////////////////////////////////////////////////////////
/*
#define DBL_NORMCDF_INV_P1_0 0.160304955844066229311E2
#define DBL_NORMCDF_INV_P1_1 -0.90784959262960326650E2
#define DBL_NORMCDF_INV_P1_2 0.18644914861620987391E3
#define DBL_NORMCDF_INV_P1_3 -0.16900142734642382420E3
#define DBL_NORMCDF_INV_P1_4 0.6545466284794487048E2
#define DBL_NORMCDF_INV_P1_5 -0.864213011587247794E1
#define DBL_NORMCDF_INV_P1_6 0.1760587821390590

#define DBL_NORMCDF_INV_Q1_0 0.147806470715138316110E2
#define DBL_NORMCDF_INV_Q1_1 -0.91374167024260313396E2
#define DBL_NORMCDF_INV_Q1_2 0.21015790486205317714E3
#define DBL_NORMCDF_INV_Q1_3 -0.22210254121855132366E3
#define DBL_NORMCDF_INV_Q1_4 0.10760453916055123830E3
#define DBL_NORMCDF_INV_Q1_5 -0.206010730328265443E2
#define DBL_NORMCDF_INV_Q1_6 0.1E1

#define DBL_NORMCDF_INV_P2_0 -0.152389263440726128E-1
#define DBL_NORMCDF_INV_P2_1 0.3444556924136125216
#define DBL_NORMCDF_INV_P2_2 -0.29344398672542478687E1
#define DBL_NORMCDF_INV_P2_3 0.11763505705217827302E2
#define DBL_NORMCDF_INV_P2_4 -0.22655292823101104193E2
#define DBL_NORMCDF_INV_P2_5 0.19121334396580330163E2
#define DBL_NORMCDF_INV_P2_6 -0.5478927619598318769E1
#define DBL_NORMCDF_INV_P2_7 0.237516689024448000

#define DBL_NORMCDF_INV_Q2_0 -0.108465169602059954E-1
#define DBL_NORMCDF_INV_Q2_1 0.2610628885843078511
#define DBL_NORMCDF_INV_Q2_2 -0.24068318104393757995E1
#define DBL_NORMCDF_INV_Q2_3 0.10695129973387014469E2
#define DBL_NORMCDF_INV_Q2_4 -0.23716715521596581025E2
#define DBL_NORMCDF_INV_Q2_5 0.24640158943917284883E2
#define DBL_NORMCDF_INV_Q2_6 -0.10014376349783070835E2
#define DBL_NORMCDF_INV_Q2_7 0.1E1

#define DBL_NORMCDF_INV_P3_0 0.56451977709864482298E-4
#define DBL_NORMCDF_INV_P3_1 0.53504147487893013765E-2
#define DBL_NORMCDF_INV_P3_2 0.12969550099727352403
#define DBL_NORMCDF_INV_P3_3 0.10426158549298266122E1
#define DBL_NORMCDF_INV_P3_4 0.28302677901754489974E1
#define DBL_NORMCDF_INV_P3_5 0.26255672879448072726E1
#define DBL_NORMCDF_INV_P3_6 0.20789742630174917228E1
#define DBL_NORMCDF_INV_P3_7 0.72718806231556811306
#define DBL_NORMCDF_INV_P3_8 0.66816807711804989575E-1
#define DBL_NORMCDF_INV_P3_9 -0.17791004575111759979E-1
#define DBL_NORMCDF_INV_P3_10 0.22419563223346345828E-2

#define DBL_NORMCDF_INV_Q3_0 0.56451699862760651514E-4
#define DBL_NORMCDF_INV_Q3_1 0.53505587067930653953E-2
#define DBL_NORMCDF_INV_Q3_2 0.12986615416911646934
#define DBL_NORMCDF_INV_Q3_3 0.10542932232626491195E1
#define DBL_NORMCDF_INV_Q3_4 0.30379331173522206237E1
#define DBL_NORMCDF_INV_Q3_5 0.37631168536405028901E1
#define DBL_NORMCDF_INV_Q3_6 0.38782858277042011263E1
#define DBL_NORMCDF_INV_Q3_7 0.20372431817412177929E1
#define DBL_NORMCDF_INV_Q3_8 0.1E1
*/
//inline double normcdfinv_double(double u) {
	/*
	* Returns the inverse of the cdf of the normal distribution.
	* Rational approximations giving 16 decimals of precision.
	* J.M. Blair, C.A. Edwards, J.H. Johnson, "Rational Chebyshev
	* approximations for the Inverse of the Error Function", in
	* Mathematics of Computation, Vol. 30, 136, pp 827, (1976)
	*/
/*
	cl_bool negatif;
	double y, z, v, w;
	double x = u;

	if (u < 0.0 || u > 1.0) {
		return NAN;
	}
	if (u <= 0.0) {
		return DBL_MIN; // Double.NEGATIVE_INFINITY;
	}
	if (u >= 1.0) {
		return DBL_MAX; // Double.POSITIVE_INFINITY;
	}

	// Transform x as argument of InvErf
	x = 2.0 * x - 1.0;
	if (x < 0.0) {
		x = -x;
		negatif = CL_TRUE;
	}
	else {
		negatif = CL_FALSE;
	}

	if (x <= 0.75) {
		y = x * x - 0.5625;
		v = w = 0.0;
		v = v * y + DBL_NORMCDF_INV_P1_6;
		v = v * y + DBL_NORMCDF_INV_P1_5;
		v = v * y + DBL_NORMCDF_INV_P1_4;
		v = v * y + DBL_NORMCDF_INV_P1_3;
		v = v * y + DBL_NORMCDF_INV_P1_2;
		v = v * y + DBL_NORMCDF_INV_P1_1;
		v = v * y + DBL_NORMCDF_INV_P1_0;

		w = w * y + DBL_NORMCDF_INV_Q1_6;
		w = w * y + DBL_NORMCDF_INV_Q1_5;
		w = w * y + DBL_NORMCDF_INV_Q1_4;
		w = w * y + DBL_NORMCDF_INV_Q1_3;
		w = w * y + DBL_NORMCDF_INV_Q1_2;
		w = w * y + DBL_NORMCDF_INV_Q1_1;
		w = w * y + DBL_NORMCDF_INV_Q1_0;

		z = (v / w) * x;
	}
	else if (x <= 0.9375) {
		y = x * x - 0.87890625;

		v = w = 0.0;
		v = v * y + DBL_NORMCDF_INV_P2_7;
		v = v * y + DBL_NORMCDF_INV_P2_6;
		v = v * y + DBL_NORMCDF_INV_P2_5;
		v = v * y + DBL_NORMCDF_INV_P2_4;
		v = v * y + DBL_NORMCDF_INV_P2_3;
		v = v * y + DBL_NORMCDF_INV_P2_2;
		v = v * y + DBL_NORMCDF_INV_P2_1;
		v = v * y + DBL_NORMCDF_INV_P2_0;

		w = w * y + DBL_NORMCDF_INV_Q2_7;
		w = w * y + DBL_NORMCDF_INV_Q2_6;
		w = w * y + DBL_NORMCDF_INV_Q2_5;
		w = w * y + DBL_NORMCDF_INV_Q2_4;
		w = w * y + DBL_NORMCDF_INV_Q2_3;
		w = w * y + DBL_NORMCDF_INV_Q2_2;
		w = w * y + DBL_NORMCDF_INV_Q2_1;
		w = w * y + DBL_NORMCDF_INV_Q2_0;

		z = (v / w) * x;
	}
	else {
		if (u > 0.5) {
			y = 1.0 / sqrt(-log(1.0 - x));
		} else {
			y = 1.0 / sqrt(-log(2.0 * u));
		}

		v = 0.0;
		v = v * y + DBL_NORMCDF_INV_P3_10;
		v = v * y + DBL_NORMCDF_INV_P3_9;
		v = v * y + DBL_NORMCDF_INV_P3_8;
		v = v * y + DBL_NORMCDF_INV_P3_7;
		v = v * y + DBL_NORMCDF_INV_P3_6;
		v = v * y + DBL_NORMCDF_INV_P3_5;
		v = v * y + DBL_NORMCDF_INV_P3_4;
		v = v * y + DBL_NORMCDF_INV_P3_3;
		v = v * y + DBL_NORMCDF_INV_P3_2;
		v = v * y + DBL_NORMCDF_INV_P3_1;
		v = v * y + DBL_NORMCDF_INV_P3_0;

		w = 0.0;
		w = w * y + DBL_NORMCDF_INV_Q3_8;
		w = w * y + DBL_NORMCDF_INV_Q3_8;
		w = w * y + DBL_NORMCDF_INV_Q3_8;
		w = w * y + DBL_NORMCDF_INV_Q3_8;
		w = w * y + DBL_NORMCDF_INV_Q3_8;
		w = w * y + DBL_NORMCDF_INV_Q3_8;
		w = w * y + DBL_NORMCDF_INV_Q3_8;
		w = w * y + DBL_NORMCDF_INV_Q3_8;
		w = w * y + DBL_NORMCDF_INV_Q3_8;

		z = (v / w) / y;
	}

	if (negatif) {
		if (u < 1.0e-105) {
			double RACPI = 1.77245385090551602729;
			w = exp(-z * z) / RACPI;  // pdf
			y = 2.0 * z * z;
			v = 1.0;
			double term = 1.0;

			// Asymptotic series for erfc(z) (apart from exp factor)
			term *= -1.0 / y;
			v += term;
			term *= -3.0 / y;
			v += term;
			term *= -5.0 / y;
			v += term;
			term *= -7.0 / y;
			v += term;
			term *= -9.0 / y;
			v += term;
			term *= -11.0 / y;
			v += term;

			// Apply 1 iteration of Newton solver to get last few decimals
			z -= u / w - 0.5 * v / z;
		}
		return -(z * CLRAND_SQRT2);
	}
	else {
		return z * CLRAND_SQRT2;
	}

} // normcdfinv_double
*/
////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////
/*
Kernel functions for fast conversion of uint to double while copying between buffers
*/
////////////////////////////////////////////////////////////////////////////////////
kernel void CopyUintAsDbl01CC(global double* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = dsimple_cc_01_uint(src[ii]);
	}
}

kernel void CopyUintAsDbl01CO(global double* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = dsimple_co_01_uint(src[ii]);
	}
}

kernel void CopyUintAsDbl01OC(global double* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = dsimple_oc_01_uint(src[ii]);
	}
}

kernel void CopyUintAsDbl01OO(global double* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = dsimple_oo_01_uint(src[ii]);
	}
}

// Kernel functions for fast conversion of ulong to double while copying between buffers
kernel void CopyUlongAsDbl01CC(global double* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = dsimple_cc_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsDbl01CO(global double* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = dsimple_co_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsDbl01OC(global double* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = dsimple_oc_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsDbl01OO(global double* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = dsimple_oo_01_ulong(src[ii]);
	}
}
/*
// Kernel functions for fast copy of Gaussian double while copying between buffers
kernel void CopyUintAsNormDbl01OO(global double* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	double localVal;
	
	for (uint ii = 0; ii < count; ii += gsize) {
		localVal = dsimple_oo_01_uint(src[ii]);
		dst[ii] = normcdfinv_double(localVal);
	}
}

kernel void CopyUlongAsNormDbl01OO(global double* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	double localVal;
	
	for (uint ii = 0; ii < count; ii += gsize) {
		localVal = dsimple_oo_01_ulong(src[ii]);
		dst[ii] = normcdfinv_double(localVal);
	}
}
*/
////////////////////////////////////////////////////////////////////////////////////

)EOK";
