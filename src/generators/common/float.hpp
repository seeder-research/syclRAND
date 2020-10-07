const char * flt_kernel_util_defs = R"EOK(
////////////////////////////////////////////////////////////////////////////////////
#define float_inv_max_uint  2.328306437080797e-10f
#define float_inv_nmax_uint 0x2f800000

#define float_inv_max_ulong 5.4210108624275221700372640e-20f
#define float_inv_nmax_ulong 0x1f800000

#define float_inv_nmax_uint_adj 0x2f000000
#define float_inv_nmax_ulong_adj 0x1f000000

inline float simple_cc_01_uint(uint x) {
	float tmp = (float)(x);
	return  tmp*float_inv_max_uint;
} // simple_cc_01_uint

inline float simple_cc_01_ulong(ulong x) {
    float tmp = (float)(x);
	return tmp*float_inv_max_ulong;
} // simple_cc_01_ulong

inline float simple_co_01_uint(uint x) {
    float tmp = (float)(x);
	return tmp*as_float(float_inv_nmax_uint);
} // simple_co_01_uint

inline float simple_co_01_ulong(ulong x) {
    float tmp = (float)(x);
	return tmp*as_float(float_inv_nmax_ulong);
} // simple_co_01_ulong

inline float simple_oc_01_uint(uint x) {
    float tmp = (float)(x);
	return (tmp+1.0f)*as_float(float_inv_nmax_uint);
} // simple_oc_01_uint

inline float simple_oc_01_ulong(ulong x) {
    float tmp = (float)(x);
	return (tmp+1.0f)*as_float(float_inv_nmax_ulong);
} // simple_oc_01_ulong

inline float simple_oo_01_uint(uint x) {
    float tmp = (float)(x);
	return tmp*as_float(float_inv_nmax_uint) + as_float(float_inv_nmax_uint_adj);
} // simple_oo_01_uint

inline float simple_oo_01_ulong(ulong x) {
    float tmp = (float)(x);
	return tmp*as_float(float_inv_nmax_ulong) + as_float(float_inv_nmax_ulong_adj);
} // simple_oo_01_ulong
////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////
/*
The auxiliary function to calculate the exponents for single- and double-precision
random numbers in the interval [0, 1) are defined as exponent_adj(). This auxiliary
function takes as input a uint bit pattern and a pre-computed uint exponent. The
pre-computed exponent is then adjusted based where the right most '0' occurs in the
bit pattern.

Random numbers in the interval (0, 1] can be obtained from random numbers generated
in the interval [0, 1) by converting any 0.0 to 1.0

Random numbers in the interval (0, 1) can be obtained from random numbers generated
in the interval [0, 1) by discarding all 0.0

Random numbers in the interval [0, 1] can be obtained from random numbers generated
in the interval [0, 1) by randomly converting 0.0 to 1.0. At least two bits in the
random bit streams given to the functions flt_exponent_co_01() and
dbl_exponent_co_01() remain unused. One of them can be used to decide whether the
0.0 gets flipped to 1.0 or remains as 0.0

Mantissa of random doubles can be generated from one ulong or a pair of uint
From ulong, just and the ulong with 0x000fffffffffffff to extract the lower 52-bits
A pair of uint can be thought of as a ulong. To extract the lower 52-bits, we keep
the uint representing the lower 32-bits, and extract the lower 20-bits from the
other uint. So we need, in total, 2 masks. One to extract lower 23-bits for
generating floats, and another to extract lower 20-bits for generating doubles
*/
////////////////////////////////////////////////////////////////////////////////////

// Kernel should iterate through uint that are equal to 0xffffffff to adjust its
// copy of the exponent. Once the kernel finds an uint that is not 0xffffffff, it
// then calls this function to determine the correct exponent, and bit shift
// accordingly
inline uint exponent_adj(uint inBits, uint inexp) {
	uint outexp=inexp; // Start exponent at the input and count down
	uint idx, tmpbits;

    if (outexp > 0) {
		for (idx = 32; idx > 0; idx--) {
			if ((outexp == 0) || ((tmpbits & 0x00000001) == 0)) {
				break;
			}
			outexp--;
			tmpbits >>= 1;
		}
	}

    return outexp;
} // exponent_adj

////////////////////////////////////////////////////////////////////////////////////
/*
Kernel functions for fast conversion of uint to float while copying between buffers
*/
////////////////////////////////////////////////////////////////////////////////////
kernel void CopyUintAsFlt01CC(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_cc_01_uint(src[ii]);
	}
}

kernel void CopyUintAsFlt01CO(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_co_01_uint(src[ii]);
	}
}

kernel void CopyUintAsFlt01OC(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_oc_01_uint(src[ii]);
	}
}

kernel void CopyUintAsFlt01OO(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_oo_01_uint(src[ii]);
	}
}

// Kernel functions for fast conversion of ulong to float while copying between buffers
kernel void CopyUlongAsFlt01CC(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_cc_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsFlt01CO(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_co_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsFlt01OC(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_oc_01_ulong(src[ii]);
	}
}

kernel void CopyUlongAsFlt01OO(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	
	for (uint ii = 0; ii < count; ii += gsize) {
		dst[ii] = simple_oo_01_ulong(src[ii]);
	}
}

// Kernel functions for generating packed float from uint in the buffer of the PRNG
kernel void CopySlowUint01COAsFlt(global uint* dst, global uint* src, uint count, uint offset, uint buf_size) {
	// dst: destination buffer which we will store the bit pattern of the float
	// src: source buffer containing the uint we want to convert
	// count: how many floats we need to generate
	// offset: offset to get the first valid uint in source buffer
	// buf_size: total number of uint the source buffer can hold
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	uint end_idx = buf_size - 5;

	for (uint idx = gid; idx < count; idx+=gsize) {
		// Each float needs 128+32 = 160 bits to generate (5 bytes)
		// Each work-item will read 5 consecutive uint from the buffer
		uint lidx = idx * 5 + offset;
		if (lidx < end_idx) {
			uint outExp = 126;
			uint tmpUintBuf;
			bool flag = true;

			// Get first uint to generate mantissa
			uint outNum = src[lidx];
			outNum &= 0x007fffff;

			// Compute the exponent...
			// First, check for the uint that will be used to compute the actual exponent
			tmpUintBuf = src[lidx+1];

			// Check for the first 32 bits
			if (tmpUintBuf != as_uint(0xffffffff)) {
				flag = false;
			} else {
				tmpUintBuf = src[lidx+2];
				outExp = 94;
			}
			// Check for the second 32 bits if first 32-bits are all ones
			if (flag) {
				if (tmpUintBuf != as_uint(0xffffffff)) {
					flag = false;
				} else {
					tmpUintBuf = src[lidx+3];
					outExp = 62;
				}
			}
			// Check for the third 32 bits if first 32-bits are all ones
			if (flag) {
				if (tmpUintBuf != as_uint(0xffffffff)) {
					flag = false;
				} else {
					tmpUintBuf = src[lidx+4];
					outExp = 30;
				}
			}
			// Check for the fourth 32 bits if first 30-bits are all ones
			if (flag) {
				if (tmpUintBuf >= as_uint(0x3fffffff)) {
					outExp = 0;
				} else {
					flag = false;
				}
			}
			if (!flag) {
				outExp = exponent_adj(tmpUintBuf, outExp);
			}
			outNum |= (outExp << 23); // Adjust the exponent bits to the correct position and logical OR with mantissa
			dst[gid] = outNum; // Store result to output buffer
		}
	}
}

// Kernel functions for generating packed float from uint in the buffer of the PRNG
kernel void CopySlowUint01OCAsFlt(global uint* dst, global uint* src, uint count, uint offset, uint buf_size) {
	// dst: destination buffer which we will store the bit pattern of the float
	// src: source buffer containing the uint we want to convert
	// count: how many floats we need to generate
	// offset: offset to get the first valid uint in source buffer
	// buf_size: total number of uint the source buffer can hold
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	uint end_idx = buf_size - 5;

	for (uint idx = gid; idx < count; idx+=gsize) {
		// Each float needs 128+32 = 160 bits to generate (5 bytes)
		// Each work-item will read 5 consecutive uint from the buffer
		uint lidx = idx * 5 + offset;
		if (lidx < end_idx) {
			uint outExp = 126;
			uint tmpUintBuf;
			bool flag = true;

			// Get first uint to generate mantissa
			uint outNum = src[lidx];
			outNum &= 0x007fffff;

			// Compute the exponent...
			// First, check for the uint that will be used to compute the actual exponent
			tmpUintBuf = src[lidx+1];

			// Check for the first 32 bits
			if (tmpUintBuf != as_uint(0xffffffff)) {
				flag = false;
			} else {
				tmpUintBuf = src[lidx+2];
				outExp = 94;
			}
			// Check for the second 32 bits if first 32-bits are all ones
			if (flag) {
				if (tmpUintBuf != as_uint(0xffffffff)) {
					flag = false;
				} else {
					tmpUintBuf = src[lidx+3];
					outExp = 62;
				}
			}
			// Check for the third 32 bits if first 32-bits are all ones
			if (flag) {
				if (tmpUintBuf != as_uint(0xffffffff)) {
					flag = false;
				} else {
					tmpUintBuf = src[lidx+4];
					outExp = 30;
				}
			}
			// Check for the fourth 32 bits if first 30-bits are all ones
			if (flag) {
				if (tmpUintBuf >= as_uint(0x3fffffff)) {
					outExp = 0;
				} else {
					flag = false;
				}
			}
			if (!flag) {
				outExp = exponent_adj(tmpUintBuf, outExp);
			}
			outNum |= (outExp << 23); // Adjust the exponent bits to the correct position and logical OR with mantissa
			if (outNum == 0) {
				outNum = as_uint(0x3f800000); // For the interval (0, 1], convert 0.0 to 1.0 if detected
			}
			dst[gid] = outNum; // Store result to output buffer
		}
	}
}

// Kernel functions for generating packed float from uint in the buffer of the PRNG
kernel void CopySlowUint01CCAsFlt(global uint* dst, global uint* src, uint count, uint offset, uint buf_size) {
	// dst: destination buffer which we will store the bit pattern of the float
	// src: source buffer containing the uint we want to convert
	// count: how many floats we need to generate
	// offset: offset to get the first valid uint in source buffer
	// buf_size: total number of uint the source buffer can hold
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	uint end_idx = buf_size - 5;

	for (uint idx = gid; idx < count; idx+=gsize) {
		// Each float needs 128+32 = 160 bits to generate (5 bytes)
		// Each work-item will read 5 consecutive uint from the buffer
		uint lidx = idx * 5 + offset;
		if (lidx < end_idx) {
			uint outExp = 126;
			uint tmpUintBuf;
			uint trailBits = src[lidx+4];
			bool flag = true;

			// Get first uint to generate mantissa
			uint outNum = src[lidx];
			outNum &= 0x007fffff;

			// Compute the exponent...
			// First, check for the uint that will be used to compute the actual exponent
			tmpUintBuf = src[lidx+1];

			// Check for the first 32 bits
			if (tmpUintBuf != as_uint(0xffffffff)) {
				flag = false;
			} else {
				tmpUintBuf = src[lidx+2];
				outExp = 94;
			}
			// Check for the second 32 bits if first 32-bits are all ones
			if (flag) {
				if (tmpUintBuf != as_uint(0xffffffff)) {
					flag = false;
				} else {
					tmpUintBuf = src[lidx+3];
					outExp = 62;
				}
			}
			// Check for the third 32 bits if first 32-bits are all ones
			if (flag) {
				if (tmpUintBuf != as_uint(0xffffffff)) {
					flag = false;
				} else {
					tmpUintBuf = trailBits;
					outExp = 30;
				}
			}
			// Check for the fourth 32 bits if first 30-bits are all ones
			if (flag) {
				if (tmpUintBuf >= as_uint(0x3fffffff)) {
					outExp = 0;
				} else {
					flag = false;
				}
			}
			if (!flag) {
				outExp = exponent_adj(tmpUintBuf, outExp);
			}
			outNum |= (outExp << 23); // Adjust the exponent bits to the correct position and logical OR with mantissa
			if (outNum == 0) {
				if ((trailBits & 0x70000000) != 0) {
					outNum = as_uint(0x3f800000); // For the interval [0, 1], randomly convert 0.0 to 1.0 if detected
				}
			}
			dst[gid] = outNum; // Store result to output buffer
		}
	}
}

// Kernel functions for generating packed float from uint in the buffer of the PRNG
kernel void CopySlowUint01COAsDbl(global uint2* dst, global uint* src, uint count, uint offset, uint buf_size) {
	// dst: destination buffer which we will store the bit pattern of the float
	// src: source buffer containing the uint we want to convert
	// count: how many floats we need to generate
	// offset: offset to get the first valid uint in source buffer
	// buf_size: total number of uint the source buffer can hold
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	uint end_idx = buf_size - 34;

	for (uint idx = gid; idx < count; idx+=gsize) {
		// Each double needs 1024+64 = 1088 bits to generate (34 bytes)
		// Each work-item will read 5 consecutive uint from the buffer
		uint lidx = idx * 34 + offset;
		if (lidx < end_idx) {
			uint2 outNum;
			uint outExp = 1022;
			uint tmpUintBuf;

			// Get first two uint to generate mantissa
			outNum.x = src[lidx];
			outNum.y = src[lidx+1];
			outNum.x &= 0x000fffff;
			// Compute the exponent...
			// First, check for the uint that will be used to compute the actual exponent
			for (uint ii = 2; ii < 35; ii++) {
				tmpUintBuf = src[lidx+ii];
				if ((ii == 34) && (tmpUintBuf >= as_uint(0x3fffffff))) {
					outExp = 0;
				} else if (tmpUintBuf == as_uint(0xffffffff)) {
					outExp -= 32;
				} else {
					break;
				}
			}

			if (outExp > 0)	{
				outExp = exponent_adj(tmpUintBuf, outExp);
			}
			outNum.x |= (outExp << 20); // Adjust the exponent bits to the correct position and logical OR with mantissa
			dst[gid] = outNum; // Store result to output buffer
		}
	}
}

// Kernel functions for generating packed float from uint in the buffer of the PRNG
kernel void CopySlowUint01OCAsDbl(global uint2* dst, global uint* src, uint count, uint offset, uint buf_size) {
	// dst: destination buffer which we will store the bit pattern of the float
	// src: source buffer containing the uint we want to convert
	// count: how many floats we need to generate
	// offset: offset to get the first valid uint in source buffer
	// buf_size: total number of uint the source buffer can hold
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	uint end_idx = buf_size - 34;

	for (uint idx = gid; idx < count; idx+=gsize) {
		// Each double needs 1024+64 = 1088 bits to generate (34 bytes)
		// Each work-item will read 5 consecutive uint from the buffer
		uint lidx = idx * 34 + offset;
		if (lidx < end_idx) {
			uint2 outNum;
			uint outExp = 1022;
			uint tmpUintBuf;

			// Get first two uint to generate mantissa
			outNum.x = src[lidx];
			outNum.y = src[lidx+1];
			outNum.x &= 0x000fffff;

			// Compute the exponent...
			// First, check for the uint that will be used to compute the actual exponent
			for (uint ii = 2; ii < 35; ii++) {
				tmpUintBuf = src[lidx+ii];
				if ((ii == 34) && (tmpUintBuf >= as_uint(0x3fffffff))) {
					outExp = 0;
				} else if (tmpUintBuf == as_uint(0xffffffff)) {
					outExp -= 32;
				} else {
					break;
				}
			}

			if (outExp > 0)	{
				outExp = exponent_adj(tmpUintBuf, outExp);
			}
			outNum.x |= (outExp << 20); // Adjust the exponent bits to the correct position and logical OR with mantissa
			if ((outNum.x == as_uint(0x00000000)) && (outNum.y == as_uint(0x00000000))) {
				outNum.x = as_uint(0x3ff00000); // For the interval (0, 1], convert 0.0 to 1.0 if detected
			}
			dst[gid] = outNum; // Store result to output buffer
		}
	}
}

// Kernel functions for generating packed float from uint in the buffer of the PRNG
kernel void CopySlowUint01CCAsDbl(global uint2* dst, global uint* src, uint count, uint offset, uint buf_size) {
	// dst: destination buffer which we will store the bit pattern of the float
	// src: source buffer containing the uint we want to convert
	// count: how many floats we need to generate
	// offset: offset to get the first valid uint in source buffer
	// buf_size: total number of uint the source buffer can hold
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	uint end_idx = buf_size - 34;

	for (uint idx = gid; idx < count; idx+=gsize) {
		// Each double needs 1024+64 = 1088 bits to generate (34 bytes)
		// Each work-item will read 5 consecutive uint from the buffer
		uint lidx = idx * 34 + offset;
		if (lidx < end_idx) {
			uint2 outNum;
			uint outExp = 1022;
			uint tmpUintBuf;

			// Get first two uint to generate mantissa
			outNum.x = src[lidx];
			outNum.y = src[lidx+1];
			outNum.x &= 0x000fffff;

			// Get the last set of uint for breaking 0.0 and 1.0 at the end
			uint trailBits = src[lidx+33];

			// Compute the exponent...
			// First, check for the uint that will be used to compute the actual exponent
			for (uint ii = 2; ii < 34; ii++) {
				tmpUintBuf = (ii == 33) ? trailBits : src[lidx+ii];
				if ((ii == 33) && (tmpUintBuf >= as_uint(0x3fffffff))) {
					outExp = 0;
				} else if (tmpUintBuf == as_uint(0xffffffff)) {
					outExp -= 32;
				} else {
					break;
				}
			}

			if (outExp > 0)	{
				outExp = exponent_adj(tmpUintBuf, outExp);
			}
			outNum.x |= (outExp << 20); // Adjust the exponent bits to the correct position and logical OR with mantissa
			if ((outNum.x == as_uint(0x00000000)) && (outNum.y == as_uint(0x00000000))) {
				if ((trailBits & 0x70000000) != 0) {
					outNum.x = as_uint(0x3ff00000);  // For the interval [0, 1], randomly convert 0.0 to 1.0 if detected
				}
			}
			dst[gid] = outNum; // Store result to output buffer
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////
/*
Define function to calculate inverse CDF of standard Gaussian distribution.
Taken from PhD thesis of Thomas Luu (Department of Mathematics at Universty College
of London). These are the same as the hybrid approximation functions in the thesis.
*/
////////////////////////////////////////////////////////////////////////////////////
inline float normcdfinv_float(float u) {
	float	v, p, q, ushift, tmp;

	tmp = u;

	if (u < 0.0f || u > 1.0f) {
		return NAN;
	}
	if (u <= 0.0f) {
		return FLT_MIN;// Float.NEGATIVE_INFINITY;
	}
	if (u >= 1.0f) {
		return FLT_MAX; // Float.POSITIVE_INFINITY;
	}

	ushift = tmp - 0.5f;

	v = copysign(ushift, 0.0f);
	
	if (v < 0.499433f) {
		v = rsqrt((-tmp*tmp) + tmp);
		v *= 0.5f;

		p = 0.001732781974270904f;
		p = p * v + 0.1788417306083325f;
		p = p * v + 2.804338363421083f;
		p = p * v + 9.35716893191325f;
		p = p * v + 5.283080058166861f;
		p = p * v + 0.07885390444279965f;
		p *= ushift;

		q = 0.0001796248328874524f;
		q = q * v + 0.02398533988976253f;
		q = q * v + 0.4893072798067982f;
		q = q * v + 2.406460595830034f;
		q = q * v + 3.142947488363618f;
	} else {
		if (ushift > 0.0f) {
			tmp = 1.0f - tmp;
		}
		v = log2(tmp+tmp);
		v *= -0.6931471805599453f;
		if (v < 22.0f) {
			p = 0.000382438382914666f;
			p = p * v + 0.03679041341785685f;
			p = p * v + 0.5242351532484291f;
			p = p * v + 1.21642047402659f;

			q = 9.14019972725528e-6f;
			q = q * v + 0.003523083799369908f;
			q = q * v + 0.126802543865968f;
			q = q * v + 0.8502031783957995f;
		} else {
			p = 0.00001016962895771568f;
			p = p * v + 0.003330096951634844f;
			p = p * v + 0.1540146885433827f;
			p = p * v + 1.045480394868638f;

			q = 1.303450553973082e-7f;
			q = q * v + 0.0001728926914526662f;
			q = q * v + 0.02031866871146244f;
			q = q * v + 0.3977137974626933f;
		}
		p *= copysign(v, ushift);
	}
	q = q * v + 1.0f;
	v = 1.0f / q;
	return p * v;
} // normcdfinv_float
////////////////////////////////////////////////////////////////////////////////////

// Kernel functions for fast copy of Gaussian float while copying between buffers
kernel void CopyFastUint01OOAsNormFlt(global float* dst, global uint* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	float localVal;
	
	for (uint ii = 0; ii < count; ii += gsize) {
		localVal = simple_oo_01_uint(src[ii]);
		dst[ii] = normcdfinv_float(localVal);
	}
}

kernel void CopyFastUlong01OOAsNormFlt(global float* dst, global ulong* src, uint count) {
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	float localVal;
	
	for (uint ii = 0; ii < count; ii += gsize) {
		localVal = simple_oo_01_ulong(src[ii]);
		dst[ii] = normcdfinv_float(localVal);
	}
}

// Kernel function for generating normally distributed packed float
// from uint in the buffer of the PRNG
kernel void CopySlowUint01AsNormFlt(global uint* dst, global uint* src, uint count, uint offset, uint buf_size) {
	// dst: destination buffer which we will store the bit pattern of the float
	// src: source buffer containing the uint we want to convert
	// count: how many floats we need to generate
	// offset: offset to get the first valid uint in source buffer
	// buf_size: total number of uint the source buffer can hold
	uint gid = get_global_id(0);
	uint gsize = get_global_size(0);
	uint end_idx = buf_size - 5;

	for (uint idx = gid; idx < count; idx+=gsize) {
		// Each float needs 128+32 = 160 bits to generate (5 bytes)
		// Each work-item will read 5 consecutive uint from the buffer
		uint lidx = idx * 5 + offset;
		if (lidx < end_idx) {
			// Get first uint to generate mantissa
			uint outNum = src[lidx];
			outNum &= 0x007fffff;
			// Compute the exponent...
			// First, check for the uint that will be used to compute the actual exponent
			uint outExp = 126;
			uint tmpUintBuf;
			tmpUintBuf = src[lidx+1];
			bool flag = true;

			// Check for the first 32 bits
			if (tmpUintBuf != as_uint(0xffffffff)) {
				flag = false;
			} else {
				tmpUintBuf = src[lidx+2];
				outExp = 94;
			}
			// Check for the second 32 bits if first 32-bits are all ones
			if (flag) {
				if (tmpUintBuf != as_uint(0xffffffff)) {
					flag = false;
				} else {
					tmpUintBuf = src[lidx+3];
					outExp = 62;
				}
			}
			if (flag) {
				if (tmpUintBuf != as_uint(0xffffffff)) {
					flag = false;
				} else {
					tmpUintBuf = src[lidx+4];
					outExp = 30;
				}
			}
			if (flag) {
				if (tmpUintBuf >= as_uint(0x3fffffff)) {
					outExp = 0;
				} else {
					flag = false;
				}
			}
			if (!flag) {
			outExp = exponent_adj(tmpUintBuf, outExp);
			}
			outNum |= (outExp << 23); // Adjust the exponent bits to the correct position and logical OR with mantissa
			dst[gid] = normcdfinv_float(outNum); // Store result to output buffer
		}
	}
}
////////////////////////////////////////////////////////////////////////////////////

)EOK";