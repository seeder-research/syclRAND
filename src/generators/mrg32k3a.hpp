typedef struct {
   ulong g1[3];
   ulong g2[3];
} mrg32k3a_state;

const char * mrg32k3a_prng_kernel = R"EOK(
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Thomas Bradley, Parallelisation Techniques for Random Number Generators
// https://www.nag.co.uk/IndustryArticles/gpu_gems_article.pdf

#define MRG32K3A_M1 4294967087
#define MRG32K3A_M2 4294944443
#define MRG32K3A_NORM_DOUBLE (2.3283065498378288e-10) // 1/MRG32K3A_M1
#define MRG32K3A_UINT_NORM (1.000000048661607) // (MRG32K3A_POW32 - 1)/(MRG32K3A_M1 - 1)

/**
State of mrg32k3a RNG.
*/
typedef struct {
   ulong g1[3];
   ulong g2[3];
} mrg32k3a_state;

/**
Generates a random 32-bit unsigned integer using mrg32k3a RNG.

@param state State of the RNG to use.
*/
#define mrg32k3a_uint(state) (uint)_mrg32k3a_ulong(&state)
#define mrg32k3a_ulong(state) _mrg32k3a_ulong(&state)
ulong _mrg32k3a_ulong(mrg32k3a_state* state){

    ulong* g1 = state->g1;
    ulong* g2 = state->g2;
    long p0, p1;
    
    /* component 1 */
    p0 = 1403580 * state->g1[1] - 810728 * state->g1[0];
    p0 %= MRG32K3A_M1;
    if (p0 < 0)
        p0 += MRG32K3A_M1;
    g1[0] = g1[1];
    g1[1] = g1[2];
    g1[2] = p0;

    /* component 2 */
    p1 = 527612 * g2[2] - 1370589 * g2[0];
    p1 %= MRG32K3A_M2;
    if (p1 < 0)
        p1 += MRG32K3A_M2;
    g2[0] = g2[1];
    g2[1] = g2[2];
    g2[2] = p1;

    return (p0 - p1) + (p0 <= p1 ? MRG32K3A_M1 : 0);
}

/**
Seeds mrg31k3p RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void mrg32k3a_seed(mrg32k3a_state* state, ulong j){
    ulong* g1 = state->g1;
    ulong* g2 = state->g2;
    g1[0] = j % MRG32K3A_M1;
    g1[1] = 1;
    g1[2] = 1;
    g2[0] = 1;
    g2[1] = 1;
    g2[2] = 1;
}
)EOK";
