/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file ExSUM.MIC.hpp
 *  \brief Provides a set of routines for working with vector types on MIC.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */
#ifndef EXSUM_MIC_HPP_
#define EXSUM_MIC_HPP_

#include <algorithm>
#include <cmath>
#include "superaccumulator.hpp"
#include "mylibm.hpp"
#include "vectorf512.h"

template<int N>
struct FPLargeBaseMIC
{
    // MIC: 2x8x64-bit FP
    //       16x32-bit int
    // Base B=2^(32-K)
    typedef Vec16i limb_t; /**< sort of floating-point expansion */
    static constexpr unsigned int K = 4; /**< size of secure digits that hold overflows */
    static constexpr unsigned int digits = 32 - K; /**< size of working digits */

    /**
     * Constructor
     * \param sa superaccumulator
     * \param exponent size of floating-point exponent
     */
    // exponent is LSB in LIMBS !
    FPLargeBaseMIC(Superaccumulator & sa, int exponent = -3);

    /** 
     * This function accumulates two arrays of values to the floating-point expansion
     * \param x1 input vector of values
     * \param x2 input vector of values
     */
    void Accumulate(Vec8d x1, Vec8d x2);

    /** 
     * This function accumulates an array of values to the floating-point expansion
     * \param x input array of values
     * \param n  size of the array
     */
    void Accumulate(double const * x, int n);

    /**
     * This function is used to flush the floating-point expansion to the superaccumulator
     */
    void Flush();

private:
    void Reset(int e);
    void InternalAccumulate(Vec8d x1, Vec8d x2);
    void Normalize();
    
    Superaccumulator & superacc;

    // All limbs are signed (2's cplt)
    limb_t limbs[N];
    int exponent;   // In digits
    
    Vec8d inputScale, deltaScale;   // Not double, would generate extra broadcast insn
    Vec8d minScaleBinary64, maxScaleBinary64;
    unsigned int ovfCounter;
    static constexpr unsigned int ovfCounterMax = (1 << K);  // Count 1 extra accum cycle to make carry propagation overflow-less too
};

template<int N> UNROLL_ATTRIBUTE
FPLargeBaseMIC<N>::FPLargeBaseMIC(Superaccumulator & sa, int exponent) :
    superacc(sa),
    exponent(exponent),
    inputScale(exp2i(-digits * (exponent + N - 1))),
    deltaScale(exp2i(digits)),
    minScaleBinary64(exp2i(digits * exponent + 53)),
    maxScaleBinary64(exp2i(digits * (exponent + N) - 1)),
    ovfCounter(ovfCounterMax)
{
    std::fill(limbs, limbs + N, 0);
}


template<int N> UNROLL_ATTRIBUTE
void FPLargeBaseMIC<N>::Reset(int e)
{
    exponent = e;
    minScaleBinary64 = exp2i(digits * exponent + 53);
    maxScaleBinary64 = exp2i(digits * (exponent + N) - 1);
    inputScale = exp2i(-digits * (exponent + N - 1));
    deltaScale = exp2i(digits);
   
    std::fill(limbs, limbs+N, 0);
    ovfCounter = ovfCounterMax;
}

// Low-level accumulate. Assumptions:
// - FP fits (Fit(x) == true)
// - Free ovf bits (ovfCounter > 0)
template<int N> UNROLL_ATTRIBUTE
void FPLargeBaseMIC<N>::InternalAccumulate(Vec8d x1, Vec8d x2)
{
    Vec8d xscaled1 = x1 * inputScale;
    Vec8d xscaled2 = x2 * inputScale;
    
    // TODO: can we fuse scaling and rounding using fma(x, scale, double(3<<51))?
    //   - no, because we need to keep the intermediate scaled value
    //       then {5555} shuffle cdab (extract low parts)?
    
    // Starting from MSB, extract and cancel out leading bits
    // This loop is supposed to be unrolled!
    for(int i = N - 1; i >= 0; --i) {

        limb_t xint_lo = round_to_int(xscaled1);   // No overflow
        limb_t xint_hi = round_to_int(xscaled2);
        limb_t xint = _mm512_mask_permute4f128_epi32(xint_lo, Vec16b(0xff00), xint_hi, _MM_PERM_BADC);  // Compact

        Vec8d xrounded1 = to_double(xint);
        Vec8d xrounded2 = to_double(xint_hi);
        
        limbs[i] += xint;   // Overflow-free 2^K times
        
        xscaled1 -= xrounded1;
        xscaled2 -= xrounded2;
        xscaled1 *= deltaScale;  // Might use absolute scale factors to cut dep chain
        xscaled2 *= deltaScale;
    }
}

// Returns overflowing bits (weight digit * (exponent + N))
template<int N> UNROLL_ATTRIBUTE
void FPLargeBaseMIC<N>::Normalize()
{
    // Propagate carries/borrows:
    // K upper bits
    
    // Carry out does not depend on carry in
    // Supposed-to-be-Unrolled loop
    limb_t carry_in = limbs[0] >> digits;
    limbs[0] -= carry_in << digits;
    for(unsigned int i = 1; i != N; ++i)
    {
        limb_t carry_out = limbs[i] >> digits;    // Arithmetic shift
        limbs[i] += carry_in - (carry_out << digits);
        carry_in = carry_out;
    }

    // Flush overflow bits to the accumulator
    if(unlikely(horizontal_or(carry_in != 0))) {
        int sumovf = horizontal_add(carry_in);   // No overflow as 2^(32-K)>vec len
        superacc.Accumulate(sumovf, (exponent + N) * digits);
    }
}

template<int N> UNROLL_ATTRIBUTE
void FPLargeBaseMIC<N>::Flush()
{
    Normalize();    // Avoid carry in hadd (assumes 2^K>vec len)
    for(unsigned int i = 0; i != N; ++i)
    {
        int sum = horizontal_add(limbs[i]);
        superacc.Accumulate(sum, (exponent + i) * digits);
        limbs[i] = 0;
    }
}

// External interface
template<int N>
void FPLargeBaseMIC<N>::Accumulate(Vec8d x1, Vec8d x2)
{
    // Check overflow counter
    if(unlikely(--ovfCounter == 0))
    {
        Normalize();
        ovfCounter = ovfCounterMax;
    }
#if 1
    // Check bounds
    Vec8d xabs1 = abs(x1);
    Vec8d xabs2 = abs(x2);
    Vec8b toobig1 = _mm512_cmpnlt_pd_mask(xabs1, maxScaleBinary64);
    Vec8b toobig2 = _mm512_cmpnlt_pd_mask(xabs2, maxScaleBinary64);
    Vec8b toosmall1 = xabs1 < minScaleBinary64;
    Vec8b toosmall2 = xabs2 < minScaleBinary64;
    
    Vec8b out1 = toobig1|toosmall1;
    Vec8b out2 = toobig2|toosmall2;
    // TODO: align accumulator on largest (shift window left)
    
    if(unlikely(!_mm512_kortestz(out1, out2))) {
        // TODO: Optimization: run fast path on !(toobig|toosmall) elements,
        // conditionally acccumulate the others
        double dx[8]  __attribute__((aligned(64)));
        int n = x1.store_compact_a(dx, out1);
        for(unsigned int i = 0; i != n; ++i) {
            superacc.Accumulate(dx[i]);
        }
        // Remove from vector
        x1 = select(out1, Vec8d(0.), x1);

        n = x2.store_compact_a(dx, out2);
        for(unsigned int i = 0; i != n; ++i) {
            superacc.Accumulate(dx[i]);
        }
        x2 = select(out2, Vec8d(0.), x2);
    }
#endif
    // Fast path
    InternalAccumulate(x1, x2);
}

template<int N>
inline void FPLargeBaseMIC<N>::Accumulate(double const * p, int n)
{
    assert(!((size_t)p & 0x3f) && !((n+1) & 0xf)); // 128B-aligned
    
    for(int i = 0; i < n; i+=16) {
        Vec8d x1 = Vec8d().load_a(p + i);
        Vec8d x2 = Vec8d().load_a(p + i + 8);
        Accumulate(x1, x2);
    }
}

#endif
