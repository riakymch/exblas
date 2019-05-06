/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file mic/blas1/superaccumulator.hpp
 *  \brief Provides a class to work with superaccumulators
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef SUPERACCUMULATOR_HPP_INCLUDED
#define SUPERACCUMULATOR_HPP_INCLUDED

#include <vector>
#include <stdint.h>
#include <iosfwd>
#include <cmath>
#include <cstdio>
#include "mylibm.hpp"
#include "vectorf512.h"

#define TRACK_BOUNDS 0

/**
 * \struct Superaccumulator
 * \ingroup xsum
 * \brief This class is meant to provide functionality for working with superaccumulators
 */
// TODO: generalize for other int64_t, such as int32_t...
//       do we need to? Superaccumulator is scalar.
struct Superaccumulator
{
    /** 
     * Construction 
     * \param e_bits maximum exponent 
     * \param f_bits maximum exponent with significand
     */
    Superaccumulator(int e_bits = 1023, int f_bits = 1023 + 52);
    
    /**
     * Function for accumulating values into superaccumulator
     * \param x value
     * \param exp exponent
     */ 
    void Accumulate(int64_t x, int exp);

    /**
     * Function for accumulating values into superaccumulator
     * \param x double-precision value
     */ 
    void Accumulate(double x);

    /**
     * Function for adding another supperaccumulator into the current
     * \param other superaccumulator
     */ 
    void Accumulate(Superaccumulator & other);   // May modify (normalize) other member

    /**
     * Function to perform correct rounding
     */
    double Round();
    
    /**< Characterizes the result of summation */
    enum Status
    {
        Exact, /**< Reproducible and accurate */
        Inexact, /**< non-accurate */
        MinusInfinity, /**< minus infinity */
        PlusInfinity, /**< plus infinity */
        Overflow, /**< overflow occurred */
        sNaN, /**< not-a-number */
        qNaN /**< not-a-number */
    };
 
    /**
     * Function to normalize the superaccumulator
     */
    bool Normalize();

    /**
     * Function to print the superaccumulator
     */
    void Dump(std::ostream & os);

    /**
     * Returns f_words
     */
    int get_f_words();

    /**
     * Returns e_words
     */
    int get_e_words();

    /*
     * Prefetching function
     */   
    void Prefetch() const;

private:
    void AccumulateWord(int64_t x, int i);

    static constexpr unsigned int K = 8;    // High-radix carry-save bits
    static constexpr int digits = 64 - K;
    static constexpr double deltaScale = double(1ull << digits); // Assumes K>0


    int f_words, e_words;
    std::vector<int64_t> accumulator;
    int imin, imax;
    Status status;

    int64_t overflow_counter;
};


// icc has trouble with scalar code, let's use intrinsics
// ... wait, there is no such thing as x87 intrinsics :/
INLINE_ATTRIBUTE inline void Superaccumulator::Accumulate(double x)
{
    assert(imin >= 0 && imax < f_words + e_words);
    if(x == 0) return;

    int e = exponent(x);                            // TODO: use fxtract (logb)
    int exp_word = e / digits;  // Word containing MSbit    // TODO: use fmul by constant 80-bit, or even fdiv? + frndint?
    int iup = exp_word + f_words;
    
    double xscaled = myldexp(x, -digits * exp_word);    // TODO: use fscale

    if(TRACK_BOUNDS && iup > imax) imax = iup;
    int i;
    for(i = iup; xscaled != 0; --i) {
        int64_t xint;
        
        xint = (int64_t)xscaled;    // Wrong rounding mode
        double xrounded = (double)xint;
        AccumulateWord(xint, i);
        
        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
    if(TRACK_BOUNDS && i + 1 < imin) imin = i + 1;
}

INLINE_ATTRIBUTE inline void Superaccumulator::AccumulateWord(int64_t x, int i)
{
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    //assert(exp_word < e_words && exp_word >= -f_words);
    assert(i >= 0 && i < e_words + f_words);
    int64_t carry = x;
    int64_t carrybit;
    unsigned char overflow;
    int64_t oldword = xadd(accumulator[i], x, overflow);
    while(unlikely(overflow))
    {
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // accumulator[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> digits;    // Arithmetic shift
        bool s = oldword > 0;
        carrybit = (s ? 1ll << K : -1ll << K);
        
        // Cancel carry-save bits
        //accumulator[i] -= (carry << digits);
        xadd(accumulator[i], -(carry << digits), overflow);
        //assert(TSAFE || !(s ^ overflow));
        if(TSAFE && unlikely(s ^ overflow)) {
            // (Another) overflow of sign S
            carrybit *= 2;
        }
        // overflow only when S=?
        
        carry += carrybit;

        ++i;
        if(i >= f_words + e_words) {
            status = Overflow;
            return;
        }
        oldword = xadd(accumulator[i], carry, overflow);
    }
}

inline void Superaccumulator::Prefetch() const
{
    for(int i = 0; i < f_words + e_words; i += 8) {
	_mm_prefetch((char const*)&accumulator[i], _MM_HINT_T0);
    }
}

// Assumption: |x| < 2^52
inline static Vec8q lrint_small(Vec8d x) {
    Vec8d unit = copysign_pos(Vec8d(1ull << 52), x);
    Vec8d xunnorm = unit + x;   // Align mantissa on 2^52
    Vec8q xint = Vec8q(Vec512b(_mm512_castpd_si512(xunnorm)) & Vec512b(Vec8q((1ull << 52) - 1)));  // Extract mantissa
    return xint;
}

inline static int64_t horizontal_add_small(Vec8q a) {
    // Split in 26-26
    // ah' = (ah << 6) | (al >> 26)
    // al' = al & ((1<<26)-1)
    Vec16b khi(0xaaaa);
    Vec16b klo = !khi;
    Vec16i a2 = Vec16i(a);
    a2 = _mm512_mask_slli_epi32(a, khi, a, 6);
    Vec16i a2h = _mm512_mask_srli_epi32(a2, khi, _mm512_swizzle_epi32(a2, _MM_SWIZ_REG_CDAB), 26);
    a2 = _mm512_mask_or_epi32(a2, khi, a2, a2h);
    a2 = _mm512_mask_and_epi32(a2, klo, a2, Vec16i((1 << 26) - 1));

    // Reduction steps
    a2 = _mm512_add_epi32(a2, _mm512_swizzle_epi32(a2, _MM_SWIZ_REG_BADC));
    a2 = a2 + _mm512_permute4f128_epi32(a2, _MM_PERM_CDAB);
    a2 = a2 + _mm512_permute4f128_epi32(a2, _MM_PERM_BADC);
    
    // Now add (ah >> 6) + al
    int32_t x[16] ALIGNED_ATTRIBUTE(64);
    a2.store_a(x);
    int64_t al = x[0];
    int64_t ah = x[1];
    return (ah >> 6) + al;    
}

inline int Superaccumulator::get_f_words() {
    return f_words;
}

inline int Superaccumulator::get_e_words() {
    return e_words;
}

#endif
