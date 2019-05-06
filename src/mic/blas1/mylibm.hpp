/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file mic/blas1/mylibm.hpp
 *  \brief Provides a set of auxiliary functions to superaccumulation.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef MYLIBM_HPP_INCLUDED
#define MYLIBM_HPP_INCLUDED

#include <stdint.h>
#include <immintrin.h>
#include <cassert>
#include "vectorf512.h"

//#undef assert
//#define assert(x) 

//#ifdef __GNUC__
#if 0
#define UNROLL_ATTRIBUTE __attribute__((optimize("unroll-loops")))
#else
#define UNROLL_ATTRIBUTE
#endif
#define ALIGNED_ATTRIBUTE(n) __attribute__((aligned(n)))
#define INLINE_ATTRIBUTE __attribute__((always_inline))

// Debug mode
//#define paranoid_assert(x) assert(x)
#define paranoid_assert(x)

// Making C code less readable in an attempt to make assembly more readable
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

inline uint64_t rdtsc()
{
	uint32_t hi, lo;
	asm volatile ("rdtsc" : "=a"(lo), "=d"(hi));
	return lo | ((uint64_t)hi << 32);
}

#ifdef __SSE2__

inline int64_t myllrint(double x) {
    return _mm_cvtsd_si64(_mm_set_sd(x));
}

template<typename T>
inline T mylrint(double x) { assert(false); }

template<>
inline int64_t mylrint<int64_t>(double x) {
    return _mm_cvtsd_si64(_mm_set_sd(x));
}

template<>
inline int32_t mylrint<int32_t>(double x) {
    return _mm_cvtsd_si32(_mm_set_sd(x));
}


inline double myrint(double x)
{
#if 0
    // Workaround gcc bug 51033
    union {
        __m128d v;
        double d[2];
    } r;
    //_mm_round_sd(_mm_undefined_pd(), _mm_set_sd(x));
    //__m128d undefined;
    //r.v = _mm_round_sd(_mm_setzero_pd(), _mm_set_sd(x), _MM_FROUND_TO_NEAREST_INT);
    //r.v = _mm_round_sd(undefined, _mm_set_sd(x), _MM_FROUND_TO_NEAREST_INT);
    r.v = _mm_round_pd(_mm_set_sd(x), _MM_FROUND_TO_NEAREST_INT);
    return r.d[0];
#else
    double r;
    //asm("roundsd $0, %1, %0" : "=x" (r) : "x" (x));
    asm("roundsd %0, %1, 0" : "=x" (r) : "x" (x));
    return r;
#endif
}
#else   // ifdef __SSE2__
#include <cmath>

inline int64_t myllrint(double x) {
    return llrint(x);
}

template<typename T>
inline T mylrint(double x) { assert(false); }

template<>
inline int64_t mylrint<int64_t>(double x) {
    return llrint(x);
}

template<>
inline int32_t mylrint<int32_t>(double x) {
    return lrint(x);
}

inline double myrint(double x)
{
    return rint(x);
}
#endif

inline int exponent(double x)
{
    // simpler frexp
    union {
        double d;
        uint64_t i;
    } caster;
    caster.d = x;
    uint64_t e = ((caster.i >> 52) & 0x7ff) - 0x3ff;
    return e;
}

inline double myldexp(double x, int e)
{
    // Scale x by e
    union {
        double d;
        uint64_t i;
    } caster;
    
    caster.d = x;
    caster.i += (uint64_t)e << 52;
    return caster.d;
}

inline double exp2i(int e)
{
    // simpler ldexp
    union {
        double d;
        uint64_t i;
    } caster;
    
    caster.i = (uint64_t)(e + 0x3ff) << 52;
    return caster.d;
}

// Assumptions: th>tl>=0, no overlap between th and tl
inline static double OddRoundSumNonnegative(double th, double tl)
{
    // Adapted from:
    // Sylvie Boldo, and Guillaume Melquiond. "Emulation of a FMA and correctly rounded sums: proved algorithms using rounding to odd." IEEE Transactions on Computers, 57, no. 4 (2008): 462-471.
    union {
        double d;
        int64_t l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

#ifdef THREADSAFE
#define TSAFE 1
#define LOCK_PREFIX "lock "
#else
#define TSAFE 0
#define LOCK_PREFIX
#endif

// signedcarry in {-1, 0, 1}
inline static int64_t xadd(int64_t & memref, int64_t x, unsigned char & of)
{
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    int64_t oldword = x;
#ifdef ATT_SYNTAX
    asm volatile (LOCK_PREFIX"xaddq %1, %0\n"
        "setob %2"
     : "+m" (memref), "+r" (oldword), "=q" (of) : : "cc", "memory");
#else
    asm volatile (LOCK_PREFIX"xadd %0, %1\n"
        "seto %2"
     : "+m" (memref), "+r" (oldword), "=q" (of) : : "cc", "memory");
#endif
    return oldword;
}

inline static long double mylogbl(long double x)
{
    long double sig, exp;
    //asm volatile ("fxtract" : "=t" (sig), "=u" (exp) : "t" (x) : "st");
    asm ("fxtract" : "=t" (sig), "=u" (exp) : "0" (x)); // Let's play it safe and copy the "fsincos" example
    return exp;
}

//inline static long double myfloorl(long double x)
//{
//}

inline static long double myscalbl(long double x, double exp)
{
    asm ("fscale" : "+t" (x) : "u" (exp) : "st");
    return x;
}

inline static long double myrintl(long double x)
{
    asm ("frndint" : "+t" (x) : : "st");
    return x;
}

inline static int64_t mylrintl(long double x)
{
    int64_t r;
    asm ("fistpll %0" : "=m" (r) : "t" (x) : "st");
    return r;
}

// Input:
// a7 a6 a5 a4 a3 a2 a1 a0
// b7 b6 b5 b4 b3 b2 b1 b0
// Output:
// a7 b7 a5 b5 a3 b3 a1 b1
// a6 b6 a4 b4 a2 b2 a0 b0
inline static void transpose1(Vec8d & a, Vec8d & b)
{
    // a7 -- a5 -- a3 -- a1 -- // a{10101010}
    // -- b7 -- b5 -- b3 -- b1 // b{badc}{01010101}
    Vec8b k1(0x55); // 01010101
    Vec8b k2 = !k1;
    //Vec8d a2 = _mm512_mask_swizzle_pd(a, k1, b, _MM_SWIZ_REG_BADC);
    Vec8d a2 = _mm512_mask_swizzle_pd(a, k1, b, _MM_SWIZ_REG_CDAB);
    // -- b6 -- b4 -- b2 -- b0 // b{01010101}
    // a6 -- a4 -- a2 -- a0 -- // a{badc}{10101010}
    //Vec8d b2 = _mm512_mask_swizzle_pd(b, k2, a, _MM_SWIZ_REG_BADC);
    Vec8d b2 = _mm512_mask_swizzle_pd(b, k2, a, _MM_SWIZ_REG_CDAB);
    a = a2;
    b = b2;
}

// Input:
// a7 a6 a5 a4 a3 a2 a1 a0
// b7 b6 b5 b4 b3 b2 b1 b0
// Output:
// a7 a6 b7 b6 a3 a2 b3 b2
// a5 a4 b5 b4 a1 a0 b1 b0
inline static void transpose2(Vec8d & a, Vec8d & b)
{
    // a7 a6 -- -- a3 a2 -- --  // a{11001100}
    // -- -- b7 b6 -- -- b3 b2  // b{CDAB}{00110011}
    // We use 32-bit permute insn, so use 16-bit masks
    Vec16b k1(0x0f0f);
    Vec16b k2 = !k1;
    Vec16i a1 = _mm512_castpd_si512(a);
    Vec16i b1 = _mm512_castpd_si512(b);
    Vec16i a2 = _mm512_mask_permute4f128_epi32(a1, k1, b1, _MM_PERM_CDAB);
    Vec16i b2 = _mm512_mask_permute4f128_epi32(b1, k2, a1, _MM_PERM_CDAB);
    a = _mm512_castsi512_pd(a2);
    b = _mm512_castsi512_pd(b2);
}

// Input:
// a7 a6 a5 a4 a3 a2 a1 a0
// b7 b6 b5 b4 b3 b2 b1 b0
// Output:
// a7 a6 a5 a4 b7 b6 b5 b4
// a3 a2 a1 a0 b3 b2 b1 b0
inline static void transpose3(Vec8d & a, Vec8d & b)
{
    // a7 a6 a5 a4 -- -- -- --
    // -- -- -- -- b7 b6 b5 b4

    // a3 a2 a1 a0 -- -- -- --
    // -- -- -- -- b3 b2 b1 b0
    Vec16b k1(0x00ff);
    Vec16b k2 = !k1;
    Vec16i a1 = _mm512_castpd_si512(a);
    Vec16i b1 = _mm512_castpd_si512(b);
    Vec16i a2 = _mm512_mask_permute4f128_epi32(a1, k1, b1, _MM_PERM_BADC);
    Vec16i b2 = _mm512_mask_permute4f128_epi32(b1, k2, a1, _MM_PERM_BADC);
    a = _mm512_castsi512_pd(a2);
    b = _mm512_castsi512_pd(b2);
}


#endif
