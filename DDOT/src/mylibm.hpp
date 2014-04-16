#ifndef MYLIBM_HPP_INCLUDED
#define MYLIBM_HPP_INCLUDED

#include <stdint.h>
#include <immintrin.h>
#include <cassert>

#ifdef ATT_SYNTAX
#define ASM_BEGIN ".intel_syntax;"
#define ASM_END ";.att_syntax"
#else
#define ASM_BEGIN
#define ASM_END
#endif

// Debug mode
#define paranoid_assert(x) assert(x)

// Making C code less readable in an attempt to make assembly more readable
#if 1
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

inline uint64_t rdtsc()
{
	uint32_t hi, lo;
	asm volatile ("rdtsc" : "=a"(lo), "=d"(hi));
	return lo | ((uint64_t)hi << 32);
}


inline int64_t myllrint(double x) {
#ifdef AMD
     return llrint(x);
#else
     return _mm_cvtsd_si64(_mm_set_sd(x));
#endif
}

template<typename T>
inline T mylrint(double x) { assert(false); }

template<>
inline int64_t mylrint<int64_t>(double x) {
    return lrint(x);
//    return _mm_cvtsd_si64(_mm_set_sd(x));
}

template<>
inline int32_t mylrint<int32_t>(double x) {
    return lrint(x);
//    return _mm_cvtsd_si32(_mm_set_sd(x));
}


inline double myrint(double x)
{
    return rint(x);
/*
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
    asm(ASM_BEGIN "roundsd %0, %1, 0" ASM_END : "=x" (r) : "x" (x));
    return r;
#endif
*/
}

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

inline int biased_exponent(double x)
{
    union {
        double d;
        uint64_t i;
    } caster;
    caster.d = x;
    uint64_t e = (caster.i >> 52) & 0x7ff;
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
inline static int64_t xadd(int64_t &memref, int64_t x, unsigned char &of)
{
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
/*
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
*/
    int64_t y = memref;
    memref += x;

    // TODO: cover also underflow
    of = 0;
    if(x > 0 && y > 0 && memref < 0)
        of = 1;
    if(x < 0 && y < 0 && memref > 0)
        of = 1;

    return y;
}
#endif
