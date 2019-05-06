/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file mic/blas1/ExSUM.FPE.hpp
 *  \brief Provides a set of routines concerning floating-point expansions
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */
#ifndef ExSUM_FPE_HPP_
#define ExSUM_FPE_HPP_

/**
 * \struct FPExpansionVect
 * \ingroup ExSUM
 * \brief This struct is meant to introduce functionality for working with
 *  floating-point expansions in conjuction with superaccumulators
 */
template<typename T, int N, bool EX=false>
struct FPExpansionVect
{
    typedef T limb_t;

    /**
     * Constructor
     * \param sa superaccumulator
     */
    FPExpansionVect(Superaccumulator & sa);

    /** 
     * This function accumulates value x to the floating-point expansion
     * \param x input value
     */
    void Accumulate(T x);

    /** 
     * This function accumulates two values x to the floating-point expansion
     * \param x1 input value
     * \param x2 input value
     */
    void Accumulate(T x1, T x2);

    /** 
     * This function accumulates value x to 
     * \param x input array of values
     * \param n  number of elementes
     */
    void Accumulate(double const * x, int n);

    /**
     * This function is used to flush the floating-point expansion to the superaccumulator
     */
    void Flush();

    /**
     * This function is meant to be used for printing the floating-point expansion
     */
    void Dump() const;

private:
    void FlushVector(T x) const;
    void DumpVector(T x) const;

    Superaccumulator & superacc;

    // Most significant digits first!
    limb_t a[N];
};

template<typename T, int N, bool EX>
FPExpansionVect<T,N,EX>::FPExpansionVect(Superaccumulator & sa) :
    superacc(sa)
{
    std::fill(a, a + N, 0);
}

#if 1
// Knuth 2Sum. 6 vops.
template<typename T>
inline static T twosum(T a, T b, T & s)
{
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
}
#elif 0
// Vector impl of fast2sum. 8 vops.
template<typename T>
inline static T twosum(T a, T b, T & s)
{
    T r = a + b;
    auto doswap = abs(b) > abs(a);
    T a2 = select(doswap, b, a);
    T b2 = select(doswap, a, b);
    s = (a2 - r) + b2;
    return r;
}
#else
// Vector impl of fast2sum with check. 6 vops + 1 sop.
// Might be faster once icc perf bugs with __mmask8 are resolved
inline static Vec8d twosum(Vec8d a, Vec8d b, Vec8d & s)
{
    Vec8d r = a + b;
    Vec8b doswap = abs(b) > abs(a);
    if(horizontal_or(doswap)) {
        Vec8d a2 = select(doswap, b, a);
        Vec8d b2 = select(doswap, a, b);
        a = a2;
        b = b2;
    }
    s = (a - r) + b;
    return r;
}
#endif

inline static void horizontal_twosum(Vec8d & r, Vec8d & s)
{
    //r = twosum(r, s, s);
    transpose1(r, s);
    r = twosum(r, s, s);
    transpose2(r, s);
    r = twosum(r, s, s);
    transpose3(r, s);
    r = twosum(r, s, s);
}

template<typename T, int N, bool EX> UNROLL_ATTRIBUTE
void FPExpansionVect<T,N,EX>::Accumulate(T x)
{
    T s;
    for(unsigned int i = 0; i != N; ++i) {
        a[i] = twosum(a[i], x, s);
        x = s;
        if(EX && !horizontal_or(x != 0)) return;
    }
    if(EX || unlikely(horizontal_or(x != 0))) {
        FlushVector(x);
    }
}

template<typename T, int N, bool EX> UNROLL_ATTRIBUTE INLINE_ATTRIBUTE
inline void FPExpansionVect<T,N,EX>::Accumulate(T x1, T x2)
{
    T xl;
    T xh = twosum(x1, x2, xl);
    T s1,s2;
    for(unsigned int i = 0; i < N - 1; ++i) {
        a[i] = twosum(a[i], xh, s1);
        xh = s1;
        if(EX && i > 0 && likely(!horizontal_or((Vec16i(_mm512_castpd_si512(xh)) | Vec16i(_mm512_castpd_si512(xl))) != 0))) return;
        a[i+1] = twosum(a[i+1], xl, s2);
        xl = s2;
    }
    a[N-1] = twosum(a[N-1], xh, s1);
    xh = s1;

    // Horizontal 2sum
    // Profitable??
    if(unlikely(horizontal_or((Vec16i(_mm512_castpd_si512(xh)) | Vec16i(_mm512_castpd_si512(xl))) != 0))) {
        //DumpVector(xh); printf(" / "); DumpVector(xl); printf("\n");
        //horizontal_twosum(xh, xl);
        //DumpVector(xh); printf(" / "); DumpVector(xl); printf("\n");
        FlushVector(xh);
        FlushVector(xl);
    }
}

template<typename T, int N, bool EX>
inline void FPExpansionVect<T,N,EX>::Accumulate(double const * p, int n)
{
    int const prefetch_distance_T0 = 1 * 16;
    int const prefetch_distance_T1 = 10 * 16;//7 * 16;
    if(EX) {
        asm volatile ("# ROI1");
    }
    else if(N == 2) {
        asm volatile ("# ROI2");
    }
    assert(!((size_t)p & 0x3f) && !((n+1) & 0xf)); // 128B-aligned
    limb_t la[N];
    
    // Restore
    //std::copy(a, a+N, la);
    std::fill(la, la+N, Vec8d(0));
    
    for(int i = 0; i < n; i+=16) {
        Vec8d x1 = Vec8d().load_a(p + i);
        Vec8d x2 = Vec8d().load_a(p + i + 8);
        T xl;
        T xh = twosum(x1, x2, xl);
        T s1,s2;
//#pragma monkey-unroll
        la[0] = twosum(la[0], xh, s1);
        xh = s1;
        la[1] = twosum(la[1], xl, s2);
        xl = s2;
        for(unsigned int j = 1; j < N - 1; ++j) {
            la[j] = twosum(la[j], xh, s1);
            xh = s1;
            if(EX && likely(!horizontal_or((Vec16i(_mm512_castpd_si512(xh)) | Vec16i(_mm512_castpd_si512(xl))) != 0))) goto continue_outer;
            la[j+1] = twosum(la[j+1], xl, s2);
            xl = s2;
        }
        la[N-1] = twosum(la[N-1], xh, s1);
        xh = s1;
        if(unlikely(horizontal_or((Vec16i(_mm512_castpd_si512(xh)) | Vec16i(_mm512_castpd_si512(xl))) != 0))) {
            FlushVector(xh);
            FlushVector(xl);
        }
continue_outer:
       ;
	    _mm_prefetch((char const *)(p+i+prefetch_distance_T1), _MM_HINT_T1);
	    _mm_prefetch((char const *)(p+i+prefetch_distance_T1+8), _MM_HINT_T1);
        _mm_prefetch((char const *)(p+i+prefetch_distance_T0), _MM_HINT_T0);
        _mm_prefetch((char const *)(p+i+prefetch_distance_T0+8), _MM_HINT_T0);
    }
    
    // Spill
    //std::copy(la, la+N, a);
    for(unsigned int j = 0; j < N; ++j) {
        FlushVector(la[j]);
    }
}

template<typename T, int N, bool EX>
void FPExpansionVect<T,N,EX>::Flush()
{
    for(unsigned int i = 0; i != N; ++i)
    {
        FlushVector(a[i]);
        a[i] = 0;
    }
}

template<typename T, int N, bool EX>
void FPExpansionVect<T,N,EX>::FlushVector(T x) const
{
    // TODO: horizontal 2sum
    // TODO: make it work for other values of 8
    double v[8] __attribute__((aligned(64)));
    Vec8b nonzero = x != 0;
    int n = x.store_compact_a(v, nonzero);
    
    for(unsigned int j = 0; j != n; ++j) {
        superacc.Accumulate(v[j]);
    }
}


template<typename T, int N, bool EX>
void FPExpansionVect<T,N,EX>::Dump() const
{
    for(unsigned int i = 0; i != N; ++i)
    {
        DumpVector(a[i]);
        std::cout << std::endl;
    }
}

template<typename T, int N, bool EX>
void FPExpansionVect<T,N,EX>::DumpVector(T x) const
{
    double v[8] __attribute__((aligned(64)));
    x.store_a(v);
    
    for(unsigned int j = 0; j != 8; ++j) {
        printf("%a ", v[j]);
    }
}

#endif
