// 512-bit vector class for Intel Xeon Phi
// Sylvain Collange <sylvain.collange@inria.fr>
// Based on Agner Fog's Vector Class
// (c) Copyright 2016 GNU General Public License http://www.gnu.org/licenses

/**
 *  \file vectorf512.h
 *  \brief Provides a set of auxiliary functions to work with Vec8d -- Vector of 8 double precision floating point values.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef VECTORF512_H
#define VECTORF512_H

#include "vectori512.h"



// TODO: Vec16f
// TODO: FMA

/*****************************************************************************
*
*          Vec8d: Vector of 8 double precision floating point values
*
*****************************************************************************/

class Vec8d {
protected:
    __m512d zmm; // double vector
public:
    // Default constructor:
    Vec8d() {
    }
    // Constructor to broadcast the same value into all elements:
    Vec8d(double d) {
        zmm = _mm512_set1_pd(d);
    }
    // Constructor to build from all elements:
    Vec8d(double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7) {
        zmm = _mm512_setr_pd(d0, d1, d2, d3, d4, d5, d6, d7);
    }
    // Constructor to convert from type __m512d used in intrinsics:
    Vec8d(__m512d const & x) {
        zmm = x;
    }
    // Assignment operator to convert from type __m512d used in intrinsics:
    Vec8d & operator = (__m512d const & x) {
        zmm = x;
        return *this;
    }
    // Type cast operator to convert to __m512d used in intrinsics
    operator __m512d() const {
        return zmm;
    }
    // Member function to load from array (unaligned)
    Vec8d & load(double const * p) {
        zmm = _mm512_loadunpacklo_pd(zmm, p);
        zmm = _mm512_loadunpackhi_pd(zmm, p+8);
        return *this;
    }
    // Member function to load from array, aligned by 8
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 64.
    Vec8d const & load_a(double const * p) {
        zmm = _mm512_load_pd(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(double * p) const {
        _mm512_packstorelo_pd(p, zmm);
        _mm512_packstorehi_pd(p+8, zmm);
    }
    // Member function to store into array, aligned by 8
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 64.
    void store_a(double * p) const {
        _mm512_store_pd(p, zmm);
    }
    // Partial load. Load n elements and set the rest to 0
    Vec8d & load_partial(int n, double const * p) {
        TODO
    }
    // Partial store. Store n elements
    void store_partial(int n, double * p) const {
        TODO
    }
    // Load and expand subset of elements selected by mask, aligned by 8
    Vec8d & load_expand_a(double const * p, Vec8b mask) {
        zmm = _mm512_mask_loadunpacklo_pd(zmm, mask, p);
        return *this;
    }
    // Load and expand subset of elements selected by mask
    Vec8d & load_expand(double const * p, Vec8b mask) {
        zmm = _mm512_mask_loadunpacklo_pd(zmm, mask, p);
        zmm = _mm512_mask_loadunpackhi_pd(zmm, mask, p+8);
        return *this;
    }
    // Compact and store subset of elements selected by mask. Returns number of elements written. Aligned version.
    int store_compact_a(double * p, Vec8b mask) const {
        _mm512_mask_packstorelo_pd(p, mask, zmm);
        return horizontal_add(mask);
    }
    // Compact and store subset of elements selected by mask. Returns number of elements written. Unaligned version.
    int store_compact(double * p, Vec8b mask) const {
        _mm512_mask_packstorelo_pd(p, mask, zmm);
        _mm512_mask_packstorehi_pd(p+8, mask, zmm);
        return horizontal_add(mask);
    }
    
    // Gather elements indexed by 8 low-order elements of Vec16i vector
    Vec8d & gather(double const * p, Vec16i indices) {
        zmm = _mm512_i32logather_pd(indices, p, 8);
        return *this;
    }
    
    void scatter(double * p, Vec16i indices) {
        _mm512_i32loscatter_pd(p, indices, zmm, 8);
    }

    // cut off vector to n elements. The last 16-n elements are set to zero
    Vec8d & cutoff(int n) {
        TODO
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec8d const & insert(uint32_t index, double value) {
        TODO    // Masked Shuffle
    };
    // Member function extract a single element from vector
    double extract(uint32_t index) const {  // SUBOPTIMAL
        double x[8];
        store(x);
        return x[index & 7];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    double operator [] (uint32_t index) const {
        return extract(index);
    }
};

/*****************************************************************************
*
*          Operators for Vec8d
*
*****************************************************************************/

// vector operator + : add element by element
static inline Vec8d operator + (Vec8d const & a, Vec8d const & b) {
    return _mm512_add_pd(a, b);
}

// vector operator + : add vector and scalar
static inline Vec8d operator + (Vec8d const & a, double b) {
    return a + Vec8d(b);
}
static inline Vec8d operator + (double a, Vec8d const & b) {
    return Vec8d(a) + b;
}

// vector operator += : add
static inline Vec8d & operator += (Vec8d & a, Vec8d const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec8d operator ++ (Vec8d & a, int) {
    Vec8d a0 = a;
    a = a + 1.0;
    return a0;
}

// prefix operator ++
static inline Vec8d & operator ++ (Vec8d & a) {
    a = a + 1.0;
    return a;
}

// vector operator - : subtract element by element
static inline Vec8d operator - (Vec8d const & a, Vec8d const & b) {
    return _mm512_sub_pd(a, b);
}

// vector operator - : subtract vector and scalar
static inline Vec8d operator - (Vec8d const & a, double b) {
    return a - Vec8d(b);
}
static inline Vec8d operator - (double a, Vec8d const & b) {
    return Vec8d(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec8d operator - (Vec8d const & a) {
    __m512i mask = _mm512_set1_epi64((1ull << 63));
    return _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(a), mask));
}

// vector operator -= : subtract
static inline Vec8d & operator -= (Vec8d & a, Vec8d const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec8d operator -- (Vec8d & a, int) {
    Vec8d a0 = a;
    a = a - 1.0;
    return a0;
}

// prefix operator --
static inline Vec8d & operator -- (Vec8d & a) {
    a = a - 1.0;
    return a;
}

// vector operator * : multiply element by element
static inline Vec8d operator * (Vec8d const & a, Vec8d const & b) {
    return _mm512_mul_pd(a, b);
}

// vector operator * : multiply vector and scalar
static inline Vec8d operator * (Vec8d const & a, double b) {
    return a * Vec8d(b);
}
static inline Vec8d operator * (double a, Vec8d const & b) {
    return Vec8d(a) * b;
}

// vector operator *= : multiply
static inline Vec8d & operator *= (Vec8d & a, Vec8d const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
static inline Vec8d operator / (Vec8d const & a, Vec8d const & b) {
    return _mm512_div_pd(a, b);
}

// vector operator / : divide vector and scalar
static inline Vec8d operator / (Vec8d const & a, double b) {
    return a / Vec8d(b);
}
static inline Vec8d operator / (double a, Vec8d const & b) {
    return Vec8d(a) / b;
}

// vector operator /= : divide
static inline Vec8d & operator /= (Vec8d & a, Vec8d const & b) {
    a = a / b;
    return a;
}


// vector operator == : returns true for elements for which a == b
static inline Vec8b operator == (Vec8d const & a, Vec8d const & b) {
    return _mm512_cmp_pd_mask(a, b, _MM_CMPINT_EQ);
}

// vector operator != : returns true for elements for which a != b
static inline Vec8b operator != (Vec8d const & a, Vec8d const & b) {
    return _mm512_cmp_pd_mask(a, b, _MM_CMPINT_NE);
}

// vector operator < : returns true for elements for which a < b
static inline Vec8b operator < (Vec8d const & a, Vec8d const & b) {
    return _mm512_cmp_pd_mask(a, b, _MM_CMPINT_LT);
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec8b operator <= (Vec8d const & a, Vec8d const & b) {
    return _mm512_cmp_pd_mask(a, b, _MM_CMPINT_LE);
}

// vector operator > : returns true for elements for which a > b
static inline Vec8b operator > (Vec8d const & a, Vec8d const & b) {
    return b < a;
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec8b operator >= (Vec8d const & a, Vec8d const & b) {
    return b <= a;
}

/*****************************************************************************
*
*          Functions for Vec8d
*
*****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
static inline Vec8d select (Vec8b const & s, Vec8d const & a, Vec8d const & b) {
    // That undocumented intrinsic...
    //return _mm512_mask_blend_pd(s, a, b);
    return _mm512_mask_mov_pd(b, s, a);
}

// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline double horizontal_add (Vec8d const & a) {
    return _mm512_reduce_add_pd(a); // Trusting Intel's implementation
}

// function max: a > b ? a : b
static inline Vec8d max(Vec8d const & a, Vec8d const & b) {
    return _mm512_max_pd(a,b);
}

// function min: a < b ? a : b
static inline Vec8d min(Vec8d const & a, Vec8d const & b) {
    return _mm512_min_pd(a,b);
}

// function abs: absolute value
// Removes sign bit, even for -0.0f, -INF and -NAN
static inline Vec8d abs(Vec8d const & a) {
    __m512i mask = _mm512_set1_epi64(~(1ull << 63));
    return _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(a), mask));
}

// function sqrt: square root
static inline Vec8d sqrt(Vec8d const & a) {
    return _mm512_sqrt_pd(a);
}

// function square: a * a
static inline Vec8d square(Vec8d const & a) {
    return a * a;
}

// pow(Vec8d, int):
// Raise floating point numbers to integer power n
static inline Vec8d pow(Vec8d const & a, int n) {
    Vec8d x = a;                       // a^(2^i)
    Vec8d y(1.0);                      // accumulator
    if (n >= 0) {                      // make sure n is not negative
        while (true) {                 // loop for each bit in n
            if (n & 1) y *= x;         // multiply if bit = 1
            n >>= 1;                   // get next bit of n
            if (n == 0) return y;      // finished
            x *= x;                    // x = a^2, a^4, a^8, etc.
        }
    }
    else {                             // n < 0
        return Vec8d(1.0)/pow(x,-n);   // reciprocal
    }
}

// Raise floating point numbers to integer power n, where n is a compile-time constant
template <int n>
static inline Vec8d pow_n(Vec8d const & a) {
    if (n < 0)    return Vec8d(1.0) / pow_n<-n>(a);
    if (n == 0)   return Vec8d(1.0);
    if (n >= 256) return pow(a, n);
    Vec8d x = a;                       // a^(2^i)
    Vec8d y;                           // accumulator
    const int lowest = n - (n & (n-1));// lowest set bit in n
    if (n & 1) y = x;
    if (n < 2) return y;
    x = x*x;                           // x^2
    if (n & 2) {
        if (lowest == 2) y = x; else y *= x;
    }
    if (n < 4) return y;
    x = x*x;                           // x^4
    if (n & 4) {
        if (lowest == 4) y = x; else y *= x;
    }
    if (n < 8) return y;
    x = x*x;                           // x^8
    if (n & 8) {
        if (lowest == 8) y = x; else y *= x;
    }
    if (n < 16) return y;
    x = x*x;                           // x^16
    if (n & 16) {
        if (lowest == 16) y = x; else y *= x;
    }
    if (n < 32) return y;
    x = x*x;                           // x^32
    if (n & 32) {
        if (lowest == 32) y = x; else y *= x;
    }
    if (n < 64) return y;
    x = x*x;                           // x^64
    if (n & 64) {
        if (lowest == 64) y = x; else y *= x;
    }
    if (n < 128) return y;
    x = x*x;                           // x^128
    if (n & 128) {
        if (lowest == 128) y = x; else y *= x;
    }
    return y;
}

template <int n>
static inline Vec8d pow(Vec8d const & a, Const_int_t<n>) {
    return pow_n<n>(a);
}

// function add_rn: add with rounding to nearest integer (even).
static inline Vec8d add_rn(Vec8d const & a, Vec8d const & b) {
    return _mm512_add_round_pd(a, b, _MM_FROUND_TO_NEAREST_INT);
}


// function round: round to nearest integer (even). (result as double vector)
static inline Vec8d round(Vec8d const & a) {
    return _mm512_nearbyint_pd(a);
}

// function truncate: round towards zero. (result as double vector)
static inline Vec8d truncate(Vec8d const & a) {
    return _mm512_trunc_pd(a);
}

// function floor: round towards minus infinity. (result as double vector)
static inline Vec8d floor(Vec8d const & a) {
    return _mm512_floor_pd(a);
}

// function ceil: round towards plus infinity. (result as double vector)
static inline Vec8d ceil(Vec8d const & a) {
    return _mm512_ceil_pd(a);
}

// function round_to_int: round to nearest integer (even). (result as integer vector, lower half)
static inline Vec16i round_to_int(Vec8d const & a) {
    // Intel doc typo?
    //return _mm512_cvt_roundpd_epi32lo(a, _MM_FROUND_TO_NEAREST_INT);
    return _mm512_cvtfxpnt_roundpd_epi32lo(a, _MM_FROUND_TO_NEAREST_INT);
}

// function truncate_to_int: round towards zero. (result as integer vector, lower half)
static inline Vec16i truncate_to_int(Vec8d const & a) {
    //return _mm512_cvt_roundpd_epi32lo(a, _MM_FROUND_TO_ZERO);
    return _mm512_cvtfxpnt_roundpd_epi32lo(a, _MM_FROUND_TO_ZERO);
}

// convert lower half of int32 vector to double.
static inline Vec8d to_double(Vec16i const & a) {
    return _mm512_cvtepi32lo_pd(a);
}

// function logb: returns exponent as FP number, like C99 logb
static inline Vec8d logb(Vec8d const & a) {
    return _mm512_getexp_pd(a);
}

// Warning: Does not support denormals, nor overflow!
static inline Vec8d scalbn(Vec8d const & a, int exp) {
    Vec16i ai = _mm512_castpd_si512(a);
    return _mm512_castsi512_pd(_mm512_mask_add_epi32(ai, _mm512_int2mask(0xaaaa), ai, Vec16i(exp << (52-32))));
}

// return a value whose magnitude is taken from x and whose sign is taken from y
// assumption: sign(x) = +
static inline Vec8d copysign_pos(Vec8d const & x, Vec8d const & y) {
    return _mm512_castsi512_pd(Vec512b(_mm512_castpd_si512(x))
        | (Vec512b(_mm512_castpd_si512(y)) & Vec512b(Vec8q(1ull << 63))));
}

#endif

