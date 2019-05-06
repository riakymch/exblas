// 512-bit vector class for Intel Xeon Phi
// Sylvain Collange <sylvain.collange@inria.fr>
// Based on Agner Fog's Vector Class
// (c) Copyright 2016 GNU General Public License http://www.gnu.org/licenses

/**
 *  \file vectori512.h
 *  \brief Auxiliary file to vectorf512.h.
 *         Provides a set of auxiliary functions to work with Vec8d -- Vector of 8 double precision floating point values.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef VECTORI512_H
#define VECTORI512_H

#include "instrset.h"  // Select supported instruction set
#include <cassert>

// TODO: mad



#define TODO assert(false);


class Vec8b {
protected:
    __mmask8 k; // mask
public:
    // Default constructor:
    Vec8b() {
    };
    // Constructor to broadcast the same value into all elements:
    explicit Vec8b(bool b) {
        k = _mm512_int2mask(b ? 0x00ff : 0);
    }
    // Constructor to build from integer mask:
    explicit Vec8b(int m) {
        k = _mm512_int2mask(m);
    }
    // Constructor to build from all elements:
    Vec8b(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7) {
        uint32_t mask = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7);
        k = _mm512_int2mask(mask);
    }
    // Constructor to convert from type __mmask8 used in intrinsics:
    Vec8b(__mmask8 const & x) {
        k = x;
    }
    // Assignment operator to convert from type __mmask8 used in intrinsics:
    Vec8b & operator = (__mmask8 const & x) {
        k = x;
        return *this;
    }
    // Type cast operator to convert to __mmask8 used in intrinsics
    operator __mmask8() const {
        return k;
    }
    // Member function to change a single element in vector
    Vec8b const & insert(uint32_t index, bool value) {
        uint32_t mask = value << index;
        k = _mm512_kor(k, _mm512_int2mask(mask));
        return *this;
    }
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        uint32_t mask = _mm512_mask2int(k);
        return (mask >> index) & 1;
    }
    // Extract a single element. Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
};


/*****************************************************************************
*
*          Operators for Vec8b
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec8b operator & (Vec8b const & a, Vec8b const & b) {
    return _mm512_kand(a, b);
}
static inline Vec8b operator && (Vec8b const & a, Vec8b const & b) {
    return a & b;
}

// vector operator &= : bitwise and
static inline Vec8b & operator &= (Vec8b & a, Vec8b const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec8b operator | (Vec8b const & a, Vec8b const & b) {
    return _mm512_kor(a, b);
}
static inline Vec8b operator || (Vec8b const & a, Vec8b const & b) {
    return a | b;
}

// vector operator |= : bitwise or
static inline Vec8b & operator |= (Vec8b & a, Vec8b const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec8b operator ^ (Vec8b const & a, Vec8b const & b) {
    return _mm512_kxor(a, b);
}

// vector operator ^= : bitwise xor
static inline Vec8b & operator ^= (Vec8b & a, Vec8b const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec8b operator ~ (Vec8b const & a) {
    return _mm512_knot(a);
}

// vector operator ! : logical not
// (same as bitwise not)
static inline Vec8b operator ! (Vec8b const & a) {
    return ~a;
}

// Functions for Vec8fb

// andnot: a & ~ b
static inline Vec8b andnot(Vec8b const & a, Vec8b const & b) {
    return _mm512_kandn(b, a);
}

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and (Vec8b const & a) {
    return _mm512_kortestc(a, Vec8b(0xff00));   // Force ignored bits to 1
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or (Vec8b const & a) {
    return !_mm512_kortestz(a, a);
}

// horizontal_add. Population count: count 1 bits
static inline int32_t horizontal_add (Vec8b const & a) {
    return _mm_countbits_32(a);
}



class Vec16b {
protected:
    __mmask16 k; // mask
public:
    // Default constructor:
    Vec16b() {
    };
    // Constructor to broadcast the same value into all elements:
    explicit Vec16b(bool b) {
        k = _mm512_int2mask(b ? 0xffff : 0);
    }
    // Constructor to build from integer mask:
    explicit Vec16b(int m) {
        k = _mm512_int2mask(m);
    }
    // Constructor to build from two Vec8b
    Vec16b(Vec8b lo, Vec8b hi) {
        k = _mm512_kmovlhb(lo, hi);
    }
    // Constructor to build from all elements:
    Vec16b(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7,
        bool b8, bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15) {
        uint32_t mask = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7)
            | (b8 << 8) | (b9 << 9) | (b10 << 10) | (b11 << 11) | (b12 << 12) | (b13 << 13) | (b14 << 14) | (b15 << 15);
        k = _mm512_int2mask(mask);
    }
    // Constructor to convert from type __mmask16 used in intrinsics:
    Vec16b(__mmask16 const & x) {
        k = x;
    }
    // Assignment operator to convert from type __mmask16 used in intrinsics:
    Vec16b & operator = (__mmask16 const & x) {
        k = x;
        return *this;
    }
    // Type cast operator to convert to __mmask8 used in intrinsics
    operator __mmask16() const {
        return k;
    }
    // Member function to change a single element in vector
    Vec16b const & insert(uint32_t index, bool value) {
        uint32_t mask = value << index;
        k = _mm512_kor(k, _mm512_int2mask(mask));
        return *this;
    }
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        uint32_t mask = _mm512_mask2int(k);
        return (mask >> index) & 1;
    }
    // Extract a single element. Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
};


/*****************************************************************************
*
*          Operators for Vec16b
*
*****************************************************************************/

// vector operator & : bitwise and
static inline Vec16b operator & (Vec16b const & a, Vec16b const & b) {
    return _mm512_kand(a, b);
}
static inline Vec16b operator && (Vec16b const & a, Vec16b const & b) {
    return a & b;
}

// vector operator &= : bitwise and
static inline Vec16b & operator &= (Vec16b & a, Vec16b const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec16b operator | (Vec16b const & a, Vec16b const & b) {
    return _mm512_kor(a, b);
}
static inline Vec16b operator || (Vec16b const & a, Vec16b const & b) {
    return a | b;
}

// vector operator |= : bitwise or
static inline Vec16b & operator |= (Vec16b & a, Vec16b const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec16b operator ^ (Vec16b const & a, Vec16b const & b) {
    return _mm512_kxor(a, b);
}

// vector operator ^= : bitwise xor
static inline Vec16b & operator ^= (Vec16b & a, Vec16b const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec16b operator ~ (Vec16b const & a) {
    return _mm512_knot(a);
}

// vector operator ! : logical not
// (same as bitwise not)
static inline Vec16b operator ! (Vec16b const & a) {
    return ~a;
}

// Functions for Vec8fb

// andnot: a & ~ b
static inline Vec16b andnot(Vec16b const & a, Vec16b const & b) {
    return _mm512_kandn(b, a);
}


/*****************************************************************************
*
*          Horizontal Boolean functions
*
*****************************************************************************/

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and (Vec16b const & a) {
    return _mm512_kortestc(a, a);
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or (Vec16b const & a) {
    return !_mm512_kortestz(a, a);
}

// horizontal_add. Population count: count 1 bits
static inline int32_t horizontal_add (Vec16b const & a) {
    return _mm_countbits_32(a);
}

#if 1

/*****************************************************************************
*
*          Vector of 512 1-bit unsigned integers or Booleans
*
*****************************************************************************/
// TODO: is this really needed?
class Vec512b {
protected:
    __m512i zmm; // Integer vector
public:
    // Default constructor:
    Vec512b() {
    };
    // Constructor to broadcast the same value into all elements:
    Vec512b(int i) {
        zmm = _mm512_set1_epi32(-(i & 1));
    };
    // Constructor to convert from type __m512i used in intrinsics:
    Vec512b(__m512i const & x) {
        zmm = x;
    };
    // Assignment operator to convert from type __m512i used in intrinsics:
    Vec512b & operator = (__m512i const & x) {
        zmm = x;
        return *this;
    };
    // Type cast operator to convert to __m512i used in intrinsics
    operator __m512i() const {
        return zmm;
    }
    // Member function to load from array (unaligned)
    Vec512b & load(void const * p) {
        zmm = _mm512_loadunpacklo_epi32(zmm, p);
        zmm = _mm512_loadunpackhi_epi32(zmm, p+16);
        return *this;
    }
    // Member function to load from array, aligned by 32
    // You may use load_a instead of load if you are certain that p points to an address
    // divisible by 32, but there is hardly any speed advantage of load_a on modern processors
    Vec512b & load_a(void const * p) {
        zmm = _mm512_load_epi32((__m512i const*)p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(void * p) const {
        _mm512_packstorelo_epi32(p, zmm);
        _mm512_packstorehi_epi32(p+16, zmm);
    }
    // Member function to store into array, aligned by 32
    // You may use store_a instead of store if you are certain that p points to an address
    // divisible by 32, but there is hardly any speed advantage of load_a on modern processors
    void store_a(void * p) const {
        _mm512_store_epi32((__m512i*)p, zmm);
    }
    // Member function to change a single bit
    // Note: This function is inefficient. Use load function if changing more than one bit
    Vec512b const & set_bit(uint32_t index, int value) {
        TODO
        return *this;
    }
    // Member function to get a single bit
    // Note: This function is inefficient. Use store function if reading more than one bit
    int get_bit(uint32_t index) const {
        union {
            __m512i x;
            uint8_t i[64];
        } u;
        u.x = zmm; 
        int wi = (index >> 3) & 0x1F;            // byte index
        int bi = index & 7;                      // bit index within byte w
        return (u.i[wi] >> bi) & 1;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return get_bit(index);
    }
};


// Define operators for this class

// vector operator & : bitwise and
static inline Vec512b operator & (Vec512b const & a, Vec512b const & b) {
    return _mm512_and_epi32(a, b);
}
static inline Vec512b operator && (Vec512b const & a, Vec512b const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec512b operator | (Vec512b const & a, Vec512b const & b) {
    return _mm512_or_epi32(a, b);
}
static inline Vec512b operator || (Vec512b const & a, Vec512b const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec512b operator ^ (Vec512b const & a, Vec512b const & b) {
    return _mm512_xor_epi32(a, b);
}

// vector operator ~ : bitwise not
static inline Vec512b operator ~ (Vec512b const & a) {
    return _mm512_xor_epi32(a, _mm512_set1_epi32(-1));
}

// vector operator &= : bitwise and
static inline Vec512b & operator &= (Vec512b & a, Vec512b const & b) {
    a = a & b;
    return a;
}

// vector operator |= : bitwise or
static inline Vec512b & operator |= (Vec512b & a, Vec512b const & b) {
    a = a | b;
    return a;
}

// vector operator ^= : bitwise xor
static inline Vec512b & operator ^= (Vec512b & a, Vec512b const & b) {
    a = a ^ b;
    return a;
}

// Define functions for this class

// function andnot: a & ~ b
static inline Vec512b andnot (Vec512b const & a, Vec512b const & b) {
    return _mm512_andnot_epi32(b, a);
}
#endif

/*****************************************************************************
*
*          Generate compile-time constant vector
*
*****************************************************************************/
// Generate a constant vector of 16 integers stored in memory.
// Can be converted to any integer vector type
template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7,
          int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
static inline __m512i constant16i() {
    static const union {
        int32_t i[16];
        __m512i zmm;
    } u = {{i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15}};
    return u.zmm;
}


#if 1
/*****************************************************************************
*
*          Horizontal Boolean functions
*
*****************************************************************************/

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and (Vec512b const & a) {
    Vec16b mask = _mm512_cmpeq_epi32_mask(a,(constant16i<-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1>()));
    return horizontal_and(mask);
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or (Vec512b const & a) {
    Vec16b mask = _mm512_test_epi32_mask(a,a);
    //return _mm512_kortestz(mask, mask);
    return horizontal_or(mask);
}
#endif


/*****************************************************************************
*
*          Vector of 16 32-bit signed integers
*
*****************************************************************************/

class Vec16i : public Vec512b {
protected:
    __m512i zmm; // Integer vector
public:
    // Default constructor:
    Vec16i() {
    };
    // Constructor to broadcast the same value into all elements:
    Vec16i(int i) {
        zmm = _mm512_set1_epi32(i);
    };
    // Constructor to build from all elements:
    Vec16i(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4, int32_t i5, int32_t i6, int32_t i7,
        int32_t i8, int32_t i9, int32_t i10, int32_t i11, int32_t i12, int32_t i13, int32_t i14, int32_t i15) {
        zmm = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7,
            i8, i9, i10, i11, i12, i13, i14, i15);
    };

    // Constructor to convert from type __m512i used in intrinsics:
    Vec16i(__m512i const & x) {
        zmm = x;
    };
    // Assignment operator to convert from type __m512i used in intrinsics:
    Vec16i & operator = (__m512i const & x) {
        zmm = x;
        return *this;
    };
    // Type cast operator to convert to __m512i used in intrinsics
    operator __m512i() const {
        return zmm;
    };
    // Member function to load from array (unaligned)
    Vec16i & load(int32_t const * p) {
        zmm = _mm512_loadunpacklo_epi32(zmm, p);
        zmm = _mm512_loadunpackhi_epi32(zmm, p+16);
        return *this;
    }
    // Member function to load from array, aligned by 32
    Vec16i & load_a(int32_t const * p) {
        zmm = _mm512_load_epi32(p);
        return *this;
    }
    // Partial load. Load n elements and set the rest to 0
    Vec16i & load_partial(int n, int32_t const * p) {
        load_expand(p, Vec16b((1<<n)-1));
        TODO // set the rest to 0
        return *this;
    }
    // Load and expand subset of elements selected by mask
    Vec16i & load_expand(int32_t const * p, Vec16b mask) {
        zmm = _mm512_mask_loadunpacklo_epi32(zmm, mask, p);
        zmm = _mm512_mask_loadunpackhi_epi32(zmm, mask, p+16);
        return *this;
    }
    
    // Member function to store into array (unaligned)
    void store(int32_t * p) const {
        _mm512_packstorelo_epi32(p, zmm);
        _mm512_packstorehi_epi32(p+16, zmm);
    }
    // Member function to store into array, aligned by 32
    void store_a(int32_t * p) const {
        _mm512_store_epi32(p, zmm);
    }
    // Partial store. Store n elements
    void store_partial(int n, int32_t * p) const {
        store_compact(p, Vec16b((1<<n)-1));
    }
    // Compact and store subset of elements selected by mask. Returns number of elements written. Aligned version.
    int store_compact_a(int32_t * p, Vec16b mask) const {
        _mm512_mask_packstorelo_epi32(p, mask, zmm);
        return horizontal_add(mask);
    }
    // Compact and store subset of elements selected by mask. Returns number of elements written. Unaligned version.
    int store_compact(int32_t * p, Vec16b mask) const {
        _mm512_mask_packstorelo_epi32(p, mask, zmm);
        _mm512_mask_packstorehi_epi32(p+16, mask, zmm);
        return horizontal_add(mask);
    }

    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec16i & cutoff(int n) {
        TODO
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec16i const & insert(uint32_t index, int32_t value) {
        TODO
        return *this;
    };
    // Member function extract a single element from vector
    int32_t extract(uint32_t index) const {
        int32_t x[16];
        store(x);
        return x[index & 15];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int32_t operator [] (uint32_t index) const {
        return extract(index);
    }
};

// Define operators for this class

// vector operator + : add element by element
static inline Vec16i operator + (Vec16i const & a, Vec16i const & b) {
    return _mm512_add_epi32(a, b);
}

// vector operator += : add
static inline Vec16i & operator += (Vec16i & a, Vec16i const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec16i operator ++ (Vec16i & a, int) {
    Vec16i a0 = a;
    a = a + 1;
    return a0;
}

// prefix operator ++
static inline Vec16i & operator ++ (Vec16i & a) {
    a = a + 1;
    return a;
}

// vector operator - : subtract element by element
static inline Vec16i operator - (Vec16i const & a, Vec16i const & b) {
    return _mm512_sub_epi32(a, b);
}

// vector operator - : unary minus
static inline Vec16i operator - (Vec16i const & a) {
    return _mm512_sub_epi32(_mm512_setzero_epi32(), a);
}

// vector operator -= : subtract
static inline Vec16i & operator -= (Vec16i & a, Vec16i const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec16i operator -- (Vec16i & a, int) {
    Vec16i a0 = a;
    a = a - 1;
    return a0;
}

// prefix operator --
static inline Vec16i & operator -- (Vec16i & a) {
    a = a - 1;
    return a;
}

// vector operator * : multiply element by element
static inline Vec16i operator * (Vec16i const & a, Vec16i const & b) {
    return _mm512_mullo_epi32(a, b);
}

// vector operator *= : multiply
static inline Vec16i & operator *= (Vec16i & a, Vec16i const & b) {
    a = a * b;
    return a;
}

// vector operator / : divide all elements by same integer
// See bottom of file


// vector operator << : shift left
static inline Vec16i operator << (Vec16i const & a, int32_t b) {
    return _mm512_slli_epi32(a, b);
}

// vector operator <<= : shift left
static inline Vec16i & operator <<= (Vec16i & a, int32_t b) {
    a = a << b;
    return a;
}

// vector operator >> : shift right arithmetic
static inline Vec16i operator >> (Vec16i const & a, int32_t b) {
    return _mm512_srai_epi32(a, b);
}

// vector operator >>= : shift right arithmetic
static inline Vec16i & operator >>= (Vec16i & a, int32_t b) {
    a = a >> b;
    return a;
}

// vector operator << : shift left variable
static inline Vec16i operator << (Vec16i const & a, Vec16i const & b) {
    return _mm512_sllv_epi32(a, b);
}

// vector operator <<= : shift left variable
static inline Vec16i & operator <<= (Vec16i & a, Vec16i const & b) {
    a = a << b;
    return a;
}

// vector operator >> : shift right arithmetic variable
static inline Vec16i operator >> (Vec16i const & a, Vec16i const & b) {
    return _mm512_srav_epi32(a, b);
}

// vector operator >>= : shift right arithmetic variable
static inline Vec16i & operator >>= (Vec16i & a, Vec16i const & b) {
    a = a >> b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec16b operator == (Vec16i const & a, Vec16i const & b) {
    return _mm512_cmpeq_epi32_mask(a, b);
}

// vector operator != : returns true for elements for which a != b
static inline Vec16b operator != (Vec16i const & a, Vec16i const & b) {
    return _mm512_cmpneq_epi32_mask(a,b);
}
  
// vector operator > : returns true for elements for which a > b
static inline Vec16b operator > (Vec16i const & a, Vec16i const & b) {
    return _mm512_cmpgt_epi32_mask(a, b);
}

// vector operator < : returns true for elements for which a < b
static inline Vec16b operator < (Vec16i const & a, Vec16i const & b) {
    return b > a;
}

// vector operator >= : returns true for elements for which a >= b (signed)
static inline Vec16b operator >= (Vec16i const & a, Vec16i const & b) {
    return _mm512_cmpge_epi32_mask(a,b);
}

// vector operator <= : returns true for elements for which a <= b (signed)
static inline Vec16b operator <= (Vec16i const & a, Vec16i const & b) {
    return b >= a;
}

// vector operator & : bitwise and
static inline Vec16i operator & (Vec16i const & a, Vec16i const & b) {
    return _mm512_and_epi32(a, b);
}

// vector operator | : bitwise or
static inline Vec16i operator | (Vec16i const & a, Vec16i const & b) {
    return _mm512_or_epi32(a, b);
}

// vector operator ^ : bitwise xor
static inline Vec16i operator ^ (Vec16i const & a, Vec16i const & b) {
    return _mm512_xor_epi32(a, b);
}

// vector operator ~ : bitwise not
static inline Vec16i operator ~ (Vec16i const & a) {
    return _mm512_xor_epi32(a, Vec16i(-1));
}

// vector operator ! : returns true for elements == 0
static inline Vec16b operator ! (Vec16i const & a) {
    return _mm512_cmpeq_epi32_mask(a, _mm512_setzero_epi32());
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
static inline Vec16i select (Vec16b const & s, Vec16i const & a, Vec16i const & b) {
    //return _mm512_mask_blend_epi32(s,a,b);
    return _mm512_mask_mov_epi32(b, s, a);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
static inline int32_t horizontal_add (Vec16i const & a) {
    return _mm512_reduce_add_epi32(a);
}

// function max: a > b ? a : b
static inline Vec16i max(Vec16i const & a, Vec16i const & b) {
    return _mm512_max_epi32(a,b);
}

// function min: a < b ? a : b
static inline Vec16i min(Vec16i const & a, Vec16i const & b) {
    return _mm512_min_epi32(a,b);
}

// function abs: a >= 0 ? a : -a
static inline Vec16i abs(Vec16i const & a) {
    return _mm512_max_epi32(a,-a);
}

/*****************************************************************************
*
*          Vector of 8 64-bit signed integers
*
*****************************************************************************/


class Vec8q : public Vec512b {
protected:
    __m512i zmm; // Integer vector
public:
    // Default constructor:
    Vec8q() {
    };
    // Constructor to broadcast the same value into all elements:
    Vec8q(int64_t i) {
        zmm = _mm512_set1_epi64(i);
    };
    // Constructor to build from all elements:
    Vec8q(int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5, int64_t i6, int64_t i7) {
        zmm = _mm512_setr_epi64(i0, i1, i2, i3, i4, i5, i6, i7);
    };

    // Constructor to convert from type __m512i used in intrinsics:
    Vec8q(__m512i const & x) {
        zmm = x;
    };
    // Assignment operator to convert from type __m512i used in intrinsics:
    Vec8q & operator = (__m512i const & x) {
        zmm = x;
        return *this;
    };
    // Type cast operator to convert to __m512i used in intrinsics
    operator __m512i() const {
        return zmm;
    };
    // Member function to load from array (unaligned)
    Vec8q & load(int64_t const * p) {
        zmm = _mm512_loadunpacklo_epi64(zmm, p);
        zmm = _mm512_loadunpackhi_epi64(zmm, p+8);
        return *this;
    }
    // Member function to load from array, aligned by 32
    Vec8q & load_a(int64_t const * p) {
        zmm = _mm512_load_epi64(p);
        return *this;
    }
    // Member function to store into array (unaligned)
    void store(int64_t * p) const {
        _mm512_packstorelo_epi64(p, zmm);
        _mm512_packstorehi_epi64(p+8, zmm);
    }
    // Member function to store into array, aligned by 32
    void store_a(int64_t * p) const {
        _mm512_store_epi64(p, zmm);
    }
    // Partial store. Store n elements
    void store_partial(int n, int64_t * p) const {
        store_compact(p, Vec8b((1<<n)-1));
    }
    // Compact and store subset of elements selected by mask. Returns number of elements written. Aligned version.
    int store_compact_a(int64_t * p, Vec8b mask) const {
        _mm512_mask_packstorelo_epi64(p, mask, zmm);
        return horizontal_add(mask);
    }
    // Compact and store subset of elements selected by mask. Returns number of elements written. Unaligned version.
    int store_compact(int64_t * p, Vec8b mask) const {
        _mm512_mask_packstorelo_epi32(p, mask, zmm);
        _mm512_mask_packstorehi_epi32(p+8, mask, zmm);
        return horizontal_add(mask);
    }

    // cut off vector to n elements. The last 8-n elements are set to zero
    Vec8q & cutoff(int n) {
        TODO
        return *this;
    }
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec8q const & insert(uint32_t index, int32_t value) {
        TODO
        return *this;
    };
    // Member function extract a single element from vector
    int64_t extract(uint32_t index) const {
        int64_t x[8];
        store(x);
        return x[index & 7];
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int64_t operator [] (uint32_t index) const {
        return extract(index);
    }
};


#endif
