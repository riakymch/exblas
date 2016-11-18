/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */
#include "superaccumulator.hpp"
#include <cassert>
#include "mylibm.hpp"
#include <ostream>
#include <cmath>

#include <iostream>

Superaccumulator::Superaccumulator(int e_bits, int f_bits) :
    f_words((f_bits + digits - 1) / digits),   // Round up
    e_words((e_bits + digits - 1) / digits),
    accumulator(f_words + e_words, 0),
    imin(f_words + e_words - 1), imax(0),
    status(Exact),
    overflow_counter((1ll<<K)-1)
{
}

void Superaccumulator::Accumulate(int64_t x, int exp)
{
    Normalize();
    // Count from lsb to avoid signed arithmetic
    unsigned int exp_abs = exp + f_words * digits;
    int i = exp_abs / digits;
    int shift = exp_abs % digits;

    imin = std::min(imin, i);
    imax = std::max(imax, i+2);

    if(shift == 0) {
        // ignore carry
        AccumulateWord(x, i);
        return;
    }
    //        xh      xm    xl
    //        |-   ------   -|shift
    // |XX-----+|XX++++++|XX+-----|
    //   a[i+1]    a[i]
    //printf("x=%lx, i=%d\n", x, i);

    int64_t xl = (x << shift) & ((1ll << digits) - 1);
    AccumulateWord(xl, i);
    x >>= digits - shift;
    if(x == 0) return;
    int64_t xm = x & ((1ll << digits) - 1);
    AccumulateWord(xm, i + 1);
    x >>= digits;
    if(x == 0) return;
    int64_t xh = x & ((1ll << digits) - 1);
    AccumulateWord(xh, i + 2);
}

#if 0 // Inlined

#if 0
// Slow standard C version
void Superaccumulator::AccumulateWord(int64_t x, int exp_word)
{
    assert(exp_word < e_words && exp_word >= -f_words);
    int i = exp_word + f_words;
    
    int64_t oldword = accumulator[i];   // Sign S
    int64_t newword = oldword + x;
    
    while(unlikely((newword < 0 && oldword > 0 && x > 0)
        || (newword > 0 && oldword < 0 && x < 0))) {

        // Carry or borrow: sign !S
        int64_t carry = newword >> digits;    // Arithmetic shift
        
        // Cancel carry-save bits: sign S
        accumulator[i] = newword - (carry << digits);
        
        // Add carry bit: sign S
        carry += (newword < 0 ? 1ll << K : -1ll << K);

        ++i;
        if(i >= f_words + e_words) {
            status = Overflow;
            return;
        }
        
        oldword = accumulator[i];
        newword = oldword + carry;
    }
    
    accumulator[i] = newword;
}
#elif 0
// gcc version using asm goto
#define XADD(memref, oldword) \
    asm volatile (LOCK_PREFIX"xadd %0, %1" : "+m" (accumulator[i]), "+r" (oldword) : : "cc")

void Superaccumulator::AccumulateWord(int64_t x, int exp_word)
{
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    //assert(exp_word < e_words && exp_word >= -f_words);
    if(!(exp_word < e_words && exp_word >= -f_words)) {
        printf("exp_word=%d\n", exp_word);
        assert(false);
    }
    
    int i = exp_word + f_words;
    int64_t carry = x;
    int64_t carrybit;
    int64_t oldword = x;
    XADD(accumulator[i], oldword);
    asm goto ("jo %l[overflow]" : : "X"(oldword) /*artificial*/ : : overflow);
    return;

overflow:
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // accumulator[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> digits;    // Arithmetic shift
        carrybit = (oldword > 0 ? 1ll << K : -1ll << K);
        
        // Cancel carry-save bits
        //accumulator[i] -= (carry << digits);
        unsigned char of;
        xadd(accumulator[i], -(carry << digits), of);
        
        carry += carrybit;

        ++i;
        if(i >= f_words + e_words) {
            status = Overflow;
            return;
        }
        oldword = carry;
        //asm volatile (LOCK_PREFIX"xadd %0, %1" : "+m" (accumulator[i]), "+r" (oldword) : : "cc");
        XADD(accumulator[i], oldword);
        asm goto ("jo %l[overflow]" : : "X"(oldword) /*artificial*/ : : overflow);
}
#else
// "portable" x64 version

void Superaccumulator::AccumulateWord(int64_t x, int exp_word)
{
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    //assert(exp_word < e_words && exp_word >= -f_words);
    if(!(exp_word < e_words && exp_word >= -f_words)) {
        printf("exp_word=%d\n", exp_word);
        assert(false);
    }
    int i = exp_word + f_words;
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


#endif


#if 0
void Superaccumulator::Accumulate(double x)
{
    if(x == 0) return;
    
    // TODO: specials (+inf, -inf, NaN)
    // x87 to the rescue!
    int e = exponent(x);                            // TODO: use fxtract (logb)
    int exp_word = e / digits;  // Word containing MSbit    // TODO: use fmul by constant 80-bit, or even fdiv? + frndint?
    
    //double inputScale = exp2i(-digits * exp_word);
    
    //double xscaled = x * inputScale;
    double xscaled = myldexp(x, -digits * exp_word);    // TODO: use fscale

    for(unsigned int i = exp_word; xscaled != 0; --i) {
        //assert(i + f_words >= 0);

        //double xrounded = myrint(xscaled);    // TODO: use frndint
        //int64_t xint = myllrint(xscaled);     // TODO: use fistp
        int64_t xint = (int64_t)xscaled;
        double xrounded = (double)xint;
        AccumulateWord(xint, i);
        
        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
   
}
#elif 1
// Using the bleeding-edge x87 ISA
void Superaccumulator::Accumulate(double xx)
{
    if(xx == 0) return;
    
    long double x = xx;
    
    // TODO: specials (+inf, -inf, NaN)
    // x87 to the rescue!
    long double e = logbl(x);
    long double exp_word = floorl(e / digits);  // Word containing MSbit    // TODO: use fmul by constant 80-bit
    
    //double inputScale = exp2i(-digits * exp_word);
    
    //double xscaled = x * inputScale;
    long double xscaled = scalbl(x, -digits * exp_word);

    for(unsigned int i = lrintl(exp_word); xscaled != 0; --i) {
        //assert(i + f_words >= 0);

        long double xrounded = rintl(xscaled);
        int64_t xint = lrintl(xscaled);
        //int64_t xint = (int64_t)xscaled;
        //double xrounded = (double)xint;
        AccumulateWord(xint, i);
        
        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
   
}

#else
void Superaccumulator::Accumulate(double x)
{
    if(x == 0) return;
    
    // TODO: specials (+inf, -inf, NaN)
    // x87 to the rescue!
    double e = logb(x);
    double exp_word = floor(e / digits);  // Word containing MSbit    // TODO: use fmul by constant 80-bit
    
    //double inputScale = exp2i(-digits * exp_word);
    
    //double xscaled = x * inputScale;
    double xscaled = scalb(x, -digits * exp_word);

    for(unsigned int i = exp_word; xscaled != 0; --i) {
        //assert(i + f_words >= 0);

        double xrounded = rint(xscaled);
        int64_t xint = lrint(xscaled);
        //int64_t xint = (int64_t)xscaled;
        //double xrounded = (double)xint;
        AccumulateWord(xint, i);
        
        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
   
}

#endif

#endif

void Superaccumulator::Accumulate(Superaccumulator & other)
{
    // Naive impl
    // TODO: keep track of bounds for sparse accumulator sum
    // TODO: update status

#if 0
    // Works ok
    // Keep track of reduction step counter to allow K normalization-free reduction steps
    int sum_overflow_counter = overflow_counter + other.overflow_counter + 1;
    if(sum_overflow_counter >= (1ll << K)) {
        // Sum would overflow.
        // Need to normalize either or both accumulators first
        if(overflow_counter >= (1ll << (K-1)) || overflow_counter >= other.overflow_counter) {
            Normalize();
        }
        if(other.overflow_counter >= (1ll << (K-1)) || other.overflow_counter > overflow_counter) {
            other.Normalize();
        }
        sum_overflow_counter = overflow_counter + other.overflow_counter + 1;
    }
    overflow_counter = sum_overflow_counter;
#else
    Normalize();
    other.Normalize();
#endif

    imin = std::min(imin, other.imin);
    imax = std::max(imax, other.imax);
    asm("# myreduction");
    // TODO: ensure Accumulate(double/int) updates ovf cntr

    // TODO: vectorize
    for(int i = imin; i <= imax; ++i) {
        accumulator[i] += other.accumulator[i];
    }
}

double Superaccumulator::Round()
{
    assert(digits >= 52);
#if TRACK_BOUNDS
    assert(imin >= 0 && imax < f_words + e_words);
    if(imin > imax) {
        return 0;
    }
#else
    imin = 0;
    imax = e_words + f_words - 1;
#endif
    bool negative = Normalize();
    
    // Find leading word
    int i;
    for(i = imax;
        accumulator[i] == 0 && i >= imin;
        --i) {
    }
    if(negative) {
        // Skip ones
        for(;
            (accumulator[i] & ((1ll << digits) - 1)) == ((1ll << digits) - 1) && i >= imin;
            --i) {
        }
    }
    if(i < 0) {
        // TODO: should we preserve sign of zero?
        return 0.;
    }
    
    int64_t hiword = negative ? ((1ll << digits) - 1) - accumulator[i] : accumulator[i];
    double rounded = double(hiword);
    double hi = ldexp(rounded, (i - f_words) * digits);
    if(i == 0) {
        return negative ? -hi : hi;  // Correct rounding achieved
    }
    hiword -= llrint(rounded);
    double mid = ldexp(double(hiword), (i - f_words) * digits);
    
    // Compute sticky
    int64_t sticky = 0;
    for(int j = imin; j != i - 1; ++j) {
        sticky |= negative ? (1ll << digits) - accumulator[j] : accumulator[j];
    }
    
    int64_t loword = negative ? (1ll << digits) - accumulator[i-1] : accumulator[i-1];
    loword |=!! sticky;
    double lo = ldexp(double(loword), (i - 1 - f_words) * digits);
 
    // Now add3(hi, mid, lo)
    // No overlap, we have already normalized
    if(mid != 0) {
        lo = OddRoundSumNonnegative(mid, lo);
    }
    // Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}

bool Superaccumulator::Normalize()
{
#if TRACK_BOUNDS
    if(!(imin >= 0 && imax < f_words + e_words)) {
        printf("imin=%d, imax=%d\n", imin, imax);
        assert(false);
    }
    if(imin > imax) {
        return false;
    }
#else
    imin = 0;
    imax = f_words + e_words - 1;
#endif
    overflow_counter = 0;
    int64_t carry_in = accumulator[imin] >> digits;
    accumulator[imin] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for(i = imin + 1;
        i < f_words + e_words;
        ++i)
    {
        accumulator[i] += carry_in;
        int64_t carry_out = accumulator[i] >> digits;    // Arithmetic shift
        accumulator[i] -= (carry_out << digits);
        carry_in = carry_out;
    }
    imax = i - 1;
    // Do not cancel the last carry to avoid losing information
    accumulator[imax] += carry_in << digits;
    
    /*if(carry_in != 0 && carry_in != -1) {
        status = Overflow;
    }*/
    return carry_in < 0;
}

void Superaccumulator::Dump(std::ostream & os)
{
    switch(status) {
    case Exact:
        os << "Exact "; break;
    case Inexact:
        os << "Inexact "; break;
    case Overflow:
        os << "Overflow "; break;
    default:
        os << "??";
    }
    os << std::hex;
    for(int i = f_words + e_words - 1; i >= 0; --i) {
        int64_t hi = accumulator[i] >> digits;
        int64_t lo = accumulator[i] - (hi << digits);
        os << "+" << hi << " " << lo;
    }
    os << std::dec;
    os << std::endl;
}

