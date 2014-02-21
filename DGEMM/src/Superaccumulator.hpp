#ifndef SUPERACCUMULATOR_HPP_INCLUDED
#define SUPERACCUMULATOR_HPP_INCLUDED

#include <vector>
#include <stdint.h>
#include <iosfwd>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "mylibm.hpp"

#include <gmp.h>
#include <mpfr.h>

// TODO: generalize for other int64_t, such as int32_t...
//       do we need to? Superaccumulator is scalar.
struct Superaccumulator
{
    Superaccumulator(int e_bits = 1023, int f_bits = 1023 + 52);
    Superaccumulator(int64_t *acc, int e_bits = 1023, int f_bits = 1023 + 52);
    
    void Accumulate(int64_t x, int exp);
    void Accumulate(double x);
    void Accumulate(Superaccumulator & other);   // May modify (normalize) other member
    double Round();
    bool CompareSuperaccumulatorWithMPFR(char *resAcc);
    bool CompareSuperaccumulatorWithMPFROld(char *resAcc);
    void PrintAccumulator();

    // TODO: make it a bitfield to simplify (and make atomic) update logic?
    // or use a canari.
    enum Status
    {
        Exact,
        Inexact,
        MinusInfinity,
        PlusInfinity,
        Overflow,
        sNaN,
        qNaN
    };
    bool Normalize();
    void Dump(std::ostream & os);

private:
    void AccumulateWord(int64_t x, int i);

    unsigned int K;    // High-radix carry-save bits
    int digits;
    double deltaScale; // Assumes K>0

    int f_words, e_words;
    std::vector<int64_t> accumulator;   // Mutable: Normalization does (should) not change externally-visible state
    int imin, imax;
    Status status;
    
    int64_t overflow_counter;
};

#endif
