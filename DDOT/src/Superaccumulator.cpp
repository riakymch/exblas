#include <ostream>
#include <cassert>
#include <cmath>
#include <string.h>

#include "mylibm.hpp"
#include "Superaccumulator.hpp"

Superaccumulator::Superaccumulator(int e_bits, int f_bits) :
      K(8), digits(64 - K), deltaScale(double(1ull << digits)),
      f_words((f_bits + digits - 1) / digits),   // Round up
      e_words((e_bits + digits - 1) / digits),
      accumulator(f_words + e_words, 0), imin(f_words + e_words - 1), imax(0),
      status(Exact), overflow_counter((1ll << K) - 1) {
}

Superaccumulator::Superaccumulator(int64_t *acc, int e_bits, int f_bits) :
      K(8), digits(64 - K), deltaScale(double(1ull << digits)),
      f_words((f_bits + digits - 1) / digits),   // Round up
      e_words((e_bits + digits - 1) / digits),
      imin(f_words + e_words - 1), imax(0),
      status(Exact), overflow_counter((1ll << K) - 1) {
  for (int i = 0; i < f_words + e_words; ++i)
    accumulator.push_back(acc[i]);
}

void Superaccumulator::AccumulateWord(int64_t x, int i) {
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    //assert(i >= 0 && i < e_words + f_words);
    //int i = exp_word + f_words;
    int64_t carry = x;
    int64_t carrybit;
    unsigned char overflow;
    int64_t oldword = xadd(accumulator[i], x, overflow);
    //while(unlikely(overflow))
    while(overflow) {
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
        xadd(accumulator[i], (int64_t) -(carry << digits), overflow);
        //assert(TSAFE || !(s ^ overflow));
        //if(TSAFE && unlikely(s ^ overflow)) {
        if(TSAFE && (s ^ overflow)) {
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

void Superaccumulator::Accumulate(double x) {
    if(x == 0) return;
    
    // TODO: get exponent right for subnormals
    // TODO: specials (+inf, -inf, NaN)
    
    int e = exponent(x);
    int exp_word = e / digits;  // Word containing MSbit
    int iup = exp_word + f_words;
    
    //double inputScale = exp2i(-digits * exp_word);
    //double xscaled = x * inputScale;
    double xscaled = myldexp(x, -digits * exp_word);

    //int idown = std::max(int(iup) - 2 - 52 / (digits+1), 0);
    //imax = std::max(imax, iup);
    if(iup > imax) imax = iup;

    int i;
    for(i = iup; xscaled != 0; --i) {
    //for(int i = imax; i >= imin; --i) {
        //assert(i >= 0);

        double xrounded = myrint(xscaled);
        int64_t xint = myllrint(xscaled);
        AccumulateWord(xint, i);
        
        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
    //imin = std::min(imin, i + 1);
    if(i + 1 < imin) imin = i + 1;
}

void Superaccumulator::Accumulate(int64_t x, int exp) {
  // Count from lsb to avoid signed arithmetic
  unsigned int exp_abs = exp + f_words * digits;
  int i = exp_abs / digits;
  int shift = exp_abs % digits;

  imin = std::min(imin, i);
  imax = std::max(imax, i + 2);

  if (shift == 0) {
    // ignore carry
    AccumulateWord(x, i);
    return;
  }
  //        xh      xm    xl
  //        |-   ------   -|shift
  // |XX-----+|XX++++++|XX+-----|
  //   a[i+1]    a[i]

  int64_t xl = (x << shift) & ((1ll << digits) - 1);
  AccumulateWord(xl, i);
  x >>= digits - shift;
  if (x == 0)
    return;
  int64_t xm = x & ((1ll << digits) - 1);
  AccumulateWord(xm, i + 1);
  x >>= digits;
  if (x == 0)
    return;
  int64_t xh = x & ((1ll << digits) - 1);
  AccumulateWord(xh, i + 2);
}

void Superaccumulator::Accumulate(Superaccumulator & other) {
  // Naive impl
  // TODO: keep track of bounds for sparse accumulator sum
  // TODO: update status
  //printf("imin=%d, imax=%d, o.imin=%d, o.imax=%d\n", imin, imax, other.imin, other.imax);

  // Keep track of reduction step counter to allow K normalization-free reduction steps
  int sum_overflow_counter = overflow_counter + other.overflow_counter + 1;
  if (sum_overflow_counter >= (1ll << K)) {
    // Sum would overflow.
    // Need to normalize either or both accumulators first
    if (overflow_counter >= (1ll << (K - 1))
      || overflow_counter >= other.overflow_counter) {
      Normalize();
    }
    if (other.overflow_counter >= (1ll << (K - 1))
      || other.overflow_counter > overflow_counter) {
      other.Normalize();
    }
    sum_overflow_counter = overflow_counter + other.overflow_counter + 1;
  }
  overflow_counter = sum_overflow_counter;

  imin = std::min(imin, other.imin);
  imax = std::max(imax, other.imax);
  // TODO: ensure Accumulate(double/int) updates ovf cntr
  for (int i = imin; i <= imax; ++i) {
    accumulator[i] += other.accumulator[i];
  }

  if (overflow_counter >= (1ll << (K - 1))) {
    Normalize(); // Make sure we have enough free carry bits for next accumulation step
  }
}

double Superaccumulator::Round() {
  // BUG for negative numbers
  assert(digits >= 52);
  //printf("imin=%d, imax=%d\n", imin, imax);
  if (imin > imax) {
    return 0;
  }
  bool negative = Normalize();

  // Find leading word
  int i;
  // Skip zeroes
  for (i = imax; accumulator[i] == 0 && i >= imin; --i) {
  }
  if (negative) {
    // Skip ones
    for (; accumulator[i] == ((1ll << digits) - 1) && i >= imin; --i) {
    }
  }
  if (i < 0) {
    // TODO: should we preserve sign of zero?
    return 0.;
  }

  int64_t hiword = negative ? (1ll << digits) - accumulator[i] : accumulator[i];
  double rounded = double(hiword);
  double hi = ldexp(rounded, (i - f_words) * digits);
  if (i == 0) {
    return negative ? -hi : hi;  // Correct rounding achieved
  }
  //printf("i=%d, hiword=%lx, ", i, hiword);
  hiword -= myllrint(rounded);
  double mid = ldexp(double(hiword), (i - f_words) * digits);

  // Compute sticky
  int64_t sticky = 0;
  for (int j = imin; j != i - 1; ++j) {
    sticky |= negative ? (1ll << digits) - accumulator[j] : accumulator[j];
  }

  int64_t loword =
      negative ? (1ll << digits) - accumulator[i - 1] : accumulator[i - 1];
  loword |= !!sticky;
  double lo = ldexp(double(loword), (i - 1 - f_words) * digits);

  //printf("loword=%lx\n", loword);
  //printf("hi=%a, mid=%a, lo=%a\n", hi, mid, lo);

  // Now add3(hi, mid, lo)
  // No overlap, we have already normalized
  if (mid != 0) {
    lo = OddRoundSumNonnegative(mid, lo);
  }
  // Final rounding
  hi = hi + lo;
  return negative ? -hi : hi;
}

// Returns sign
// Does not really normalize!
bool Superaccumulator::Normalize() {
  if (imin > imax) {
    return false;
  }
  overflow_counter = 0;
  int64_t carry_in = accumulator[imin] >> digits;
  accumulator[imin] -= carry_in << digits;
  int i;
  // Sign-extend all the way
  for (i = imin + 1;
//        (i <= imax || carry_in != 0) && i < f_words + e_words;
      i < f_words + e_words; ++i) {
#if 1
    int64_t carry_out = accumulator[i] >> digits;    // Arithmetic shift
    accumulator[i] += carry_in - (carry_out << digits);
#else
    // BUGGY
    // get carry of accumulator[i] + carry_in
    unsigned char overflow;
    int64_t oldword = xadd(accumulator[i], carry_in, overflow);
    bool s = oldword > 0;
    int64_t carrybit = (s ? 1ll << K : -1ll << K);

    int64_t carry_out = (accumulator[i] >> digits) + carrybit;// Arithmetic shift
    accumulator[i] -= carry_out << digits;
#endif
    carry_in = carry_out;
  }
  imax = i - 1;

  if (carry_in != 0 && carry_in != -1) {
    status = Overflow;
  }
  return carry_in < 0;
}

void Superaccumulator::Dump(std::ostream & os) {
  switch (status) {
  case Exact:
    os << "Exact ";
    break;
  case Inexact:
    os << "Inexact ";
    break;
  case Overflow:
    os << "Overflow ";
    break;
  default:
    os << "??";
  }
  os << std::hex;
  for (int i = f_words + e_words - 1; i >= 0; --i) {
    int64_t hi = accumulator[i] >> digits;
    int64_t lo = accumulator[i] - (hi << digits);
    os << "+" << hi << " " << lo;
  }
  os << std::dec;
  os << std::endl;
}

bool Superaccumulator::CompareSuperaccumulatorWithMPFR(mpfr_t *res_mpfr) {
  mpfr_t result;
  mpfr_exp_t exp_ptr;

  mpfr_init2(result, 2098);
  mpfr_set_d(result, 0.0, MPFR_RNDN);

  // BUG for negative numbers
  assert(digits >= 52);

  double error = 0.0;
  for (int j = 0; j < (e_words + f_words); j++) {
    // get the number
    int64_t x = accumulator[j];
    // round
    double xrounded = double(x);
    // scale
    double xscaled = ldexp(xrounded, (j - f_words) * digits);
    // take into account the propagation from the previous step
    if (error != 0) {
      //xscaled = OddRoundSumNonnegative(error, xscaled);
      mpfr_add_d(result, result, error, MPFR_RNDN);
    }
    // compute the current error
    //x -= llrint(xrounded);
    x -= myllrint(xrounded);
    error = ldexp(double(x), (j - f_words) * digits);

    //printf("acc[%d] = %lX \t hi = %e \t mid = %e\n", j, accumulator[j], xscaled,
    //       error);

    mpfr_add_d(result, result, xscaled, MPFR_RNDN);
  }

  char *res = mpfr_get_str(NULL, &exp_ptr, 10, 2098, *res_mpfr, MPFR_RNDD);
  char *res_str = mpfr_get_str(NULL, &exp_ptr, 10, 2098, result, MPFR_RNDD);
  printf("\tSum MPFR (2098)    : %s\n", res);
  printf("\tSum Superaccumulator (2098) : %s\n", res_str);
  mpfr_free_str(res);
  mpfr_free_str(res_str);

  //Compare the results with MPFR using native functions
  bool res_cmp = false;
  int cmp = mpfr_cmp(*res_mpfr, result);
  if (cmp == 0){
      printf("\t The result is EXACT -- matches the MPFR algorithm!\n\n");
      res_cmp = true;
  } else {
      printf("\t The result is WRONG -- does not match the MPFR algorithm!\n\n");
  }

  double rounded_mpfr = mpfr_get_d(*res_mpfr, MPFR_RNDD);
  double rounded_sum = mpfr_get_d(result, MPFR_RNDD);
  printf("Rounded value of MPFR: %.17g\n", rounded_mpfr);
  printf("Rounded value of superaccumulator: %.17g\n", rounded_sum);
  if (rounded_mpfr == rounded_sum){
      printf("\t The result is EXACT -- matches the MPFR algorithm!\n\n");
      res_cmp = true;
  } else {
      printf("\t The result is WRONG -- does not match the MPFR algorithm!\n\n");
  }

  mpfr_clear(result);
  mpfr_free_cache();

  return res_cmp;
}

bool Superaccumulator::CompareRoundedResults(mpfr_t *res_mpfr, double res_rounded) {
  double rounded_mpfr = mpfr_get_d(*res_mpfr, MPFR_RNDD);
  printf("Rounded value of MPFR: %.17g\n", rounded_mpfr);
  printf("Rounded value of superaccumulator: %.17g\n", res_rounded);

  //Compare the results with MPFR using native functions
  bool res_cmp = false;
  if (rounded_mpfr == res_rounded){
      printf("Rounded results of superaccumulator match the results of MPFR\n\n");
      res_cmp = true;
  } else {
      printf("Rounded results of superaccumulator do NOT match the results of MPFR\n\n");
  }

  mpfr_clear(*res_mpfr);
  free(res_mpfr);
  mpfr_free_cache();

  return res_cmp;
}

void Superaccumulator::PrintAccumulator() {
  for (int i = 0; i < (e_words + f_words); i++)
    printf("bin[%3d]: %lX\n", i, accumulator[i]);
}
