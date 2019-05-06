/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  // For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_khr_fp64                 : enable  // For double precision numbers
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

typedef double data_t;

#define BIN_COUNT      39
#define K               8                   // High-radix carry-save bits
#define digits         56
#define deltaScale     72057594037927936.0  // Assumes K>0
#define f_words        20
#define TSAFE           0

#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
double KnuthTwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}

double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

// signedcarry in {-1, 0, 1}
long xadd(long *sa, long x, uchar *of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    long y = *sa;
    *sa = *sa + x;

    // TODO: cover also underflow
    *of = 0;
    if(x > 0 && y > 0 && *sa < 0)
        *of = 1;
    if(x < 0 && y < 0 && *sa > 0)
        *of = 1;

    return y;
}


////////////////////////////////////////////////////////////////////////////////
// Rounding functions
////////////////////////////////////////////////////////////////////////////////
double OddRoundSumNonnegative(double th, double tl) {
    union {
        double d;
        long l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

int Normalize(long *accumulator, int *imin, int *imax) {
    long carry_in = accumulator[*imin] >> digits;
    accumulator[*imin] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i] += carry_in;
        long carry_out = accumulator[i] >> digits;    // Arithmetic shift
        accumulator[i] -= (carry_out << digits);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax] += carry_in << digits;

    return carry_in < 0;
}

double Round(long *accumulator) {
    int imin = 0;
    int imax = 38;
    int negative = Normalize(accumulator, &imin, &imax);

    //Find leading word
    int i;
    //Skip zeroes
    for (i = imax; accumulator[i] == 0 && i >= imin; --i) {
    }
    if (negative) {
        //Skip ones
        for(; (accumulator[i] & ((1l << digits) - 1)) == ((1l << digits) - 1) && i >= imin; --i) {
        }
    }
    if (i < 0)
        //TODO: should we preserve sign of zero?
        return 0.0;

    long hiword = negative ? ((1l << digits) - 1) - accumulator[i] : accumulator[i];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - f_words) * digits);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long) rint(rounded);
    double mid = ldexp((double) hiword, (i - f_words) * digits);

    //Compute sticky
    long sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1l << digits) - accumulator[j] : accumulator[j];

    long loword = negative ? (1l << digits) - accumulator[i - 1] : accumulator[i - 1];
    loword |= !!sticky;
    double lo = ldexp((double) loword, (i - 1 - f_words) * digits);

    //Now add3(hi, mid, lo)
    //No overlap, we have already normalized
    if (mid != 0)
        lo = OddRoundSumNonnegative(mid, lo);

    //Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial accumulators
////////////////////////////////////////////////////////////////////////////////
void AccumulateWord(long *sa, int i, long x) {
  // With atomic accumulator updates
  // accumulation and carry propagation can happen in any order
  long carry = x;
  long carrybit;
  uchar overflow;
  long oldword = xadd(&sa[i], x, &overflow);

  // To propagate over- or underflow
  while (overflow) {
    // Carry or borrow
    // oldword has sign S
    // x has sign S
    // accumulator[i] has sign !S (just after update)
    // carry has sign !S
    // carrybit has sign S
    carry = (oldword + carry) >> digits;
    bool s = oldword > 0;
    carrybit = (s ? 1l << K : -1l << K);

    // Cancel carry-save bits
    xadd(&sa[i], (long) -(carry << digits), &overflow);
    if (TSAFE && (s ^ overflow)) {
      carrybit *= 2;
    }
    carry += carrybit;

    ++i;
    if (i >= BIN_COUNT) {
      return;
    }
    oldword = xadd(&sa[i], carry, &overflow);
  }
}

void Accumulate(long *sa, double x) {
  if (x == 0)
    return;

  int e;
  frexp(x, &e);
  int exp_word = e / digits;  // Word containing MSbit
  int iup = exp_word + f_words;

  double xscaled = ldexp(x, -digits * exp_word);

  int i;
  for (i = iup; xscaled != 0; --i) {
    double xrounded = rint(xscaled);
    long xint = (long) xrounded;

    AccumulateWord(sa, i, xint);

    xscaled -= xrounded;
    xscaled *= deltaScale;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Matrix multiplication on the device: C := beta * C + alpha * A * B.
//     So far just C = C + A * B
////////////////////////////////////////////////////////////////////////////////
__kernel void gemm(
    uint m,
    uint n,
    uint k,
    double alpha,
    __global data_t* A,
    uint lda,
    __global data_t* B,
    uint ldb,
    double beta,
    __global data_t* C,
    uint ldc,
    __local data_t* As,
    __local data_t* Bs
) {
    //Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    //Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    //Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * m;

    //int bdimx = n / BLOCK_SIZE;
    int bdimy = m / BLOCK_SIZE;
    //int bsizex = get_num_groups(0);
    int bsizey = get_num_groups(1);

    //for (int i = bx; i < bdimx; i += bsizex) {
        for (int j = by; j < bdimy; j += bsizey) {
            //Index of the first sub-matrix of A processed by the block
            int aBegin = m * BLOCK_SIZE * by;

            //Index of the last sub-matrix of A processed by the block
            int aEnd   = aBegin + m - 1;

            //A superaccumulator that corresponds to a single value in the matrix C
            long p_workingBase[BIN_COUNT] = {0};

            //for floating-point expansion
            double sum[4] = {0.0};

            //Loop over all the sub-matrices of A and B required to compute the block sub-matrix
            for (int a = aBegin, b = bBegin;
                     a <= aEnd;
                     a += aStep, b += bStep) {
                //Load the matrices from device memory to shared memory;
                //each thread loads one element of each matrix
                AS(ty, tx) = A[a + m * ty + tx];
                BS(ty, tx) = B[b + m * ty + tx];

                //Synchronize to make sure the matrices are loaded
                barrier(CLK_LOCAL_MEM_FENCE);

                //Multiply the two matrices together;
                //each thread computes one element of the block sub-matrix
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    double r; //residual of multiplication
                    double x = TwoProductFMA(AS(ty, k), BS(k, tx), &r);

                    double s; //residual of addition
                    sum[0] = KnuthTwoSum(sum[0], x, &s);
                    x = s;
                    if(x != 0.0) {
                        sum[1] = KnuthTwoSum(sum[1], x, &s);
                        x = s;
                        if(x != 0.0) {
                            sum[2] = KnuthTwoSum(sum[2], x, &s);
                            x = s;
                            if(x != 0.0) {
                                sum[3] = KnuthTwoSum(sum[3], x, &s);
                                x = s;
                            }
                        }
                    }
                    if(x != 0.0) {
                        Accumulate(p_workingBase, x);
                        //Flush to the superacc
                        Accumulate(p_workingBase, sum[0]);
                        Accumulate(p_workingBase, sum[1]);
                        Accumulate(p_workingBase, sum[2]);
                        Accumulate(p_workingBase, sum[3]);
                        sum[0] = 0.0;
                        sum[1] = 0.0;
                        sum[2] = 0.0;
                        sum[3] = 0.0;
                    }

                    //if(r != 0.0) {
                        /*sum[0] = KnuthTwoSum(sum[0], r, &s);
                        r = s;
                        if(r != 0.0) {*/
                            sum[1] = KnuthTwoSum(sum[1], r, &s);
                            r = s;
                            if(r != 0.0) {
                                sum[2] = KnuthTwoSum(sum[2], r, &s);
                                r = s;
                                if (r != 0.0) {
                                    sum[3] = KnuthTwoSum(sum[3], r, &s);
                                    r = s;
                                }
			    }
			//}
                        if(r != 0.0) {
                            Accumulate(p_workingBase, r);
			    #ifdef NVIDIA
                                //Flush to the superacc
                                Accumulate(p_workingBase, sum[0]);
                                Accumulate(p_workingBase, sum[1]);
                                Accumulate(p_workingBase, sum[2]);
                                Accumulate(p_workingBase, sum[3]);
                                sum[0] = 0.0;
                                sum[1] = 0.0;
                                sum[2] = 0.0;
                                sum[3] = 0.0;
			    #endif
                        }
                    //}
                }

                //Synchronize to make sure that the preceding computation is done before 
                //loading two new sub-matrices of A and B in the next iteration
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            //Flush to the accumulator
            Accumulate(p_workingBase, sum[0]);
            Accumulate(p_workingBase, sum[1]);
            Accumulate(p_workingBase, sum[2]);
            Accumulate(p_workingBase, sum[3]);
            sum[0] = 0.0;
            sum[1] = 0.0;
            sum[2] = 0.0;
            sum[3] = 0.0;

            //TODO: the first non-zero from rigth
            int c = (m * by + bx) * BLOCK_SIZE;
            C[c + m * ty + tx] += Round(p_workingBase);
        }
    //}
}

