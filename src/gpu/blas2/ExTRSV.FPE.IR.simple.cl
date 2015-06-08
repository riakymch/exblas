
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  //For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
    #pragma OPENCL EXTENSION cl_khr_fp64               : enable  // For double precision numbers
    #pragma OPENCL EXTENSION cl_nv_pragma_unroll       : enable
#endif

#define BIN_COUNT  39
#define K           8                   //High-radix carry-save bits
#define digits     56
#define deltaScale 72057594037927936.0  //Assumes K > 0
#define f_words    20
#define TSAFE       0


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
#if 1
double TwoProd(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}
#else
double split(double a, double *a2){
    double factor = 134217729.0;
    double c = factor * a;
    double a1 = c - (c - a);
    *a2 = a - a1;
    return a1;
}

double TwoProd(double a, double b, double *d) {
    double a2, b2;
    double x = a * b;
    double a1 = split(a, &a2);
    double b1 = split(a, &b2);
    *d = a2 * b2 - (((x - a1 * b1) - a2 * b1) - a1 * b2);
    return x;
}
#endif

#ifdef USE_KNUTH
    double TwoSum(double a, double b, double *s) {
        double r = a + b;
        double z = r - a;
        *s = (a - (r - z)) + (b - z);
        return r;
    }
/*#else
    //twosum
    double TwoSum(double a, double b, double *s) {
        double r = a + b;
        int doswap = fabs(b) > fabs(a);
        double a2 = doswap ? b : a;
        double b2 = doswap ? a : b;
        *s = (a2 - r) + b2;
        return r;
    }*/
#endif

// signedcarry in {-1, 0, 1}
long xadd(__global volatile long *sa, long x, uchar *of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    long y = atom_add(sa, x);
    long z = y + x; // since the value sa->accumulator[i] can be changed by another work item

    // TODO: cover also underflow
    *of = 0;
    if(x > 0 && y > 0 && z < 0)
        *of = 1;
    if(x < 0 && y < 0 && z > 0)
        *of = 1;

    return y;
}

double OddRoundSum(double *fpe) {
    double th, tl;
    union {
        double d;
        long l;
    } thdb;

    th = TwoSum(fpe[1], fpe[0], &tl);

    // round to odd th if tl is not zero
    if (tl != 0.0) {
        thdb.d = th;
        // if the mantissa of th is odd, there is nothing to do
        if (!(thdb.l & 1)) {
            // choose the rounding direction
            // depending on the signs of th and tl
            if ((tl > 0.0) ^ (th < 0.0))
                thdb.l++;
            else
                thdb.l--;
            thdb.d = th;
        }
    }

    // final addition rounder to nearest
    return fpe[2] + th;
}


////////////////////////////////////////////////////////////////////////////////
// Kulisch accumulator: rounding and accumulation functions
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

int Normalize(__global long *accumulator, int lda, int *imin, int *imax) {
    long carry_in = accumulator[*imin * lda] >> digits;
    accumulator[*imin * lda] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i * lda] += carry_in;
        long carry_out = accumulator[i * lda] >> digits;    // Arithmetic shift
        accumulator[i * lda] -= (carry_out << digits);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax * lda] += carry_in << digits;

    return carry_in < 0;
}

double Round(__global long *accumulator, int lda) {
    int imin = 0;
    int imax = 38;
    int negative = Normalize(accumulator, lda, &imin, &imax);

    //Find leading word
    int i;
    //Skip zeroes
    for (i = imax; accumulator[i * lda] == 0 && i >= imin; --i) {
    }
    if (negative) {
        //Skip ones
        for (; (accumulator[i * lda] & ((1l << digits) - 1)) == ((1l << digits) - 1) && i >= imin; --i) {
        }
    }
    if (i < 0)
        //TODO: should we preserve sign of zero?
        return 0.0;

    long hiword = negative ? ((1l << digits) - 1) - accumulator[i * lda] : accumulator[i * lda];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - f_words) * digits);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long) rint(rounded);
    double mid = ldexp((double) hiword, (i - f_words) * digits);

    //Compute sticky
    long sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1l << digits) - accumulator[j * lda] : accumulator[j * lda];

    long loword = negative ? (1l << digits) - accumulator[(i - 1) * lda] : accumulator[(i - 1) * lda];
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

void AccumulateWord(__global volatile long *sa, int lda, int i, long x) {
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    uchar overflow;
    long carry = x;
    long carrybit;
    long oldword = xadd(&sa[i * lda], x, &overflow);

    // To propagate over- or underflow
    while (overflow) {
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // accumulator[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> digits;    // Arithmetic shift
        bool s = oldword > 0;
        carrybit = (s ? 1l << K : -1l << K);

        // Cancel carry-save bits
        xadd(&sa[i * lda], (long) -(carry << digits), &overflow);
        if (TSAFE && (s ^ overflow))
            carrybit *= 2;
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT)
            return;
        oldword = xadd(&sa[i * lda], carry, &overflow);
    }
}

void Accumulate(__global volatile long *sa, int lda, double x) {
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

        AccumulateWord(sa, lda, i, xint);

        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Substitution algorithm
////////////////////////////////////////////////////////////////////////////////
/* loops until *sync > val.
 * Needs to be seperate function to force volatile onto *sync.
 */
void wait_until_ge(
    int tid,
    __global volatile int *sync,
    int col_to_wait,
    int *col_done
){
    if(tid == 0) {
        // Only read global memory when necessary
        if (*col_done < col_to_wait) {
            while(*sync < col_to_wait) {}
            *col_done = *sync;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/* Returns next block row index that requires processing */
#if 0
void nextRow(
   __local volatile int *old,
   __global volatile int *address
){
   if(get_local_id(0)==0 && get_local_id(1)==0)
      *old = atomic_add(address, 1);

   barrier(CLK_GLOBAL_MEM_FENCE);
}
#else
int nextRow(
   __global volatile int *address
){
   //__local volatile int old;
   int old;
   if(get_local_id(0)==0 && get_local_id(1)==0)
      old = atomic_add(address, 1);

   barrier(CLK_GLOBAL_MEM_FENCE);
   return old;
}
#endif

/* Sets sync values correctly prior to call to trsv_ln_exec */
__kernel void trsv_init(
    __global int *sync
){
   sync[0] = -1; // Last ready column
   sync[1] = 0;  // Next row to assign
}

/* Copies a nbi x nbi block of a to provided cache.
 * Copies -a and only the half triangle
 */
void tocache(
    __global const double *a,
    __local double *cache,
    const uint nbi,
    const uint ntid,
    const uint trans,
    const uint isunit,
    const uint tid,
    const uint lda
){
    int x = tid % nbi;
    int y = tid / nbi;
    int ty = ntid / nbi;

    if(trans == 0) {
        for (int i = 0; i < nbi; i += ty) {
            if (x > (i + y))
                cache[(i + y) * nbi + x] = -a[(i + y) * lda + x];
            else if ((i + y) < nbi)
                cache[(i + y) * nbi + x] = 0.0;
            if (!isunit && (x == (i + y)))
                cache[x * (nbi + 1)] = a[x * (lda + 1)];
            //    cache[x * (nbi + 1)] = 1.0 / a[x * (lda + 1)];
        }
    }
}

void __trsv_lnn(
    __global double *d_x,
    __global double *d_b,
    __global double *d_a,
    __global int *sync,
    __global long *d_Superaccs,
    __local double* cache,
    const uint n
){
    //__local double cache[BLOCK_SIZE * BLOCK_SIZE];

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);
    int tid  = threadsx * lidy + lidx;
    int isunit = 0;
    int lda = threadsx * threadsy;

    __global long *l_working = d_Superaccs + get_group_id(0) * lda * BIN_COUNT + tid;

    // Get row handled by this block
#if 0
    __local int row;
    row = 0;
    nextRow(&row, &sync[1]);
#else
    int row = nextRow(&sync[1]);
#endif

    // Copy diagonal block to shared memory
    tocache(&d_a[row * BLOCK_SIZE * n + row * BLOCK_SIZE], cache, BLOCK_SIZE, lda, 0, isunit, tid, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over blocks as they become available
    // Initialize accumulators
    for (uint i = 0; i < BIN_COUNT; i++)
        l_working[i * lda] = 0;
    // FPEs
    double fpe[3] = {0.0};

    double x, s, r;
    int col_done = -1;

    for (int col = 0; col < row; col++) {
        wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (int j = 0; j < BLOCK_SIZE; j+=threadsy) {
            double xp = -d_x[col * BLOCK_SIZE + lidy + j];
            x = TwoProd(d_a[(col * BLOCK_SIZE + lidy) * n + row * BLOCK_SIZE + lidx + j * n], xp, &r);

            fpe[2] = TwoSum(fpe[2], x, &s);
            x = s;
            if(x != 0.0) {
                fpe[1] = TwoSum(fpe[1], x, &s);
                x = s;
                if(x != 0.0) {
                    fpe[0] = TwoSum(fpe[0], x, &s);
                    x = s;
                }
            }
            if(x != 0.0) {
                Accumulate(l_working, lda, x);
                //So, there is not space in FPEs -- need to flush to the accumulator
                Accumulate(l_working, lda, fpe[0]);
                Accumulate(l_working, lda, fpe[1]);
                Accumulate(l_working, lda, fpe[2]);
                fpe[2] = 0.0;
                fpe[1] = 0.0;
                fpe[0] = 0.0;
            }

            if(r != 0.0) {
                fpe[2] = TwoSum(fpe[2], r, &s);
                r = s;
                if(r != 0.0) {
                    fpe[1] = TwoSum(fpe[1], r, &s);
                    r = s;
                    if(r != 0.0) {
                        fpe[0] = TwoSum(fpe[0], r, &s);
                        r = s;
                    }
                }
                if(r != 0.0) {
                    Accumulate(l_working, lda, r);
                    Accumulate(l_working, lda, fpe[0]);
                    Accumulate(l_working, lda, fpe[1]);
                    Accumulate(l_working, lda, fpe[2]);
                    fpe[2] = 0.0;
                    fpe[1] = 0.0;
                    fpe[0] = 0.0;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Apply update from diagonal block (row, row)
    if (lidy == 0) {
        double val = 0.0;
        __local volatile double xs;
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (uint i = 0; i < BLOCK_SIZE; i++) {
            if (lidx == i) {
#if 1
                // TODO: remove this if
                //if ((lidx == 22) && (get_group_id(0) == 0))
                //    return;

                // Add the right-hand side
                x = d_x[row * threadsx + lidx];
                fpe[2] = TwoSum(fpe[2], x, &s);
                x = s;
                if(x != 0.0) {
                    fpe[1] = TwoSum(fpe[1], x, &s);
                    x = s;
                    if(x != 0.0) {
                        fpe[0] = TwoSum(fpe[0], x, &s);
                        x = s;
                    }
                }

                if(x != 0.0)
                    Accumulate(l_working, lda, x);

                // Flush FPE to the accumulator
                Accumulate(l_working, lda, fpe[0]);
                Accumulate(l_working, lda, fpe[1]);
                Accumulate(l_working, lda, fpe[2]);

                // Rounding
                val = Round(l_working, lda);
#else
#if 1
                val = OddRoundSum(fpe);
#else
                //Now add3(hi, mid, lo)
                //No overlap, we have already normalized??
                if (fpe[1] != 0)
                    fpe[0] = OddRoundSumNonnegative(fpe[1], fpe[0]);
                //Final rounding
                val = fpe[2] + fpe[0];
#endif
#endif
                // Set FPE to zero
                fpe[2] = 0.0;
                fpe[1] = 0.0;
                fpe[0] = 0.0;

                if (!isunit)
                    //val = val / d_a[row * threadsx * n + row * BLOCK_SIZE + i * n + i];
                    val = val / cache[i * (BLOCK_SIZE + 1)];
                xs = val;
            }
            if (lidx > i) {
                x = TwoProd(cache[i * BLOCK_SIZE + lidx], xs, &r);
#if 0
                int index = 2 * i;
                if ((lidx == 22) && (get_group_id(0) == 0)) {
                    d_b[index] = x; //fpe[2];
                    d_b[index + 1] = r; //fpe[1];
                    //d_b[index + 2] = fpe[0];
                }
#endif

                // TODO: remove this flush
                /*if (i == 25) {
                    Accumulate(l_working, lda, fpe[2]);
                    Accumulate(l_working, lda, fpe[1]);
                    Accumulate(l_working, lda, fpe[0]);
                    fpe[2] = 0.0;
                    fpe[1] = 0.0;
                    fpe[0] = 0.0;
                }*/
                fpe[2] = TwoSum(fpe[2], x, &s);
                x = s;
                if(fabs(x) > 0.0) {
                    fpe[1] = TwoSum(fpe[1], x, &s);
                    x = s;
                    if(fabs(x) > 0.0) {
                        fpe[0] = TwoSum(fpe[0], x, &s);
                        x = s;
                    }
                }
                if(fabs(x) > 0.0) {
                    Accumulate(l_working, lda, x);
                    //So, there is not space in FPEs -- need to flush to the accumulator
                    Accumulate(l_working, lda, fpe[0]);
                    Accumulate(l_working, lda, fpe[1]);
                    Accumulate(l_working, lda, fpe[2]);
                    fpe[2] = 0.0;
                    fpe[1] = 0.0;
                    fpe[0] = 0.0;
                }

                if(fabs(r) > 0.0) {
                    fpe[2] = TwoSum(fpe[2], r, &s);
                    r = s;
                    if(fabs(r) > 0.0) {
                        fpe[1] = TwoSum(fpe[1], r, &s);
                        r = s;
                        if(fabs(r) > 0.0) {
                            fpe[0] = TwoSum(fpe[0], r, &s);
                            r = s;
                        }
                    }
                    if(fabs(r) > 0.0) {
                        Accumulate(l_working, lda, r);
                        Accumulate(l_working, lda, fpe[0]);
                        Accumulate(l_working, lda, fpe[1]);
                        Accumulate(l_working, lda, fpe[2]);
                        fpe[2] = 0.0;
                        fpe[1] = 0.0;
                        fpe[0] = 0.0;
                    }
                }
#if 0
                if (i > 0) {
                    int index = 3 * (i - 1);
                    if ((lidx == 22) && (get_group_id(0) == 0)) {
                        d_b[index] = fpe[0];
                        d_b[index + 1] = fpe[1];
                        d_b[index + 2] = fpe[2];
                    }
                }
#endif
            }
        }
        d_x[row * BLOCK_SIZE + tid] = val;
    }

    // Notify other blocks that soln is ready for this row
    barrier(CLK_GLOBAL_MEM_FENCE); // Wait for d_x to be visible to other blocks
    if(tid == 0)
        atomic_add(&sync[0], 1);   // Use atomicAdd to bypass L1 miss
    barrier(CLK_GLOBAL_MEM_FENCE); // Flush sync[0] asap
}


void __trsv_lnn_simple(
    __global double *d_x,
    __global double *d_b,
    __global double *d_a,
    __global int *sync,
    __global long *d_Superaccs,
    const uint n
){
    double r, s, z;
    int lda = 1;

    __global long *l_working = d_Superaccs;

    for (uint i = 0; i < n; i++) {
        double fpe[3] = {0.0};

        for (uint j = 0; j < BIN_COUNT; j++)
            l_working[j] = 0.0;

        for(uint j = 0; j < i; j++) {
            r = TwoProd(d_a[j * n + i], -d_x[j], &s);

            fpe[2] = TwoSum(fpe[2], r, &z);
            r = z;
            if (r != 0.0) {
                fpe[1] = TwoSum(fpe[1], r, &z);
                r = z;
                if (r != 0.0) {
                    fpe[0] = TwoSum(fpe[0], r, &z);
                    r = z;
                }
            }
            if (r != 0.0)
                Accumulate(l_working, lda, r);

            if (s != 0.0) {
                fpe[2] = TwoSum(fpe[2], s, &z);
                s = z;
                if (s != 0.0) {
                    fpe[1] = TwoSum(fpe[1], s, &z);
                    s = z;
                    if (s != 0.0) {
                        fpe[0] = TwoSum(fpe[0], s, &z);
                        s = z;
                    }
                }
            }
            if (s != 0.0)
                Accumulate(l_working, lda, s);
        }

        //Accumulate(l_working, lda, d_x[i]);
        r = d_x[i];
        fpe[2] = TwoSum(fpe[2], r, &z);
        r = z;
        if (r != 0.0) {
            fpe[1] = TwoSum(fpe[1], r, &z);
            r = z;
            if (r != 0.0) {
                fpe[0] = TwoSum(fpe[0], r, &z);
                r = z;
            }
        }
        if (r != 0.0)
            Accumulate(l_working, lda, r);

        Accumulate(l_working, lda, fpe[2]);
        Accumulate(l_working, lda, fpe[1]);
        Accumulate(l_working, lda, fpe[0]);
        double sum = Round(l_working, lda);

        d_x[i] = sum / d_a[i * (n + 1)];
    }
}


__kernel void trsv_lnn(
    __global double *d_x,
    __global double *d_a,
    __global double *d_b,
    __global int *sync,
    __global long *d_Superaccs,
    const uint n
){

    // At first we call ExTRSV. d_x holds the result
    //__trsv_lnn_simple(d_x, d_b, d_a, sync, d_Superaccs, n);
    __trsv_lnn(d_x, d_b, d_a, sync, d_Superaccs, n);

    //int lidx = get_local_id(0);
    //d_x[get_group_id(0) * threadsx + lidx] = d_b[get_group_id(0) * threadsx + lidx];

    return;

    // One step of iterative refinement
    /* ExGEMV: rm = A x - b. d_b holds the result
       ExTRSV: A rm = xm. d_b contains the result
       ExAXPY: x = x + xm. d_x contains the final result */

    //__gemv(d_b, d_a, d_x, d_Superaccs, n);

    //trsv_init(sync);
    //__trsv_lnn(d_b, d_a, sync, d_Superaccs, n);

    //__axpy(d_x, d_b, n);

    //trsv_init(sync);
    //__trsv_ir(d_x, d_a, d_b, sync, d_Superaccs, n);
}
