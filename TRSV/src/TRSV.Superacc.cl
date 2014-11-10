
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  //For double precision numbers
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  //For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
    #pragma OPENCL EXTENSION cl_nv_pragma_unroll       : enable
#endif

#define BIN_COUNT  76
#define K          8                    //High-radix carry-save bits
#define digits     56
#define deltaScale 72057594037927936.0  //Assumes K > 0
#define f_words    39 
#define TSAFE      0


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

// signedcarry in {-1, 0, 1}
long xadd(__local volatile long *sa, long x, uchar *of) {
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

int Normalize(__local long *accumulator, int *imin, int *imax) {
    if (*imin > *imax)
        return 0;

    long carry_in = accumulator[*imin] >> digits;
    accumulator[*imin] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
#if 1
        long carry_out = accumulator[i] >> digits;    // Arithmetic shift
        accumulator[i] += carry_in - (carry_out << digits);
#else
        // BUGGY
        // get carry of accumulator[i] + carry_in
        unsigned char overflow;
        long oldword = xadd(&accumulator[i], carry_in, &overflow);
        int s = oldword > 0;
        long carrybit = (s ? 1ll << K : -1ll << K);

        long carry_out = (accumulator[i] >> digits) + carrybit;// Arithmetic shift
        accumulator[i] -= carry_out << digits;
#endif
        carry_in = carry_out;
    }
    *imax = i - 1;

    if ((carry_in != 0) && (carry_in != -1)) {
        //TODO: handle overflow
        //status = Overflow;
    }

    return carry_in < 0;
}

double Round(__local long *accumulator) {
    int imin = 0;
    int imax = 75;
    int negative = Normalize(accumulator, &imin, &imax);

    //Find leading word
    int i;
    //Skip zeroes
    for (i = imax; accumulator[i] == 0 && i >= imin; --i) {
    }
    if (negative) {
        //Skip ones
        for (; accumulator[i] == ((1 << digits) - 1) && i >= imin; --i) {
        }
    }
    if (i < 0)
        //TODO: should we preserve sign of zero?
        return 0.0;

    long hiword = negative ? (1 << digits) - accumulator[i] : accumulator[i];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - f_words) * digits);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long) rint(rounded);
    double mid = ldexp((double) hiword, (i - f_words) * digits);

    //Compute sticky
    long sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1 << digits) - accumulator[j] : accumulator[j];

    long loword = negative ? (1 << digits) - accumulator[i - 1] : accumulator[i - 1];
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

void AccumulateWord(__local volatile long *sa, int i, long x) {
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    uchar overflow;
    long carry = x;
    long carrybit;
    long oldword = xadd(&sa[i * BLOCK_SIZE], x, &overflow);

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
        xadd(&sa[i * BLOCK_SIZE], (long) -(carry << digits), &overflow);
        if (TSAFE && (s ^ overflow))
            carrybit *= 2;
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT)
            return;
        oldword = xadd(&sa[i * BLOCK_SIZE], carry, &overflow);
    }
}

void Accumulate(__local volatile long *sa, double x) {
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


////////////////////////////////////////////////////////////////////////////////
// Substitution algorithm
////////////////////////////////////////////////////////////////////////////////
double dblkSolver(
    __local double *a,
    const uint isunit,
    const int lda,
    double val
){
    volatile __local double xs;
    uint lidx = get_local_id(0);

    __local long l_sa[BLOCK_SIZE * BIN_COUNT] __attribute__((aligned(8)));
    __local long *l_working = l_sa + (get_local_id(0) & (BLOCK_SIZE - 1));

    //Initialize accumulators
    for (uint i = 0; i < BIN_COUNT; i++)
        l_working[i * BLOCK_SIZE] = 0;
    //Accumulate val
    Accumulate(l_working, val);
    barrier(CLK_LOCAL_MEM_FENCE);

    #ifdef NVIDIA
       #pragma unroll
    #endif
    for (uint i = 0; i < BLOCK_SIZE; i++) {
        if (lidx == i) {
            val = Round(l_working);
            if (!isunit) {
                /*//TODO: this part could be and should be done without using Kulisch accumulator
                double r = 0.0;
                double x = TwoProductFMA(val, a[i * (lda + 1)], &r);

                Accumulate(l_working, x);
                if (r != 0.0)
                    Accumulate(l_working, r);*/
                val *= a[i * (lda + 1)];
            }
            xs = val;
        }
        if (lidx > i) {
            double r = 0.0;
            double x = TwoProductFMA(xs, a[i * lda + lidx], &r);

            Accumulate(l_working, x);
            if (r != 0.0)
                Accumulate(l_working, r);
        }
    }

    return val;
}

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
        /* Only read global memory when necessary */
        if (*col_done < col_to_wait) {
            while(*sync < col_to_wait) {}
            *col_done = *sync;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/* Returns next block row index that requires processing */
int nextRow(
   __global volatile int *address
){
   __local volatile int old;
   if(get_local_id(0)==0 && get_local_id(1)==0)
      old = atomic_add(address, 1);

   barrier(CLK_GLOBAL_MEM_FENCE);
   return old;
}

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
    const uint nbi,
    const uint ntid,
    const uint trans,
    const uint isunit,
    const uint tid,
    const uint lda,
    __local double *cache
){
    int x = tid % nbi;
    int y = tid / nbi;
    int ty = ntid/nbi;
    //int lidx = get_local_id(0);
    //int lidy = get_local_id(1);

    if(trans == 0) {
        /*for (int j = 0; j < BLOCK_SIZE; j+=threadsy) {
            if (lidx > (lidy + j))
                cache[threadsx * (lidy + j) + lidx] = a[lda * (lidy + j) + lidx];
            else if ((lidy + j) < BLOCK_SIZE)
                cache[threadsx * (lidy + j) + lidx] = 0.0;
            if (isunit && (lidx == (lidy + j)))
                cache[threadsx * (lidy + j) + lidx] = 1.0;
        }*/
        for (int i = 0; i < nbi; i += ty) {
            if (x > (i + y))
                cache[(i + y) * nbi + x] = -a[(i + y) * lda + x];
            else if ((i + y) < nbi)
                cache[(i + y) * nbi + x] = 0.0;
            if (!isunit && (x == (i + y)))
                cache[x * (nbi + 1)] = 1.0 / a[x * (lda + 1)];
        }
    }
}

//__attribute__((reqd_work_group_size(threadsx, threadsy, 1)))
__kernel void trsv_lnn(
    //__global long *d_Superaccs,
    __global double *d_x,
    __global double *d_a,
    __global double *d_b,
    __global int *sync,
    const uint n
){
    __local double cache[BLOCK_SIZE * BLOCK_SIZE];
    __local double partSum[threadsy * threadsx];

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);
    int tid  = threadsx * lidy + lidx;
    int isunit = 0;

    // Get row handled by this block
    int row = nextRow(&sync[1]);

    // Copy diagonal block to shared memory
    tocache(&d_a[row * BLOCK_SIZE * n + row * BLOCK_SIZE], BLOCK_SIZE, threadsx * threadsy, 0, isunit, tid, n, cache);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over blocks as they become available
    double val = 0.0;
    if(lidy == 0)
        val = -d_b[row * BLOCK_SIZE + lidx];
    int col_done = -1;

    for (int col = 0; col < row; col++) {
        wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (int j = 0; j < BLOCK_SIZE; j+=threadsy)
            val += d_a[(col * BLOCK_SIZE + lidy) * n + row * BLOCK_SIZE + lidx + j * n] * d_x[col * BLOCK_SIZE + lidy + j];
    }
    partSum[lidy * threadsx + lidx] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Apply update from diagonal block (row, row)
    if (lidy == 0) {
        for(int i = 1; i < threadsy; i++)
            val += partSum[i * threadsx + lidx];
        val = -val;

        barrier(CLK_LOCAL_MEM_FENCE);
        val = dblkSolver(cache, isunit, BLOCK_SIZE, val);
        d_x[row * BLOCK_SIZE + tid] = val;
    }

    // Notify other blocks that soln is ready for this row
    barrier(CLK_GLOBAL_MEM_FENCE); // Wait for d_x to be visible to other blocks
    if(tid == 0)
        atomic_add(&sync[0], 1);   // Use atomicAdd to bypass L1 miss
    barrier(CLK_GLOBAL_MEM_FENCE); // Flush sync[0] asap
}
