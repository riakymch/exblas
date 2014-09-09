
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  //For double precision numbers
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  //For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
    #pragma OPENCL EXTENSION cl_nv_pragma_unroll       : enable
#endif

//Data type used for input data fetches
typedef double data_t;

#define BIN_COUNT  39
#define K          8                    //High-radix carry-save bits
#define digits     56
#define deltaScale 72057594037927936.0  //Assumes K > 0
#define f_words    20
#define TSAFE      0


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

double KnuthTwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
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

int Normalize(__global long *accumulator, int *imin, int *imax) {
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

double Round(__global long *accumulator) {
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


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial accumulators
////////////////////////////////////////////////////////////////////////////////
void AccumulateWord(__local volatile long *sa, int i, long x) {
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    uchar overflow;
    long carry = x;
    long carrybit;
    long oldword = xadd(&sa[i * WARP_COUNT], x, &overflow);

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
        xadd(&sa[i * WARP_COUNT], (long) -(carry << digits), &overflow);
        if (TSAFE && (s ^ overflow))
            carrybit *= 2;
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT)
            return;
        oldword = xadd(&sa[i * WARP_COUNT], carry, &overflow);
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

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void DDOT(
    __global long *d_PartialSuperaccs,
    __global data_t *d_a,
    __global data_t *d_b,
    const uint NbElements
){
    __local long l_sa[WARP_COUNT * BIN_COUNT] __attribute__((aligned(8)));
    __local long *l_workingBase = l_sa + (get_local_id(0) & (WARP_COUNT - 1));

    //Initialize accumulators
    for (uint i = 0; i < BIN_COUNT; i++)
        l_workingBase[i * WARP_COUNT] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    //Read data from global memory and scatter it to sub-accumulators
    double a[4] = {0.0};
    for(uint pos = get_global_id(0); pos < NbElements; pos += get_global_size(0)){
        double r = 0.0;
        data_t x = TwoProductFMA(d_a[pos], d_b[pos], &r);

        double s;
        a[0] = KnuthTwoSum(a[0], x, &s);
        x = s;
        if(x != 0.0) {
            a[1] = KnuthTwoSum(a[1], x, &s);
            x = s;
            if(x != 0.0) {
                a[2] = KnuthTwoSum(a[2], x, &s);
                x = s;
                if(x != 0.0) {
                    a[3] = KnuthTwoSum(a[3], x, &s);
                    x = s;
                }
            }
        }
        if(x != 0.0)
            Accumulate(l_workingBase, x);

        a[0] = KnuthTwoSum(a[0], r, &s);
        r = s;
        if(r != 0.0) {
            a[1] = KnuthTwoSum(a[1], r, &s);
            r = s;
            if(r != 0.0) {
                a[2] = KnuthTwoSum(a[2], r, &s);
                r = s;
                if(r != 0.0) {
                    a[3] = KnuthTwoSum(a[3], r, &s);
                    r = s;
                }
            }
        }
        if(r != 0.0)
            Accumulate(l_workingBase, r);
    }
    //Flush to the accumulator
    Accumulate(l_workingBase, a[0]);
    Accumulate(l_workingBase, a[1]);
    Accumulate(l_workingBase, a[2]);
    Accumulate(l_workingBase, a[3]);
    barrier(CLK_LOCAL_MEM_FENCE);

    //Merge sub-accumulators into work-group partial-accumulator
    uint pos = get_local_id(0);
    if (pos < BIN_COUNT){
        long sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];
        barrier(CLK_LOCAL_MEM_FENCE);

        d_PartialSuperaccs[get_group_id(0) * BIN_COUNT + pos] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge SuperAccumulators
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void DDOTComplete(
    __global long *d_Superacc,
    __global long *d_PartialSuperaccs,
    const uint NbPartialSuperaccs
){
    __local long l_Data[MERGE_WORKGROUP_SIZE];

    //Reduce to one work group
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);

    long sum = 0;
    #ifdef NVIDIA
        #pragma unroll
    #endif
    for(uint i = lid; i < NbPartialSuperaccs; i += MERGE_WORKGROUP_SIZE)
        sum += d_PartialSuperaccs[gid + i * BIN_COUNT];
    l_Data[lid] = sum;

    //Reduce within the work group
    #ifdef NVIDIA
        #pragma unroll
    #endif
    for(uint stride = MERGE_WORKGROUP_SIZE / 2; stride > 0; stride >>= 1){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < stride)
            l_Data[lid] += l_Data[lid + stride];
    }

    if(lid == 0)
        d_Superacc[gid] = l_Data[0];
}

////////////////////////////////////////////////////////////////////////////////
// Round the results
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void DDOTRound(
    __global double *d_res,
    __global long *d_Superacc
){
    uint pos = get_local_id(0);
    if (pos == 0)
        d_res[0] = Round(d_Superacc);
}
