/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  // For double precision numbers
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  // For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
    #pragma OPENCL EXTENSION cl_nv_pragma_unroll       : enable
#endif

//Data type used for input data fetches
typedef double2 data_t;

#define BIN_COUNT      39
#define K               8                   // High-radix carry-save bits
#define digits         56
#define deltaScale     72057594037927936.0  // Assumes K>0
#define f_words        20
#define TSAFE           0
#define WORKGROUP_SIZE (WARP_COUNT * WARP_SIZE)


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

int NormalizeLocal(__local long *accumulator, int *imin, int *imax) {
    long carry_in = (accumulator[*imin * WARP_COUNT] >> digits);
    accumulator[*imin * WARP_COUNT] -= (carry_in << digits);
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i * WARP_COUNT] += carry_in;
        long carry_out = (accumulator[i * WARP_COUNT] >> digits);    // Arithmetic shift
        accumulator[i * WARP_COUNT] -= (carry_out << digits);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax * WARP_COUNT] += carry_in << digits;

    return carry_in < 0;
}

int Normalize(__global long *accumulator, int *imin, int *imax) {
    long carry_in = (accumulator[*imin] >> digits);
    accumulator[*imin] -= (carry_in << digits);
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i] += carry_in;
        long carry_out = (accumulator[i] >> digits);    // Arithmetic shift
        accumulator[i] -= (carry_out << digits);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax] += carry_in << digits;

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
        for (; (accumulator[i] & ((1l << digits) - 1)) == ((1l << digits) - 1) && i >= imin; --i) {
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
// Main computation pass: compute partial superaccs
////////////////////////////////////////////////////////////////////////////////
bool Accumulate(__local volatile long *sa, __local volatile uint *check, double x) {
    if (x == 0)
        return false;

    int e;
    bool is_norm = false;
    frexp(x, &e);
    int exp_word = e / digits;  // Word containing MSbit
    int iup = exp_word + f_words;

    double xscaled = ldexp(x, -digits * exp_word);

    int i;
    for (i = iup; xscaled != 0; --i) {
        double xrounded = rint(xscaled);
        long xint = (long) xrounded;

        atom_add(&sa[i * WARP_COUNT], xint);
        atomic_inc(&check[i * WARP_COUNT]);
        if (check[i * WARP_COUNT] > 256 - 4 * WARP_SIZE)
            is_norm = true;

        xscaled -= xrounded;
        xscaled *= deltaScale;
    }

    return is_norm;
}

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void ExSUM(
    __global long *d_PartialSuperaccs,
    __global data_t *d_Data,
    const uint inca,
    const uint NbElements
) {
    __local long l_sa[WARP_COUNT * BIN_COUNT] __attribute__((aligned(8)));
    __local uint l_scheck[WARP_COUNT * BIN_COUNT];
    __local long *l_workingBase = l_sa + (get_local_id(0) & (WARP_COUNT - 1));
    __local uint *l_check = l_scheck + (get_local_id(0) & (WARP_COUNT - 1));

    //Initialize superaccs
    for (uint i = 0; i < BIN_COUNT; i++) {
        l_workingBase[i * WARP_COUNT] = 0;
        l_check[i * WARP_COUNT] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Read data from global memory and scatter it to sub-superaccs
    for(uint pos = get_global_id(0); pos < NbElements; pos += get_global_size(0)){
        data_t x = d_Data[pos];

        __local bool is_norm;
        is_norm = Accumulate(l_workingBase, l_check, x.x);
        is_norm &= Accumulate(l_workingBase, l_check, x.y);
        if (is_norm) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (get_local_id(0) < WARP_SIZE) {
                int imin = 0;
                int imax = 38;
                NormalizeLocal(l_workingBase, &imin, &imax);
            }
            for (uint i = 0; i < BIN_COUNT; i++)
                l_check[i * WARP_COUNT] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint pos = get_local_id(0);
    int imin = 0;
    int imax = 38;
    if (pos < WARP_COUNT) {
        NormalizeLocal(l_workingBase, &imin, &imax);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Merge sub-superaccs into work-group partial-superacc
#if 1
    if (pos < BIN_COUNT) {
        long sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];
        barrier(CLK_LOCAL_MEM_FENCE);

        d_PartialSuperaccs[get_group_id(0) * BIN_COUNT + pos] = sum;
    }
#else
    /*if (pos == 0){
        for (uint j = 0; j < BIN_COUNT; j++) {
            for (uint i = 1; i < WARP_COUNT; i++) {
                AccumulateWord(l_sa, j, l_sa[j * WARP_COUNT + i]);
            }
            d_PartialSuperaccs[j] = l_sa[j * WARP_COUNT];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);*/
    if (pos < BIN_COUNT) {
        for(uint i = 1; i < WARP_COUNT; i++) {
            atom_add(&l_sa[pos * WARP_COUNT], l_sa[pos * WARP_COUNT + i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        d_PartialSuperaccs[get_group_id(0) * BIN_COUNT + pos] = l_sa[pos * WARP_COUNT];
    }
    /*if (pos == 0) {
        for (uint j = 0; j < BIN_COUNT; j++) {
            d_PartialSuperaccs[j] = l_sa[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);*/
#endif
    /*if (pos == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[get_group_id(0) * BIN_COUNT], &imin, &imax);
    }*/
}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void ExSUMComplete(
    __global long *d_Superacc,
    __global long *d_PartialSuperaccs,
    uint PartialSuperaccusCount
) {
    uint lid = get_local_id(0);
#if 1
    __local long l_Data[MERGE_WORKGROUP_SIZE];

    //Reduce to one work group
    uint gid = get_group_id(0);

    long sum = 0;
    for(uint i = lid; i < PartialSuperaccusCount; i += MERGE_WORKGROUP_SIZE)
        sum += d_PartialSuperaccs[gid + i * BIN_COUNT];
    l_Data[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reduce within the work group
    for(uint stride = MERGE_WORKGROUP_SIZE / 2; stride > 0; stride >>= 1){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < stride)
            l_Data[lid] += l_Data[lid + stride];
    }

    if(lid == 0)
        d_Superacc[gid] = l_Data[0];
#else

    if (lid < BIN_COUNT) {
        for(uint i = 1; i < PartialSuperaccusCount; i++) {
            AccumulateWordGlobal(d_PartialSuperaccs, lid, d_PartialSuperaccs[lid + i * BIN_COUNT]);
        }
        d_Superacc[lid] = d_PartialSuperaccs[lid];
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Round the results
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void ExSUMRound(
    __global double *d_Res,
    __global long *d_Superacc
){
    uint pos = get_local_id(0);
    if (pos == 0)
        d_Res[0] = Round(d_Superacc);
}

