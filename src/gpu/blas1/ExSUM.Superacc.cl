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


#define BIN_COUNT      39
#define K               8                   // High-radix carry-save bits
#define digits         56
#define deltaScale     72057594037927936.0  // Assumes K>0
#define f_words        20
#define TSAFE           0
#define WORKGROUP_SIZE (WARP_COUNT * WARP_SIZE)


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
// signedcarry in {-1, 0, 1}
long xadd(__local volatile long *sa, long x, uchar *of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    long y = atom_add(sa, x);
    long z = y + x; // since the value sa->superacc[i] can be changed by another work item

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

int Normalize_local(__local long *accumulator, int *imin, int *imax) {
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
    accumulator[*imax * WARP_COUNT] += (carry_in << digits);

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
    accumulator[*imax] += (carry_in << digits);

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
        for(; (accumulator[i] & ((1l << digits) - 1)) == ((1l << digits) - 1) && i >= imin; --i) {
        }
    }
    if (i < 0)
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
        sticky |= negative ? ((1l << digits) - accumulator[j]) : accumulator[j];

    long loword = negative ? ((1l << digits) - accumulator[i - 1]) : accumulator[i - 1];
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
void AccumulateWord(__local volatile long *sa, int i, long x) {
    // With atomic superacc updates
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
        // superacc[i] has sign !S (just after update)
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

void Accumulate(__local volatile long *sa, __local bool *res, double x) {
//void Accumulate(__local volatile long *sa, double x) {
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

        //AccumulateWord(sa, i, xint);
        atom_add(&sa[i * WARP_COUNT], xint);
        if ((sa[i * WARP_COUNT] & 0x000000000000003F) > 0)
            *res = true;

        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
}

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void ExSUM(
    __global long *d_PartialSuperaccs,
    __global double *d_Data,
    const uint inca,
    const uint offset,
    const uint NbElements
) {
    __local long l_sa[WARP_COUNT * BIN_COUNT] __attribute__((aligned(8)));
    __local long *l_workingBase = l_sa + (get_local_id(0) & (WARP_COUNT - 1));
    __local bool l_sa_check[WARP_COUNT];
    __local bool *l_workingBase_check = l_sa_check + (get_local_id(0) & (WARP_COUNT - 1));

    //Initialize superaccs
    for (uint i = 0; i < BIN_COUNT; i++)
        l_workingBase[i * WARP_COUNT] = 0;
    *l_workingBase_check = false;
    barrier(CLK_LOCAL_MEM_FENCE);

    //Read data from global memory and scatter it to sub-superaccs
	if ((offset == 0) && (inca == 1)) {
		for(uint pos = get_global_id(0); pos < NbElements; pos += get_global_size(0)) {
			double x = d_Data[pos];

			//Accumulate(l_workingBase, x);
			Accumulate(l_workingBase, l_workingBase_check, x);
			if (*l_workingBase_check) {
				barrier(CLK_LOCAL_MEM_FENCE);
				if (get_local_id(0) < WARP_COUNT){
					int imin = 0;
					int imax = 38;
					Normalize_local(l_workingBase, &imin, &imax);
				}
				*l_workingBase_check = false;
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
	} else {
		for(uint pos = get_global_id(0); pos < NbElements; pos += get_global_size(0)) {
			double x = d_Data[offset + pos * inca];

			//Accumulate(l_workingBase, x);
			Accumulate(l_workingBase, l_workingBase_check, x);
			if (*l_workingBase_check) {
				barrier(CLK_LOCAL_MEM_FENCE);
				if (get_local_id(0) < WARP_COUNT){
					int imin = 0;
					int imax = 38;
					Normalize_local(l_workingBase, &imin, &imax);
				}
				*l_workingBase_check = false;
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    /*if (get_local_id(0) < WARP_COUNT){
        int imin = 0;
        int imax = 38;
        Normalize_local(l_workingBase, &imin, &imax);
    }
    barrier(CLK_LOCAL_MEM_FENCE);*/

    //Merge sub-superaccs into work-group partial-accumulator
    uint pos = get_local_id(0);
    if (pos < BIN_COUNT) {
        long sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];

        d_PartialSuperaccs[get_group_id(0) * BIN_COUNT + pos] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (pos == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[get_group_id(0) * BIN_COUNT], &imin, &imax);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void ExSUMComplete(
    //__global long *d_Superacc,
    __global double *d_Res,
    __global long *d_PartialSuperaccs,
    uint PartialSuperaccusCount
) {
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);

#if 0
    __local long l_Data[MERGE_WORKGROUP_SIZE];

    //Reduce to one work group
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
        long sum = 0;

        for(uint i = 0; i < MERGE_SUPERACCS_SIZE; i++)
            sum += d_PartialSuperaccs[(gid * MERGE_SUPERACCS_SIZE + i) * BIN_COUNT + lid];

        d_PartialSuperaccs[gid * BIN_COUNT + lid] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[gid * BIN_COUNT], &imin, &imax);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if ((lid < BIN_COUNT) && (gid == 0)) {
        long sum = 0;

        for(uint i = 0; i < get_global_size(0) / get_local_size(0); i++)
            sum += d_PartialSuperaccs[i * BIN_COUNT + lid];

        d_PartialSuperaccs[lid] = sum;

        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0)
            d_Res[0] = Round(d_PartialSuperaccs);
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

