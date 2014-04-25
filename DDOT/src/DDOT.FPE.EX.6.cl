
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
// Main computation pass: compute partial accumulators
////////////////////////////////////////////////////////////////////////////////
void AccumulateWord(__local volatile long *sa, int i, long x) {
  // With atomic accumulator updates
  // accumulation and carry propagation can happen in any order,
  // as long as addition is atomic
  // only constraint is: never forget an overflow bit
  long carry = x;
  long carrybit;
  uchar overflow;
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
    if (TSAFE && (s ^ overflow)) {
      carrybit *= 2;
    }
    carry += carrybit;

    ++i;
    if (i >= BIN_COUNT) {
      return;
    }
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
    //TODO: optimize
    if (get_local_id(0) < WARP_COUNT) {
        for (uint i = 0; i < BIN_COUNT; i++)
           l_workingBase[i * WARP_COUNT] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Read data from global memory and scatter it to sub-accumulators
    double a[6] = {0.0};
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
                    if(x != 0.0) {
                        a[4] = KnuthTwoSum(a[4], x, &s);
                        x = s;
                        if(x != 0.0) {
                            a[5] = KnuthTwoSum(a[5], x, &s);
                            x = s;
	                }
	            }
	        }
	    }
	}
        if(x != 0.0) 
	    Accumulate(l_workingBase, x);

	//if (r != 0.0) { // without it is better for the performance, especially on nvidia
            //double s;
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
                        if(r != 0.0) {
                            a[4] = KnuthTwoSum(a[4], r, &s);
                            r = s;
                            if(r != 0.0) {
                                a[5] = KnuthTwoSum(a[5], r, &s);
                                r = s;
   	                    }
   	                }
	            }
   	        }
            }
            if(r != 0.0)
	        Accumulate(l_workingBase, r);
	//}
    }
    //Flush to the accumulator
    Accumulate(l_workingBase, a[0]);
    Accumulate(l_workingBase, a[1]);
    Accumulate(l_workingBase, a[2]);
    Accumulate(l_workingBase, a[3]);
    Accumulate(l_workingBase, a[4]);
    Accumulate(l_workingBase, a[5]);
    barrier(CLK_LOCAL_MEM_FENCE);

    //Merge sub-accumulators into work-group partial-accumulator
    uint pos = get_local_id(0);
    if (pos < BIN_COUNT){
        long sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++){
            sum += l_sa[pos * WARP_COUNT + i];
        }
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
    for(uint i = lid; i < NbPartialSuperaccs; i += MERGE_WORKGROUP_SIZE) {
        sum += d_PartialSuperaccs[gid + i * BIN_COUNT];
    }
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
