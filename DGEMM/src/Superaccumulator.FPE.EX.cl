
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  // For double precision numbers
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  // For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

//Data type used for input data fetches
typedef double2 data_t;

#define BIN_COUNT      39
#define K              8                    // High-radix carry-save bits
#define digits         56
#define deltaScale     72057594037927936.0  // Assumes K>0
#define f_words        20 
#define TSAFE          0
#define EARLY_EXIT     1
#define WORKGROUP_SIZE (WARP_COUNT * WARP_SIZE)


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
#ifdef USE_KNUTH
    double Knuth2Sum(double a, double b, double *s) {
        double r = a + b;
        double z = r - a;
        *s = (a - (r - z)) + (b - z);
        return r;
    }
#else
    //twosum
    double Knuth2Sum(double a, double b, double *s) {
        double r = a + b;
        int doswap = fabs(b) > fabs(a);
        double a2 = doswap ? b : a;
        double b2 = doswap ? a : b;
        *s = (a2 - r) + b2;
        return r;
    }
#endif

/*//twosum -- does not work well -- the performance is worse than of the upper variant
double Knuth2Sum(double a, double b, double *s) {
    double r = a + b;
    int doswap = fabs(b) > fabs(a);
    if(doswap) {
        // Fast path
        *s = (a - r) + b;
    } else {
        // Slow path
        double a2 = b;
        double b2 = a;
        *s = (a2 - r) + b2;
    }
    return r;
}*/

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
    long xint = rint(xscaled);

    AccumulateWord(sa, i, xint);
    //For a moment we do not care about overflow, we just sum up
    //atom_add((__local volatile long *) &sa[i * WARP_COUNT], xint);

    xscaled -= xrounded;
    xscaled *= deltaScale;
  }
}

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void Superaccumulator(
    __global long *d_PartialKulischAccumulators,
    __global data_t *d_Data,
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
    double a[NBFPE] = {0};
    for(uint pos = get_global_id(0); pos < NbElements; pos += get_global_size(0)){
        data_t x = d_Data[pos];
        double s;
        a[0] = Knuth2Sum(a[0], x.x, &s);
        x.x = s;
	if (x.x != 0.0) {
            a[1] = Knuth2Sum(a[1], x.x, &s);
            x.x = s;
	    if (x.x != 0.0) {
                a[2] = Knuth2Sum(a[2], x.x, &s);
                x.x = s;
	        if (x.x != 0.0) {
                    a[3] = Knuth2Sum(a[3], x.x, &s);
                    x.x = s;
	            if (x.x != 0.0) {
                        a[4] = Knuth2Sum(a[4], x.x, &s);
                        x.x = s;
	                if (x.x != 0.0) {
                            a[5] = Knuth2Sum(a[5], x.x, &s);
                            x.x = s;
	                    if (x.x != 0.0) {
                                a[6] = Knuth2Sum(a[6], x.x, &s);
                                x.x = s;
	                        if (x.x != 0.0) {
                                    a[7] = Knuth2Sum(a[7], x.x, &s);
                                    x.x = s;
       	                        }
     	                    }
   	                }
   	            }
 	        }
	    }
        }
        if(x.x != 0.0)
	    Accumulate(l_workingBase, x.x);

        a[0] = Knuth2Sum(a[0], x.y, &s);
        x.y = s;
	if (x.y != 0.0) {
            a[1] = Knuth2Sum(a[1], x.y, &s);
            x.y = s;
	    if (x.y != 0.0) {
                a[2] = Knuth2Sum(a[2], x.y, &s);
                x.y = s;
	        if (x.y != 0.0) {
                    a[3] = Knuth2Sum(a[3], x.y, &s);
                    x.y = s;
	            if (x.y != 0.0) {
                        a[4] = Knuth2Sum(a[4], x.y, &s);
                        x.y = s;
	                if (x.y != 0.0) {
                            a[5] = Knuth2Sum(a[5], x.y, &s);
                            x.y = s;
	                    if (x.y != 0.0) {
                                a[6] = Knuth2Sum(a[6], x.y, &s);
                                x.y = s;
	                        if (x.y != 0.0) {
                                    a[7] = Knuth2Sum(a[7], x.y, &s);
                                    x.y = s;
       	                        }
     	                    }
   	                }
   	            }
 	        }
	    }
        }
        if(x.y != 0.0)
	    Accumulate(l_workingBase, x.y);
    }
    //Flush to the accumulator
    Accumulate(l_workingBase, a[0]);
    Accumulate(l_workingBase, a[1]);
    Accumulate(l_workingBase, a[2]);
    Accumulate(l_workingBase, a[3]);
    Accumulate(l_workingBase, a[4]);
    Accumulate(l_workingBase, a[5]);
    Accumulate(l_workingBase, a[6]);
    Accumulate(l_workingBase, a[7]);
    barrier(CLK_LOCAL_MEM_FENCE);

    //Merge sub-accumulators into work-group partial-accumulator
    uint pos = get_local_id(0);
    if (pos < BIN_COUNT) {
	long sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++){
            sum += l_sa[pos * WARP_COUNT + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
	
        d_PartialKulischAccumulators[get_group_id(0) * BIN_COUNT + pos] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge SuperAccumulators
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void mergeSuperaccumulators(
    __global long *d_KulischAccumulator,
    __global long *d_PartialKulischAccumulators,
    uint KulischAccumulatorCount
){
    __local long l_Data[MERGE_WORKGROUP_SIZE];

    //Reduce to one work group
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);

    long sum = 0;
    for(uint i = lid; i < KulischAccumulatorCount; i += MERGE_WORKGROUP_SIZE)
        sum += d_PartialKulischAccumulators[gid + i * BIN_COUNT];
    l_Data[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reduce within the work group
    for(uint stride = MERGE_WORKGROUP_SIZE / 2; stride > 0; stride >>= 1){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < stride)
            l_Data[lid] += l_Data[lid + stride];
    }
    
    if(lid == 0)
        d_KulischAccumulator[gid] = l_Data[0];
}
