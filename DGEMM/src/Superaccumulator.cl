
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
    long z = y + x; // since the value sa->accumulator[i] can be changed by another work item

    // TODO: cover also underflow
    *of = 0;
    if(x > 0 && y > 0 && z < 0)
        *of = 1;
    if(x < 0 && y < 0 && z > 0)
        *of = 1;

    return y;
}

long xaddReduce(long *sa, long x, uchar *of) {
    long y = *sa;
    *sa = y + x;

    // TODO: cover also underflow
    *of = 0;
    if(x > 0 && y > 0 && *sa < 0)
        *of = 1;
    if(x < 0 && y < 0 && *sa > 0)
        *of = 1;

    return y;
}

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial accumulators
////////////////////////////////////////////////////////////////////////////////
void AccumulateWordReduceLocal(long *sa, int i, long x) {//, __global volatile int *d_Overflow) {
  // With atomic accumulator updates
  // accumulation and carry propagation can happen in any order,
  // as long as addition is atomic
  // only constraint is: never forget an overflow bit
  long carry = x;
  long carrybit;
  uchar overflow;
  long oldword = xaddReduce(&sa[i], x, &overflow);

  // To propagate over- or underflow 
  while (overflow) {
    //atomic_inc(d_Overflow); 
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
    xaddReduce(&sa[i], (long) -(carry << digits), &overflow);
    if (TSAFE && (s ^ overflow)) {
      carrybit *= 2;
    }
    carry += carrybit;

    ++i;
    if (i > 1) {
      return;
    }
    oldword = xaddReduce(&sa[i], carry, &overflow);
  }
}

void AccumulateWordReduceGlobal(__local volatile long *sa, int i, long x) {//, __global volatile int *d_Overflow) {
  // With atomic accumulator updates
  // accumulation and carry propagation can happen in any order,
  // as long as addition is atomic
  // only constraint is: never forget an overflow bit
  long carry = x;
  long carrybit;
  uchar overflow;
  long oldword = xadd(&sa[i], x, &overflow);

  // To propagate over- or underflow 
  while (overflow) {
    //atomic_inc(d_Overflow); 
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

void AccumulateWordReduceLocalNew(__local volatile long *sa, int i, long x) {
    long oldword = sa[i];
    long newword = oldword + x;
    
    while(!(newword < 0 && oldword > 0 && x > 0) || !(newword > 0 && oldword < 0 && x < 0)) {
        // Carry or borrow: sign !S
        long carry = newword >> digits;    // Arithmetic shift
        
        // Cancel carry-save bits: sign S
        sa[i] = newword - (carry << digits);
        
        // Add carry bit: sign S
        carry += (newword < 0 ? 1l << K : -1l << K);

        ++i;
        if(i >= BIN_COUNT) {
            return;
        }
        
        oldword = sa[i];
        newword = oldword + carry;
    }
    
    sa[i] = newword;
}

void AccumulateWord(__local volatile long *sa, int i, long x) {//, __global volatile int *d_Overflow) {
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
    //atomic_inc(d_Overflow); 
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

void AccumulateWordNew(__local volatile long *sa, int i, long x) {
    long oldword = sa[i * WARP_COUNT];
    long newword = oldword + x;
    
    while(!(newword < 0 && oldword > 0 && x > 0) || !(newword > 0 && oldword < 0 && x < 0)) {
        // Carry or borrow: sign !S
        long carry = newword >> digits;    // Arithmetic shift
        
        // Cancel carry-save bits: sign S
        sa[i * WARP_COUNT] = newword - (carry << digits);
        
        // Add carry bit: sign S
        carry += (newword < 0 ? 1l << K : -1l << K);

        ++i;
        if(i >= BIN_COUNT) {
            return;
        }
        
        oldword = sa[i * WARP_COUNT];
        newword = oldword + carry;
    }
    
    sa[i * WARP_COUNT] = newword;
}

void Accumulate(__local volatile long *sa, double x) {//, __global volatile int *d_Overflow) {
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

    AccumulateWord(sa, i, xint);//, d_Overflow);

    xscaled -= xrounded;
    xscaled *= deltaScale;
  }
}

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void Superaccumulator(
    __global long *d_PartialKulischAccumulators,
    __global data_t *d_Data,
    //__global int *d_Overflow,
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
    //if (get_local_id(0) == 0)
    //    *d_Overflow = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    //Read data from global memory and scatter it to sub-accumulators
    for(uint pos = get_global_id(0); pos < NbElements; pos += get_global_size(0)){
        data_t x = d_Data[pos];
	Accumulate(l_workingBase, x.x);//, d_Overflow);
	Accumulate(l_workingBase, x.y);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Merge sub-accumulators into work-group partial-accumulator
    uint pos = get_local_id(0);
    if (pos < BIN_COUNT){
        //long sum[2];
	//sum[0] = 0;
	//sum[1] = 0;
        //d_PartialKulischAccumulators[get_group_id(0) * BIN_COUNT + pos] = 0;
        long sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++){
            sum += l_sa[pos * WARP_COUNT + i];
	    //AccumulateWordReduceLocal(sum, 0, l_sa[pos * WARP_COUNT + i]);//, d_Overflow);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
	
        d_PartialKulischAccumulators[get_group_id(0) * BIN_COUNT + pos] = sum;
        //atom_add((__global volatile long *) &d_PartialKulischAccumulators[get_group_id(0) * BIN_COUNT + pos], sum[0]);
	//if (pos < BIN_COUNT)
        //    atom_add((__global volatile long *) &d_PartialKulischAccumulators[get_group_id(0) * BIN_COUNT + pos + 1], sum[1]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge SuperAccumulators
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void mergeSuperaccumulators(
    __global long *d_KulischAccumulator,
    __global long *d_PartialKulischAccumulators,
    //__global int *d_Overflow,
    uint KulischAccumulatorCount
){
    __local long l_Data[MERGE_WORKGROUP_SIZE];

    //Reduce to one work group
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);
    //if (lid == 0)
    //    *d_Overflow = 0;

    long sum = 0;
    //l_Data[lid] = 0;
    for(uint i = lid; i < KulischAccumulatorCount; i += MERGE_WORKGROUP_SIZE)
        sum += d_PartialKulischAccumulators[gid + i * BIN_COUNT];
	//AccumulateWordReduceGlobal(l_Data, i, d_PartialKulischAccumulators[gid + i * BIN_COUNT]);//, d_Overflow);
    l_Data[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reduce within the work group
    for(uint stride = MERGE_WORKGROUP_SIZE / 2; stride > 0; stride >>= 1){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < stride)
            l_Data[lid] += l_Data[lid + stride];
	    //AccumulateWordReduceGlobal(l_Data, lid, l_Data[lid + stride]);//, d_Overflow);
    }
    
    if(lid == 0)
        d_KulischAccumulator[gid] = l_Data[0];
}
