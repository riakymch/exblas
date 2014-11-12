
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

#ifdef USE_KNUTH
    double KnuthTwoSum(double a, double b, double *s) {
        double r = a + b;
        double z = r - a;
        *s = (a - (r - z)) + (b - z);
        return r;
    }
#else
    //twosum
    double KnuthTwoSum(double a, double b, double *s) {
        double r = a + b;
        int doswap = fabs(b) > fabs(a);
        double a2 = doswap ? b : a;
        double b2 = doswap ? a : b;
        *s = (a2 - r) + b2;
        return r;
    }
#endif


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
                cache[x * (nbi + 1)] = 1.0 / a[x * (lda + 1)];
        }
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
    __local double cache[BLOCK_SIZE * BLOCK_SIZE];

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);
    int tid  = threadsx * lidy + lidx;
    int isunit = 0;

    // Get row handled by this block
    int row = nextRow(&sync[1]);

    // Copy diagonal block to shared memory
    tocache(&d_a[row * BLOCK_SIZE * n + row * BLOCK_SIZE], cache, BLOCK_SIZE, threadsx * threadsy, 0, isunit, tid, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over blocks as they become available
    // Initialize FPEs
    double x, s, r;
    double fpe[NBFPE] = {0.0};
    if(lidy == 0) {
        x = d_b[row * BLOCK_SIZE + lidx];
        #ifdef NVIDIA
          #pragma unroll
        #endif
        for(uint i = 0; i != NBFPE; ++i) {
            fpe[i] = KnuthTwoSum(fpe[i], x, &s);
            x = s;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int col_done = -1;

    for (int col = 0; col < row; col++) {
        wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (int j = 0; j < BLOCK_SIZE; j+=threadsy) {
            r = 0.0;
            double xp = -d_x[col * BLOCK_SIZE + lidy + j];
            x = TwoProductFMA(d_a[(col * BLOCK_SIZE + lidy) * n + row * BLOCK_SIZE + lidx + j * n], xp, &r);

            #ifdef NVIDIA
                #pragma unroll
            #endif
            for(uint i = 0; i != NBFPE; ++i) {
                fpe[i] = KnuthTwoSum(fpe[i], x, &s);
                x = s;
            }

            #ifdef NVIDIA
                #pragma unroll
            #endif
            for(uint i = 0; i != NBFPE; ++i) {
                s = 0.0;
                fpe[i] = KnuthTwoSum(fpe[i], r, &s);
                r = s;
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
                //TODO: round fpes
                val = fpe[0];
                if (!isunit)
                    val *= cache[i * (BLOCK_SIZE + 1)];
                xs = val;
            }
            if (lidx > i) {
                r = 0.0;
                x = TwoProductFMA(cache[i * BLOCK_SIZE + lidx], xs, &r);

                #ifdef NVIDIA
                    #pragma unroll
                #endif
                for(uint i = 0; i != NBFPE; ++i) {
                    s = 0.0;
                    fpe[i] = KnuthTwoSum(fpe[i], x, &s);
                    x = s;
                }

                #ifdef NVIDIA
                    #pragma unroll
                #endif
                for(uint i = 0; i != NBFPE; ++i) {
                    s = 0.0;
                    fpe[i] = KnuthTwoSum(fpe[i], r, &s);
                    r = s;
                }
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

