
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  //For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
    #pragma OPENCL EXTENSION cl_khr_fp64               : enable  //For double precision numbers
    #pragma OPENCL EXTENSION cl_nv_pragma_unroll       : enable
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
        // Only read global memory when necessary
        if (*col_done < col_to_wait) {
            while(*sync < col_to_wait) {}
            *col_done = *sync;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/* Returns next block row index that requires processing */
void nextRow(
   __local volatile int *old,
   __global volatile int *address
){
	if(get_local_id(0)==0 && get_local_id(1)==0) {
    	*old = atomic_inc(address);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
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
        }
    }
}


/*
 * Sets sync values correctly prior to call to trsv_ln_exec
 */
__kernel void trsv_init(
    __global int *sync
){
   sync[0] = -1; // Last ready column
   sync[1] = 0;  // Next row to assign
}


__kernel void dtrsv(
    const uint n,
    __global double *d_a,
    const uint lda,
    const uint offseta,
    __global double *d_x,
    const uint incx,
    const uint offsetx,
    __global volatile int *sync,
    __local double *cache,
    __local int *row,
    __local volatile double *xs
){
    int lidx = get_local_id(0);
    int lidy = get_local_id(1);
    int tid  = threadsx * lidy + lidx;
    int isunit = 0;
    int ntid = threadsx * threadsy;

    // Get row handled by this block
    nextRow(row, &sync[1]);

    // Copy diagonal block to shared memory
    tocache(&d_a[*row * BLOCK_SIZE * lda + *row * BLOCK_SIZE], cache, BLOCK_SIZE, ntid, 0, isunit, tid, lda);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over blocks as they become available
    double val = 0.0;
    int col_done = -1;

    double x, r;
    for (int col = 0; col < *row; col++) {
        wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (int j = 0; j < BLOCK_SIZE; j+=threadsy) {
            val -= d_x[col * BLOCK_SIZE + lidy + j] * d_a[(col * BLOCK_SIZE + lidy) * lda + *row * BLOCK_SIZE + lidx + j * lda];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Apply update from diagonal block (row, row)
    if (lidy == 0) {
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (uint i = 0; i < BLOCK_SIZE; i++) {
            if (lidx == i) {
                val += d_x[*row * threadsx + lidx];

                if (!isunit)
                    val = val / cache[i * (BLOCK_SIZE + 1)];
                *xs = val;
            }
            if (lidx > i) {
                val += *xs * cache[i * BLOCK_SIZE + lidx];
            }
        }
        d_x[*row * BLOCK_SIZE + tid] = val;
    }

    // Notify other blocks that soln is ready for this row
    barrier(CLK_GLOBAL_MEM_FENCE); // Wait for d_x to be visible to other blocks
    if(tid == 0)
        atomic_inc(&sync[0]);   // Use atomicAdd to bypass L1 miss
    barrier(CLK_GLOBAL_MEM_FENCE); // Flush sync[0] asap
}


