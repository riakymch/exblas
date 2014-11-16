
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  //For double precision numbers
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial reductions
////////////////////////////////////////////////////////////////////////////////
double dblkSolver(
    __local double *a,
    const uint isunit,
    const int lda,
    double val
){
    volatile __local double xs;
    uint lidx = get_local_id(0);

    #ifdef NVIDIA
       #pragma unroll
    #endif
    for (uint i = 0; i < BLOCK_SIZE; i++) {
        if (lidx == i) {
            if (!isunit)
                val *= a[i * (lda + 1)];
            xs = val;
        }
        if (lidx > i)
            val += a[i * lda + lidx] * xs;
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
    __global double *d_x,
    __global double *d_a,
    __global double *d_b,
    __global int *sync,
    const uint n
){
    __local double cache[BLOCK_SIZE * BLOCK_SIZE];
    __local double partSum[threadsy * threadsx];
    //__local double xlocal[BLOCK_SIZE];
    //double regcache[threadsx / threadsy];

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);
    int tid  = threadsx * lidy + lidx;
    int isunit = 0;

    // Get row handled by this block
    int row = nextRow(&sync[1]);

    /*if(row != 0)
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for(int j = 0; j < BLOCK_SIZE; j += threadsy)
            regcache[j / threadsy] = d_a[((row - 1) * BLOCK_SIZE + lidy) * n + row * BLOCK_SIZE + lidx + j * n];
    */
    // Copy diagonal block to shared memory
    tocache(&d_a[row * BLOCK_SIZE * n + row * BLOCK_SIZE], cache, BLOCK_SIZE, threadsx * threadsy, 0, isunit, tid, n);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over blocks as they become available
    double val = 0.0;
    if(lidy == 0)
        val = d_b[row * BLOCK_SIZE + lidx];
    int col_done = -1;

    for (int col = 0; col < row; col++) {
        wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
        //__local double *xl = xlocal + lidy;
        //xlocal[tid] = d_x[col * BLOCK_SIZE + tid];
        //barrier(CLK_LOCAL_MEM_FENCE);
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (int j = 0; j < BLOCK_SIZE; j+=threadsy)
            val -= d_a[(col * BLOCK_SIZE + lidy) * n + row * BLOCK_SIZE + lidx + j * n] * d_x[col * BLOCK_SIZE + lidy + j];
            //val += d_a[(col * BLOCK_SIZE + lidy) * n + row * BLOCK_SIZE + lidx + j * n] * xl[j];
    }
    /*if (row != 0) {
        const int col = row - 1;
        wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
        //__local double *xl = xlocal + lidy;
        //xlocal[tid] = d_x[col * BLOCK_SIZE + tid];
        //barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j += threadsy)
            val += regcache[j / threadsy] * d_x[col * BLOCK_SIZE + lidy + j];
            //val += regcache[j / threadsy] * xl[j];
    }*/
    partSum[lidy * threadsx + lidx] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Apply update from diagonal block (row, row)
    if (lidy == 0) {
        for(int i = 1; i < threadsy; i++)
            val += partSum[i * threadsx + lidx];
        barrier(CLK_LOCAL_MEM_FENCE);

        //val = dblkSolver(cache, isunit, BLOCK_SIZE, val);
        volatile __local double xs;
        #ifdef NVIDIA
           #pragma unroll
        #endif
        for (uint i = 0; i < BLOCK_SIZE; i++) {
            if (lidx == i) {
                if (!isunit)
                    val *= cache[i * (BLOCK_SIZE + 1)];
                xs = val;
            }
            if (lidx > i)
                val += cache[i * BLOCK_SIZE + lidx] * xs;
        }
        d_x[row * BLOCK_SIZE + tid] = val;
    }

    // Notify other blocks that soln is ready for this row
    barrier(CLK_GLOBAL_MEM_FENCE); // Wait for d_x to be visible to other blocks
    if(tid == 0)
        atomic_add(&sync[0], 1);   // Use atomicAdd to bypass L1 miss
    barrier(CLK_GLOBAL_MEM_FENCE); // Flush sync[0] asap
}

/*__kernel void trsv_lnn_bak(
    __global double *d_x,
    __global double *d_a,
    __global double *d_b,
    __global int *sync,
    const uint n
){
    int nblk = n / threadsx;
    double __local cache[threadsx * threadsx];
    double regcache[threadsx / threadsy];
    double __local partSum[threadsy * threadsx];
    //double __local xlocal[threadsx];

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);
    int tid = threadsx * lidy + lidx;

    // Get row handled by this block
    int row = nextRow(&sync[1]);

    if(row != 0)
        #pragma unroll
        for(int j = 0; j < threadsx; j += threadsy)
            regcache[j / threadsy] = d_a[((row - 1) * threadsx + lidy) * n + row * threadsx + lidx + j * n];

    // Copy diagonal block to shared memory
    tocache(&d_a[row * threadsx * n + row * threadsx], 0, 0, n, BLOCK_SIZE, tid, threadsx * threadsy, cache);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Loop over blocks as they become available
    double val = 0.0;
    if(lidy == 0)
        val = -d_b[row * threadsx + lidx];
    int col_done = -1;

    for (int col = 0; col < row - 1; col++) {
        wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
        for (int j = 0; j < threadsx; j += threadsy)
            val -= d_a[(col * threadsx + lidy) * n + row * n + lidx + j * n] * d_x[col * threadsx + j];
    }
    if (row != 0) {
        const int col = row - 1;
        wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
        for (int j = 0; j < threadsx; j += threadsy)
            val += regcache[j / threadsy] * d_x[col * threadsx + j];
    }
    partSum[tid] = val;
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Apply update from diagonal block (row, row)
    if (lidy == 0) {
        for(int i = 1; i < threadsx; i++)
            val += partSum[i * threadsx + lidx];
        d_x[row * threadsx + tid] = dblkSolver(cache, threadsx, val);
    }

    // Notify other blocks that soln is ready for this row
    barrier(CLK_GLOBAL_MEM_FENCE); // Wait for d_x to be visible to other blocks
    if(tid==0)
        atomic_add(&sync[0], 1); // Use atomicAdd to bypass L1 miss
    barrier(CLK_GLOBAL_MEM_FENCE); // Flush sync[0] asap
}*/
