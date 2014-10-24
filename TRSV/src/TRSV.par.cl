
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  //For double precision numbers
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial reductions
////////////////////////////////////////////////////////////////////////////////
double dblkSolver(__global double *a, int lda, double val){
    volatile __local double xs;
    uint tid = get_local_id(0);

    #ifdef NVIDIA
       #pragma unroll
    #endif
    for (uint j = 0; j < BLOCK_SIZE; j++) {
        if (tid == j)
            xs = val;
        if (tid >= (j+1)) {
            val -= a[j * lda + tid] * xs;
        }
    }

    return val;
}

__kernel void TRSVLNU(
    __global double *d_x,
    __global double *d_a,
    __global double *d_b,
    const uint n
){
    double __local a_rect[(n - BLOCK_SIZE) * BLOCK_SIZE];
    double __local x_local[n];
    double __local a_trsv[BLOCK_SIZE * BLOCK_SIZE];

    int tid = (n - BLOCK_SIZE) * get_local_id(1) + get_local_id(0);
    if (tid < n)
        x_local[tid] = d_b[tid];

    for (int i = 0; i < (n / BLOCK_SIZE); i++) {
        if (get_global_id(0) == 0)

        double val = d_b[get_local_id(0)];
        d_x[get_local_id(0)] = dblkSolver(d_a, n, val);
    }
}

