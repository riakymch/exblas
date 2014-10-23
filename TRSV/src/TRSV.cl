
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
    double val = d_b[get_local_id(0)];
    volatile __local double xs;

    d_x[get_local_id(0)] = dblkSolver(d_a, n, val);
}

