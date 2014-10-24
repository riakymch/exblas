
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  //For double precision numbers
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial reductions
////////////////////////////////////////////////////////////////////////////////
double dblkSolver(__local double *a, int lda, double val){
    volatile __local double xs;
    uint lidx = get_local_id(0);

    #ifdef NVIDIA
       #pragma unroll
    #endif
    for (uint j = 0; j < BLOCK_SIZE; j++) {
        if (lidx == j)
            xs = val;
        if (lidx >= (j+1)) {
            val -= a[j * lda + lidx] * xs;
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
    //double __local x_local[BLOCK_SIZE];

    int lidx = get_local_id(0);

    for (int i = 0; i < (n / BLOCK_SIZE); i++) {
        // gemv and copying the data to local memory
        /*if (1) {
            // load the matrix and a vector
            double __local a_local[BLOCK_SIZE * BLOCK_SIZE];
            a_local[]
            // compute
            // store back
        }*/

        // solve diagonal block by the first workgroup
        if (get_global_id(0) == 0) {
            double __local a_local[32 * 32];
            double __local a_local = &d_a[i * n * BLOCK_SIZE + i * BLOCK_SIZE + lidx * n + get_local_id(1)];
            double val = d_b[i * BLOCK_SIZE + lidx];
            d_x[i * BLOCK_SIZE + lidx] = dblkSolver(a_local, BLOCK_SIZE, val);
        }
    }
}

