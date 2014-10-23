
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  //For double precision numbers
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial reductions
////////////////////////////////////////////////////////////////////////////////
__kernel void TRSVLNU(
    __global double *d_x,
    __global double *d_a,
    const uint n
){
    uint tidx = get_local_id(0);

    double s = d_x[tidx];
    volatile __local double xs;

    #ifdef NVIDIA
       #pragma unroll
    #endif
    for (uint i = 0; i < BLOCK_SIZE; i++) {
        if (tidx == i)
            xs = s;
        if (tidx >= (i + 1))
            s -= d_a[tidx * n + i] * xs;
    }

    d_x[tidx] = s;
}

