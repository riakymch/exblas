
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
    uint lid = get_local_id(0);

    double s = d_x[lid];
    volatile __local double xs;

    #ifdef NVIDIA
       #pragma unroll
    #endif
    for (uint i = 0; i < BLOCK_SIZE; i++) {
        if (lid == i)
            xs = s;
        if (lid >= (i + 1))
            s -= d_a[i * n + lid] * xs;
    }

    d_x[lid] = s;
}

