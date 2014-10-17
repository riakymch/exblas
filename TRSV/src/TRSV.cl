
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  //For double precision numbers
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial reductions
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void TRSV(
    __global double *d_PartialSuperaccs,
    __global double *d_a,
    __global double *d_b,
    const uint NbElements
){
    __local double l_sa[WORKGROUP_SIZE] __attribute__((aligned(8)));
    uint lid = get_local_id(0);

    // Each work-item accumulates as many elements as necessary into local variable "sum"
    double sum = 0.0;
    #ifdef NVIDIA
        #pragma unroll
    #endif
    for(uint gid = get_global_id(0); gid < NbElements; gid += get_global_size(0))
        sum = fma(d_a[gid], d_b[gid], sum);
    l_sa[lid] = sum;

    // Perform parallel reduction to add each work-item's
    // partial summation together
    for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        // Synchronize to make sure each work-item is done updating
        // shared memory; this is necessary because work-items read
        // results written by other work-items during each parallel
        // reduction step
        barrier(CLK_LOCAL_MEM_FENCE);

        // Only the first work-items in the work-group add elements together
        if (get_local_id(0) < stride) {
            // Add two elements from the l_sa array
            // and store the result in l_sa[index]
            l_sa[lid] += l_sa[lid + stride];
        } 
    } 

    // Write the result of the reduction to global memory 
    if (lid == 0)
        d_PartialSuperaccs[get_group_id(0)] = l_sa[0];

    // Synchronize to make sure the first work-item is done reading partialSummation 
    barrier(CLK_LOCAL_MEM_FENCE);
}

