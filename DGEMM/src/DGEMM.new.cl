/*
 * Vasily Volkov's code modified for OpenCL
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_khr_fp64                 : enable  // For double precision numbers
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

//Data type used for input data fetches
typedef double data_t;

void saxpy(data_t a, __local data_t *b, data_t *c)
{
    c[0] += a*b[0];
    c[1] += a*b[1];
    c[2] += a*b[2];
    c[3] += a*b[3];
    c[4] += a*b[4];
    c[5] += a*b[5];
    c[6] += a*b[6];
    c[7] += a*b[7];
    c[8] += a*b[8];
    c[9] += a*b[9];
    c[10] += a*b[10];
    c[11] += a*b[11];
    c[12] += a*b[12];
    c[13] += a*b[13];
    c[14] += a*b[14];
    c[15] += a*b[15];

    /*c[0] = fma(a, b[0], c[0]);
    c[1] = fma(a, b[1], c[1]);
    c[2] = fma(a, b[2], c[2]);
    c[3] = fma(a, b[3], c[3]);
    c[4] = fma(a, b[4], c[4]);
    c[5] = fma(a, b[5], c[5]);
    c[6] = fma(a, b[6], c[6]);
    c[7] = fma(a, b[7], c[7]);
    c[8] = fma(a, b[8], c[8]);
    c[9] = fma(a, b[9], c[9]);
    c[10] = fma(a, b[10], c[10]);
    c[11] = fma(a, b[11], c[11]);
    c[12] = fma(a, b[12], c[12]);
    c[13] = fma(a, b[13], c[13]);
    c[14] = fma(a, b[14], c[14]);
    c[15] = fma(a, b[15], c[15]);*/
}

__kernel void matrixMulKernel (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k
) {
    int inx = get_local_id(0);
    int iny = get_local_id(1);
    int ibx = get_group_id(0) * 64;
    int iby = get_group_id(1) * 16;
    int id  = inx + iny * 16;

    //Load Asub and Bsub from device memory to shared memory
    A += ibx + id;
    B += inx + mul24(iby + iny, k);
    C += ibx + id + mul24(iby, m);

    data_t c[16] = {0.0};

    __local data_t bs[16][17];

    for (int j = 0; j < k; j += 16, B += 16) {
        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (int i = 0; i < 16; i += 4) // only for 16x4 threads per block
            bs[inx][iny + i] = B[i * k];
        barrier(CLK_LOCAL_MEM_FENCE);

        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (int i = 0; i < 16; i++, A += m)
            saxpy(A[0], &bs[i][0], c);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < 16; i++, C += m)
        C[0] = c[i];
}

