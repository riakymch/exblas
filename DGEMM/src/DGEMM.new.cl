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
}

__kernel void matrixMulKernel (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k
) {
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int bx = get_group_id(0);
    const int by = get_group_id(1);

    //Load Asub and Bsub from device memory to shared memory
    A += bx * m + id;
    B += inx + (iby + iny) * k;
    C += ibx + id + iby * m;

    data_t c[64] = {0.0};

    __local data_t bs[16][17];

    for (int j = 0; j < k; j += 16, B += 16) {
        #pragma unroll
        for (int i = 0; i < 16; i += 4)
            bs[inx][iny + i] = B[i * m];
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int i = 0; i < 16; i++, A += m)
            saxpy(A[0], &bs[i][0], c);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < 16; i++, C += m)
        C[0] = c[i];

    /*const int inx = get_local_id(0);
    const int iny = get_local_id(1);
    const int ibx = get_group_id(0) * 64;
    const int iby = get_group_id(1) * 16;
    const int id = inx + iny * 16;

    //Load Asub and Bsub from device memory to shared memory
    A += ibx + id;
    B += inx + (iby + iny) * k;
    C += ibx + id + iby * m;

    data_t c[64] = {0.0};

    __local data_t bs[16][17];

    for (int j = 0; j < k; j += 16, B += 16) {
        #pragma unroll
        for (int i = 0; i < 16; i += 4)
            bs[inx][iny + i] = B[i * m];
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int i = 0; i < 16; i++, A += m)
            saxpy(A[0], &bs[i][0], c);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < 16; i++, C += m)
        C[0] = c[i];*/
}

