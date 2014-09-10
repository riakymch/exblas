
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_khr_fp64                 : enable  // For double precision numbers
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

//Data type used for input data fetches
typedef double data_t;

void DGEMM (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k,
    __local data_t *As,
    __local data_t *Bs,
    int bx,
    int by,
    int tx,
    int ty
) {
    //Each thread block computes one sub-matrix of C
    __global data_t* Csub = &C[n * BLOCK_SIZE * by + BLOCK_SIZE * bx];

    //Each thread computes one element of Csub
    data_t sum[4] = {0.0};

    //step
    int step = 4;
    #ifdef NVIDIA
      step = step * 2;
    #endif

    //Loop over all sub-matrices of A and B to compute Csub
    for (int i = 0; i < (m / BLOCK_SIZE); ++i) {
        //Sub-matrix Asub of A and Bsub of B
        __global data_t* Asub = &A[m * BLOCK_SIZE * by + BLOCK_SIZE * i];
        __global data_t* Bsub = &B[k * BLOCK_SIZE * i + BLOCK_SIZE * bx];

        //Load Asub and Bsub from device memory to shared memory
        As[ty * BLOCK_SIZE + tx] = Asub[ty * m + tx];
        Bs[ty * BLOCK_SIZE + tx] = Bsub[ty * k + tx];
        As[(ty + step) * BLOCK_SIZE + tx] = Asub[(ty + step) * m + tx];
        Bs[(ty + step) * BLOCK_SIZE + tx] = Bsub[(ty + step) * k + tx];
        As[(ty + 2 * step) * BLOCK_SIZE + tx] = Asub[(ty + 2 * step) * m + tx];
        Bs[(ty + 2 * step) * BLOCK_SIZE + tx] = Bsub[(ty + 2 * step) * k + tx];
        As[(ty + 3 * step) * BLOCK_SIZE + tx] = Asub[(ty + 3 * step) * m + tx];
        Bs[(ty + 3 * step) * BLOCK_SIZE + tx] = Bsub[(ty + 3 * step) * k + tx];

        //Synchronize to make sure that the sub-matrices are loaded before the computation starts
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiply Asub and Bsub
        #ifdef NVIDIA
           #pragma unroll
        #endif
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum[0] = fma(As[ty * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx], sum[0]);
            sum[1] = fma(As[(ty + step) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx], sum[1]);
            sum[2] = fma(As[(ty + 2 * step) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx], sum[2]);
            sum[3] = fma(As[(ty + 3 * step) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx], sum[3]);
        }
    }
    Csub[ty * n + tx] = sum[0];
    Csub[(ty + step) * n + tx] = sum[1];
    Csub[(ty + 2 * step) * n + tx] = sum[2];
    Csub[(ty + 3 * step) * n + tx] = sum[3];
}

__kernel void matrixMul(
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k,
    __local data_t* As,
    __local data_t* Bs
) {
    //Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    //Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    /*int bdimx = n / BLOCK_SIZE;
    int bdimy = m / BLOCK_SIZE;
    int bsizex = get_num_groups(0);
    int bsizey = get_num_groups(1);

    for (int i = bx; i < bdimx; i += bsizex)
        for (int j = by; j < bdimy; j += bsizey)
            DGEMM_1(C, A, B, m, n, k, As, Bs, i, j, tx, ty);*/

    //Each thread block computes one sub-matrix of C
    __global data_t* Csub = &C[n * BLOCK_SIZE * by + BLOCK_SIZE * bx];

    //Each thread computes one element of Csub
    data_t sum[4] = {0.0};

    //step
    int step = 4;
    #ifdef NVIDIA
      step = step * 2;
    #endif

    //Loop over all sub-matrices of A and B to compute Csub
    for (int i = 0; i < (k / BLOCK_SIZE); ++i) {
        //Sub-matrix Asub of A and Bsub of B
        __global data_t* Asub = &A[k * BLOCK_SIZE * by + BLOCK_SIZE * i];
        __global data_t* Bsub = &B[n * BLOCK_SIZE * i + BLOCK_SIZE * bx];

        //Load Asub and Bsub from device memory to shared memory
        As[ty * BLOCK_SIZE + tx] = Asub[ty * k + tx];
        Bs[ty * BLOCK_SIZE + tx] = Bsub[ty * n + tx];
        As[(ty + step) * BLOCK_SIZE + tx] = Asub[(ty + step) * m + tx];
        Bs[(ty + step) * BLOCK_SIZE + tx] = Bsub[(ty + step) * k + tx];
        As[(ty + 2 * step) * BLOCK_SIZE + tx] = Asub[(ty + 2 * step) * m + tx];
        Bs[(ty + 2 * step) * BLOCK_SIZE + tx] = Bsub[(ty + 2 * step) * k + tx];
        As[(ty + 3 * step) * BLOCK_SIZE + tx] = Asub[(ty + 3 * step) * m + tx];
        Bs[(ty + 3 * step) * BLOCK_SIZE + tx] = Bsub[(ty + 3 * step) * k + tx];
        /*As[(ty + 4 * step) * BLOCK_SIZE + tx] = Asub[(ty + 4 * step) * m + tx];
        Bs[(ty + 4 * step) * BLOCK_SIZE + tx] = Bsub[(ty + 4 * step) * k + tx];
        As[(ty + 5 * step) * BLOCK_SIZE + tx] = Asub[(ty + 5 * step) * m + tx];
        Bs[(ty + 5 * step) * BLOCK_SIZE + tx] = Bsub[(ty + 5 * step) * k + tx];
        As[(ty + 6 * step) * BLOCK_SIZE + tx] = Asub[(ty + 6 * step) * m + tx];
        Bs[(ty + 6 * step) * BLOCK_SIZE + tx] = Bsub[(ty + 6 * step) * k + tx];
        As[(ty + 7 * step) * BLOCK_SIZE + tx] = Asub[(ty + 7 * step) * m + tx];
        Bs[(ty + 7 * step) * BLOCK_SIZE + tx] = Bsub[(ty + 7 * step) * k + tx];*/

        //Synchronize to make sure that the sub-matrices are loaded before the computation starts
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiply Asub and Bsub
        #ifdef NVIDIA
           #pragma unroll
        #endif
        for (int l = 0; l < BLOCK_SIZE; ++l) {
            sum[0] = fma(As[ty * BLOCK_SIZE + l], Bs[l * BLOCK_SIZE + tx], sum[0]);
            sum[1] = fma(As[(ty + step) * BLOCK_SIZE + l], Bs[l * BLOCK_SIZE + tx], sum[1]);
            sum[2] = fma(As[(ty + 2 * step) * BLOCK_SIZE + l], Bs[l * BLOCK_SIZE + tx], sum[2]);
            sum[3] = fma(As[(ty + 3 * step) * BLOCK_SIZE + l], Bs[l * BLOCK_SIZE + tx], sum[3]);
            /*sum[4] = fma(As[(ty + 4 * step) * BLOCK_SIZE + l], Bs[l * BLOCK_SIZE + tx], sum[4]);
            sum[5] = fma(As[(ty + 5 * step) * BLOCK_SIZE + l], Bs[l * BLOCK_SIZE + tx], sum[5]);
            sum[6] = fma(As[(ty + 6 * step) * BLOCK_SIZE + l], Bs[l * BLOCK_SIZE + tx], sum[6]);
            sum[7] = fma(As[(ty + 7 * step) * BLOCK_SIZE + l], Bs[l * BLOCK_SIZE + tx], sum[7]);*/
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    Csub[ty * n + tx] = sum[0];
    Csub[(ty + step) * n + tx] = sum[1];
    Csub[(ty + 2 * step) * n + tx] = sum[2];
    Csub[(ty + 3 * step) * n + tx] = sum[3];
    /*Csub[(ty + 4 * step) * n + tx] = sum[4];
    Csub[(ty + 5 * step) * n + tx] = sum[5];
    Csub[(ty + 6 * step) * n + tx] = sum[6];
    Csub[(ty + 7 * step) * n + tx] = sum[7];*/
}

