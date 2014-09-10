/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  // For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_khr_fp64                 : enable
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

typedef double data_t;

#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]

///////////////////////////////////////////////////////////////////////////////
// Matrix multiplication on the device: C = A * B
// m is A's width and n is B's width
////////////////////////////////////////////////////////////////////////////////
void DGEMM(
    __global data_t* C,
    __global data_t* A,
    __global data_t* B, 
    int m,
    int n,
    __local data_t* As,
    __local data_t* Bs,
    int bx,
    int by,
    int tx,
    int ty
) {
    //Index of the first sub-matrix of A processed by the block
    int aBegin = m * BLOCK_SIZE * by;

    //Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + m - 1;

    //Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * n;

    //sum is used to store the element of the block sub-matrix that is computed by the thread
    data_t sum[4] = {0.0};

    //step
    int step = 4;
    #ifdef NVIDIA
      step = step * 2;
    #endif

    //Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        //Load the matrices from device memory to shared memory; 
        //each thread loads one element of each matrix
        AS(ty, tx) = A[a + ty * m + tx];
        BS(ty, tx) = B[b + ty * n + tx];
        AS(ty + step, tx) = A[a + (ty + step) * m + tx];
        BS(ty + step, tx) = B[b + (ty + step) * n + tx];
        AS(ty + 2 * step, tx) = A[a + (ty + 2 * step) * m + tx];
        BS(ty + 2 * step, tx) = B[b + (ty + 2 * step) * n + tx];
        AS(ty + 3 * step, tx) = A[a + (ty + 3 * step) * m + tx];
        BS(ty + 3 * step, tx) = B[b + (ty + 3 * step) * n + tx];

        //Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiply the two matrices together;
        //each thread computes one element of the block sub-matrix
        #ifdef NVIDIA
          #pragma unroll
        #endif
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum[0] = fma(AS(ty, k), BS(k, tx), sum[0]);
            sum[1] = fma(AS(ty + step, k), BS(k, tx), sum[1]);
            sum[2] = fma(AS(ty + 2 * step, k), BS(k, tx), sum[2]);
            sum[3] = fma(AS(ty + 3 * step, k), BS(k, tx), sum[3]);
        }

        //Synchronize to make sure that the preceding computation is done before
        //loading two new sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int c = (m * by + bx) * BLOCK_SIZE;
    C[c + ty * m + tx] = sum[0];
    C[c + (ty + step) * m + tx] = sum[1];
    C[c + (ty + 2 * step) * m + tx] = sum[2];
    C[c + (ty + 3 * step) * m + tx] = sum[3];
}

__kernel void DGEMM8(
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    __local data_t* As,
    __local data_t* Bs
) {
    //Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    //Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    //Index of the first sub-matrix of A processed by the block
    int aBegin = m * BLOCK_SIZE * by;

    //Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + m - 1;

    //Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * n;

    //sum is used to store the element of the block sub-matrix that is computed by the thread
    data_t sum[8] = {0.0};

    //step
    int step = 2;
    #ifdef NVIDIA
      step = step * 2;
    #endif

    //Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        //Load the matrices from device memory to shared memory; 
        //each thread loads one element of each matrix
        AS(ty, tx) = A[a + ty * m + tx];
        BS(ty, tx) = B[b + ty * n + tx];
        AS(ty + step, tx) = A[a + (ty + step) * m + tx];
        BS(ty + step, tx) = B[b + (ty + step) * n + tx];
        AS(ty + 2 * step, tx) = A[a + (ty + 2 * step) * m + tx];
        BS(ty + 2 * step, tx) = B[b + (ty + 2 * step) * n + tx];
        AS(ty + 3 * step, tx) = A[a + (ty + 3 * step) * m + tx];
        BS(ty + 3 * step, tx) = B[b + (ty + 3 * step) * n + tx];
        AS(ty + 4 * step, tx) = A[a + (ty + 4 * step) * m + tx];
        BS(ty + 4 * step, tx) = B[b + (ty + 4 * step) * n + tx];
        AS(ty + 5 * step, tx) = A[a + (ty + 5 * step) * m + tx];
        BS(ty + 5 * step, tx) = B[b + (ty + 5 * step) * n + tx];
        AS(ty + 6 * step, tx) = A[a + (ty + 6 * step) * m + tx];
        BS(ty + 6 * step, tx) = B[b + (ty + 6 * step) * n + tx];
        AS(ty + 7 * step, tx) = A[a + (ty + 7 * step) * m + tx];
        BS(ty + 7 * step, tx) = B[b + (ty + 7 * step) * n + tx];

        //Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiply the two matrices together;
        //each thread computes one element of the block sub-matrix
        #ifdef NVIDIA
          #pragma unroll
        #endif
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum[0] = fma(AS(ty, k), BS(k, tx), sum[0]);
            sum[1] = fma(AS(ty + step, k), BS(k, tx), sum[1]);
            sum[2] = fma(AS(ty + 2 * step, k), BS(k, tx), sum[2]);
            sum[3] = fma(AS(ty + 3 * step, k), BS(k, tx), sum[3]);
            sum[4] = fma(AS(ty + 4 * step, k), BS(k, tx), sum[4]);
            sum[5] = fma(AS(ty + 5 * step, k), BS(k, tx), sum[5]);
            sum[6] = fma(AS(ty + 6 * step, k), BS(k, tx), sum[6]);
            sum[7] = fma(AS(ty + 7 * step, k), BS(k, tx), sum[7]);
        }

        //Synchronize to make sure that the preceding computation is done before 
        //loading two new sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int c = m * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + ty * m + tx] = sum[0];
    C[c + (ty + step) * m + tx] = sum[1];
    C[c + (ty + 2 * step) * m + tx] = sum[2];
    C[c + (ty + 3 * step) * m + tx] = sum[3];
    C[c + (ty + 4 * step) * m + tx] = sum[4];
    C[c + (ty + 5 * step) * m + tx] = sum[5];
    C[c + (ty + 6 * step) * m + tx] = sum[6];
    C[c + (ty + 7 * step) * m + tx] = sum[7];
}

__kernel void DGEMM2(
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    __local data_t* As,
    __local data_t* Bs
) {
    //Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    //Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    //Index of the first sub-matrix of A processed by the block
    int aBegin = m * BLOCK_SIZE * by;

    //Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + m - 1;

    //Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * n;

    //sum is used to store the element of the block sub-matrix that is computed by the thread
    data_t sum[2] = {0.0};

    //step
    int step = 8;
    #ifdef NVIDIA
      step = step * 2;
    #endif

    //Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        //Load the matrices from device memory to shared memory;
        //each thread loads one element of each matrix
        AS(ty, tx) = A[a + m * ty + tx];
        BS(ty, tx) = B[b + n * ty + tx];
        AS(ty + step, tx) = A[a + m * (ty + step) + tx];
        BS(ty + step, tx) = B[b + n * (ty + step) + tx];

        //Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiply the two matrices together;
        //each thread computes one element of the block sub-matrix
        #ifdef NVIDIA
          #pragma unroll
        #endif
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum[0] = fma(AS(ty, k), BS(k, tx), sum[0]);
            sum[1] = fma(AS(ty + step, k), BS(k, tx), sum[1]);
        }

        //Synchronize to make sure that the preceding computation is done before 
        //loading two new sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int c = m * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + ty * m + tx] = sum[0];
    C[c + (ty + step) * m + tx] = sum[1];
}

void DGEMM1(
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    __local data_t* As,
    __local data_t* Bs,
    int bx,
    int by,
    int tx,
    int ty
) {
    //Index of the first sub-matrix of A processed by the block
    int aBegin = m * BLOCK_SIZE * by;

    //Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + m - 1;

    //Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * n;

    //sum is used to store the element of the block sub-matrix that is computed by the thread
    data_t sum[1] = {0.0};

    //Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        //Load the matrices from device memory to shared memory; 
        //each thread loads one element of each matrix
        AS(ty, tx) = A[a + m * ty + tx];
        BS(ty, tx) = B[b + n * ty + tx];

        //Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiply the two matrices together;
        //each thread computes one element of the block sub-matrix
        #ifdef NVIDIA
          #pragma unroll
        #endif
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum[0] = fma(AS(ty, k), BS(k, tx), sum[0]);

        //Synchronize to make sure that the preceding computation is done before
        //loading two new sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int c = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + n * ty + tx] = sum[0];
}

__kernel void matrixMul(
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    __local data_t* As,
    __local data_t* Bs
) {
    //Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    //Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    int bdimx = n / BLOCK_SIZE;
    int bdimy = m / BLOCK_SIZE;
    int bsizex = get_num_groups(0);
    int bsizey = get_num_groups(1);

    for (int i = bx; i < bdimx; i += bsizex)
        for (int j = by; j < bdimy; j += bsizey)
            DGEMM1(C, A, B, m, n, As, Bs, i, j, tx, ty);
}
