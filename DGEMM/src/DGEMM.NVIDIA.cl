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

#pragma OPENCL EXTENSION cl_khr_fp64                   : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

typedef double data_t;

#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]

///////////////////////////////////////////////////////////////////////////////
// Matrix multiplication on the device: C = A * B
// uiWA is A's width and uiWB is B's width
////////////////////////////////////////////////////////////////////////////////
__kernel void matrixMul(
    __global data_t* C,
    __global data_t* A,
    __global data_t* B, 
    int uiWA,
    int uiWB,
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
    int aBegin = uiWA * BLOCK_SIZE * by;

    //Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + uiWA - 1;

    //Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * uiWB;

    //sum is used to store the element of the block sub-matrix
    //that is computed by the thread
    data_t sum[2] = {0.0, 0.0};

    //Loop over all the sub-matrices of A and B
    //required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        //Load the matrices from device memory
        //to shared memory; each thread loads
        //one element of each matrix
        AS(ty, tx) = A[a + uiWA * ty + tx];
        BS(ty, tx) = B[b + uiWB * ty + tx];
        AS(ty + 16, tx) = A[a + uiWA * (ty + 16) + tx];
        BS(ty + 16, tx) = B[b + uiWB * (ty + 16) + tx];
	
        //Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiply the two matrices together;
        //each thread computes one element
        //of the block sub-matrix        
        #ifdef NVIDIA
          #pragma unroll
        #endif
        for (int k = 0; k < BLOCK_SIZE; ++k) {
	    sum[0] += AS(ty, k) * BS(k, tx);
	    sum[1] += AS(ty + 16, k) * BS(k, tx);
	}

        //Synchronize to make sure that the preceding
        //computation is done before loading two new
        //sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int c = uiWB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    //C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = sum;
    C[c + uiWB * ty + tx] = sum[0];
    C[c + uiWB * (ty + 16) + tx] = sum[1];
}

