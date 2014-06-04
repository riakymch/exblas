/**********************************************************************
Copyright ©2012 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

typedef double2 data_t;

#define TILEX       2
#define TILEX_SHIFT 1
#define TILEY       2
#define TILEY_SHIFT 1

/* 
 * Matrix A is cached into local memory block
 * Output tile size : 2x2 = Each thread computes 16 double values
 * Required global threads = (widthC / 2, heightC / 2)
*/
__kernel void mmmKernel_local(
    __global data_t *matrixC,
    __global data_t *matrixA,
    __global data_t *matrixB,
    int widthC,
    __local data_t *blockA
) {
    int blockPos = get_local_id(0) + get_local_size(0) * (get_local_id(1) << TILEY_SHIFT);
    
    //Position of thread will be according to the number of values it writes i.e TILE size
    int globalPos =  get_global_id(0) + (get_global_id(1) << TILEY_SHIFT) * get_global_size(0);

    //Each thread writes 2 data_t
    data_t sum0 = (data_t)(0.0);
    data_t sum1 = (data_t)(0.0);

    int temp = widthC / 2;
    //This loop runs for number of blocks of A in horizontal direction 
    for(int i = 0; i < (temp / get_local_size(0)); i++) {
        //Calculate global ids of threads from the particular block to load from matrix A depending on i
        int globalPosA = i * get_local_size(0) + get_local_id(0) + (get_global_id(1) << TILEY_SHIFT) * temp;

        //Load values in blockA from matrixA
        blockA[blockPos] =		       matrixA[globalPosA];
        blockA[blockPos + get_local_size(0)] = matrixA[globalPosA + temp];

        barrier(CLK_LOCAL_MEM_FENCE);

        //Calculate global ids of threads from the particular block to load from matrix B depending on i
        int globalPosB = get_global_id(0) + ((i * get_local_size(0)) << TILEY_SHIFT) * get_global_size(0);

        //This loop runs for number of threads in horizontal direction in the block of A
        #ifdef NVIDIA
           #pragma unroll
        #endif
        for(int j = 0; j < get_local_size(0) * 2; j=j+2) {
            //Load 2 data_ts from blockA : access patters = strided from local memory
            data_t tempA0 = blockA[(j >> 1) + get_local_id(1) * TILEY * get_local_size(0)];
            data_t tempA1 = blockA[(j >> 1) + (get_local_id(1) * TILEY + 1) * get_local_size(0)];

            //Load corresponding values from matrixB, access pattern = linear from global memory
            data_t tempB0 = matrixB[globalPosB  + j *  get_global_size(0)]; //Should be localId.x * (TILEX / 2)
            data_t tempB1 = matrixB[globalPosB  + (j + 1) * get_global_size(0)];
    
            sum0.x = fma(tempA0.x, tempB0.x, sum0.x);
            sum0.x = fma(tempA0.y, tempB1.x, sum0.x);
            sum0.y = fma(tempA0.x, tempB0.y, sum0.y);
            sum0.y = fma(tempA0.y, tempB1.y, sum0.y);

            sum1.x = fma(tempA1.x, tempB0.x, sum1.x);
            sum1.x = fma(tempA1.y, tempB1.x, sum1.x);
            sum1.y = fma(tempA1.x, tempB0.y, sum1.y);
            sum1.y = fma(tempA1.y, tempB1.y, sum1.y);
		
            /*//old
            sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x;
            sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y;

            sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x;
            sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y;*/
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //Write 4 values to matrixC
    matrixC[globalPos] = sum0;
    matrixC[globalPos +  get_global_size(0)] = sum1;
}

