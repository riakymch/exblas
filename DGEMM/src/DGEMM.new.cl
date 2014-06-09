
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_khr_fp64                 : enable  // For double precision numbers
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

//Data type used for input data fetches
typedef double data_t;

__kernel void matrixMulKernel (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k,
    __local data_t *Bs
) {
    //Block ty and txumn
    int by = get_group_id(1);
    int bx = get_group_id(0);

    //Thread ty and txumn within Csub
    int ty = get_local_id(1);
    int tx = get_local_id(0);
	
    //Each thread computes one element of Csub
    data_t Cvalue = 0.0;

    //Load Asub and Bsub from device memory to shared memory
    As[ty * BLOCK_SIZE + tx] = A[by * BLOCK_SIZE + tx];
    Bs[ty * BLOCK_SIZE + tx] = B[ty * k + bx];

    //Synchronize to make sure that the sub-matrices are loaded before the computation starts
    barrier(CLK_LOCAL_MEM_FENCE);
	
    //Multiply As and Bs
    #ifdef NVIDIA
      #pragma unroll
    #endif
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        Cvalue += As[ty * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + tx];
    }
    C[by * m + bx] = Cvalue;
}

__kernel void matrixMulKernel4 (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k,
    __local data_t *As,
    __local data_t *Bs
) {
    //Block ty and txumn
    int bx = get_group_id(0);
    int by = get_group_id(1);

    //Each thread block computes one sub-matrix of C
    __global data_t* Csub = &C[m * BLOCK_SIZE * by + BLOCK_SIZE * bx];

    //Thread ty and txumn within Csub
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
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
    Csub[ty * m + tx] = sum[0];
    Csub[(ty + step) * m + tx] = sum[1];
    Csub[(ty + 2 * step) * m + tx] = sum[2];
    Csub[(ty + 3 * step) * m + tx] = sum[3];
}

__kernel void matrixMulKernel8 (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k,
    __local data_t *As,
    __local data_t *Bs
) {
    //Block ty and txumn
    int by = get_group_id(1);
    int bx = get_group_id(0);

    //Each thread block computes one sub-matrix of C
    __global data_t* Csub = &C[m * BLOCK_SIZE * by + BLOCK_SIZE * bx];

    //Thread ty and txumn within Csub
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
    //Each thread computes one element of Csub
    data_t sum[8] = {0.0};

    //Loop over all sub-matrices of A and B to compute Csub
    for (int i = 0; i < (m / BLOCK_SIZE); ++i) {
	//Sub-matrix Asub of A
        __global data_t* Asub = &A[m * BLOCK_SIZE * by + BLOCK_SIZE * i];

	//Sub-matrix Bsub of B
	__global data_t* Bsub = &B[k * BLOCK_SIZE * i + BLOCK_SIZE * bx];

	//Load Asub and Bsub from device memory to shared memory
	As[ty * BLOCK_SIZE + tx] = Asub[ty * m + tx];
	Bs[ty * BLOCK_SIZE + tx] = Bsub[ty * k + tx];
	As[(ty + 2) * BLOCK_SIZE + tx] = Asub[(ty + 2) * m + tx];
	Bs[(ty + 2) * BLOCK_SIZE + tx] = Bsub[(ty + 2) * k + tx];
	As[(ty + 4) * BLOCK_SIZE + tx] = Asub[(ty + 4) * m + tx];
	Bs[(ty + 4) * BLOCK_SIZE + tx] = Bsub[(ty + 4) * k + tx];
	As[(ty + 6) * BLOCK_SIZE + tx] = Asub[(ty + 6) * m + tx];
	Bs[(ty + 6) * BLOCK_SIZE + tx] = Bsub[(ty + 6) * k + tx];
	As[(ty + 8) * BLOCK_SIZE + tx] = Asub[(ty + 8) * m + tx];
	Bs[(ty + 8) * BLOCK_SIZE + tx] = Bsub[(ty + 8) * k + tx];
	As[(ty + 10) * BLOCK_SIZE + tx] = Asub[(ty + 10) * m + tx];
	Bs[(ty + 10) * BLOCK_SIZE + tx] = Bsub[(ty + 10) * k + tx];
	As[(ty + 12) * BLOCK_SIZE + tx] = Asub[(ty + 12) * m + tx];
	Bs[(ty + 12) * BLOCK_SIZE + tx] = Bsub[(ty + 12) * k + tx];
	As[(ty + 14) * BLOCK_SIZE + tx] = Asub[(ty + 14) * m + tx];
	Bs[(ty + 14) * BLOCK_SIZE + tx] = Bsub[(ty + 14) * k + tx];

	//Synchronize to make sure that the sub-matrices are loaded before the computation starts
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//Multiply Asub and Bsub
        #ifdef NVIDIA
           #pragma unroll
        #endif
	for (int i = 0; i < BLOCK_SIZE; ++i) {
	    sum[0] = fma(As[ty * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + tx], sum[0]);
	    sum[1] = fma(As[(ty + 2) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + tx], sum[1]);
	    sum[2] = fma(As[(ty + 4) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + tx], sum[2]);
	    sum[3] = fma(As[(ty + 6) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + tx], sum[3]);
	    sum[4] = fma(As[(ty + 8) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + tx], sum[4]);
	    sum[5] = fma(As[(ty + 10) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + tx], sum[5]);
	    sum[6] = fma(As[(ty + 12) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + tx], sum[6]);
	    sum[7] = fma(As[(ty + 14) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + tx], sum[7]);
	}
	Csub[ty * m + tx] = sum[0];
	Csub[(ty + 2) * m + tx] = sum[1];
	Csub[(ty + 4) * m + tx] = sum[2];
	Csub[(ty + 6) * m + tx] = sum[3];
	Csub[(ty + 8) * m + tx] = sum[4];
	Csub[(ty + 10) * m + tx] = sum[5];
	Csub[(ty + 12) * m + tx] = sum[6];
	Csub[(ty + 14) * m + tx] = sum[7];
    }
}
