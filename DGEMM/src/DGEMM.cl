
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  // For double precision numbers
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
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
    __local data_t *As,
    __local data_t *Bs
) {
    //Block row and column
    int blockCol = get_group_id(0);
    int blockRow = get_group_id(1);

    //Each thread block computes one sub-matrix of C
    __global data_t* Csub = &C[m * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    //Thread row and column within Csub
    int col = get_local_id(0);
    int row = get_local_id(1);
    
    //Each thread computes one element of Csub
    data_t sum[4] = {0.0};
  
    //step
    int step = 4;
    #ifdef NVIDIA
      step = step * 2;
    #endif

    //Loop over all sub-matrices of A and B to compute Csub
    for (int i = 0; i < (m / BLOCK_SIZE); ++i) {
	//Sub-matrix Asub of A
        __global data_t* Asub = &A[m * BLOCK_SIZE * blockRow + BLOCK_SIZE * i];

	//Sub-matrix Bsub of B
	__global data_t* Bsub = &B[k * BLOCK_SIZE * i + BLOCK_SIZE * blockCol];

	//Load Asub and Bsub from device memory to shared memory
	As[row * BLOCK_SIZE + col] = Asub[row * m + col];
	Bs[row * BLOCK_SIZE + col] = Bsub[row * k + col];
	As[(row + step) * BLOCK_SIZE + col] = Asub[(row + step) * m + col];
	Bs[(row + step) * BLOCK_SIZE + col] = Bsub[(row + step) * k + col];
	As[(row + 2 * step) * BLOCK_SIZE + col] = Asub[(row + 2 * step) * m + col];
	Bs[(row + 2 * step) * BLOCK_SIZE + col] = Bsub[(row + 2 * step) * k + col];
	As[(row + 3 * step) * BLOCK_SIZE + col] = Asub[(row + 3 * step) * m + col];
	Bs[(row + 3 * step) * BLOCK_SIZE + col] = Bsub[(row + 3 * step) * k + col];

	//Synchronize to make sure that the sub-matrices are loaded before the computation starts
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//Multiply Asub and Bsub
        #ifdef NVIDIA
           #pragma unroll
        #endif
	for (int i = 0; i < BLOCK_SIZE; ++i) {
	    sum[0] = fma(As[row * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[0]);
	    sum[1] = fma(As[(row + step) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[1]);
	    sum[2] = fma(As[(row + 2 * step) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[2]);
	    sum[3] = fma(As[(row + 3 * step) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[3]);
	}
	Csub[row * m + col] = sum[0];
	Csub[(row + step) * m + col] = sum[1];
	Csub[(row + 2 * step) * m + col] = sum[2];
	Csub[(row + 3 * step) * m + col] = sum[3];
    }
}

__kernel void matrixMulKernelOld (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k,
    __local data_t *As,
    __local data_t *Bs
) {
    //Block row and column
    int blockRow = get_group_id(1);
    int blockCol = get_group_id(0);

    //Each thread block computes one sub-matrix of C
    __global data_t* Csub = &C[m * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    //Thread row and column within Csub
    int col = get_local_id(0);
    int row = get_local_id(1);
    
    //Each thread computes one element of Csub
    data_t sum[8] = {0.0};

    //Loop over all sub-matrices of A and B to compute Csub
    for (int i = 0; i < (m / BLOCK_SIZE); ++i) {
	//Sub-matrix Asub of A
        __global data_t* Asub = &A[m * BLOCK_SIZE * blockRow + BLOCK_SIZE * i];

	//Sub-matrix Bsub of B
	__global data_t* Bsub = &B[k * BLOCK_SIZE * i + BLOCK_SIZE * blockCol];

	//Load Asub and Bsub from device memory to shared memory
	As[row * BLOCK_SIZE + col] = Asub[row * m + col];
	Bs[row * BLOCK_SIZE + col] = Bsub[row * k + col];
	As[(row + 2) * BLOCK_SIZE + col] = Asub[(row + 2) * m + col];
	Bs[(row + 2) * BLOCK_SIZE + col] = Bsub[(row + 2) * k + col];
	As[(row + 4) * BLOCK_SIZE + col] = Asub[(row + 4) * m + col];
	Bs[(row + 4) * BLOCK_SIZE + col] = Bsub[(row + 4) * k + col];
	As[(row + 6) * BLOCK_SIZE + col] = Asub[(row + 6) * m + col];
	Bs[(row + 6) * BLOCK_SIZE + col] = Bsub[(row + 6) * k + col];
	As[(row + 8) * BLOCK_SIZE + col] = Asub[(row + 8) * m + col];
	Bs[(row + 8) * BLOCK_SIZE + col] = Bsub[(row + 8) * k + col];
	As[(row + 10) * BLOCK_SIZE + col] = Asub[(row + 10) * m + col];
	Bs[(row + 10) * BLOCK_SIZE + col] = Bsub[(row + 10) * k + col];
	As[(row + 12) * BLOCK_SIZE + col] = Asub[(row + 12) * m + col];
	Bs[(row + 12) * BLOCK_SIZE + col] = Bsub[(row + 12) * k + col];
	As[(row + 14) * BLOCK_SIZE + col] = Asub[(row + 14) * m + col];
	Bs[(row + 14) * BLOCK_SIZE + col] = Bsub[(row + 14) * k + col];

	//Synchronize to make sure that the sub-matrices are loaded before the computation starts
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//Multiply Asub and Bsub
        #ifdef NVIDIA
           #pragma unroll
        #endif
	for (int i = 0; i < BLOCK_SIZE; ++i) {
	    sum[0] = fma(As[row * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[0]);
	    sum[1] = fma(As[(row + 2) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[1]);
	    sum[2] = fma(As[(row + 4) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[2]);
	    sum[3] = fma(As[(row + 6) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[3]);
	    sum[4] = fma(As[(row + 8) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[4]);
	    sum[5] = fma(As[(row + 10) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[5]);
	    sum[6] = fma(As[(row + 12) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[6]);
	    sum[7] = fma(As[(row + 14) * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], sum[7]);
	}
	Csub[row * m + col] = sum[0];
	Csub[(row + 2) * m + col] = sum[1];
	Csub[(row + 4) * m + col] = sum[2];
	Csub[(row + 6) * m + col] = sum[3];
	Csub[(row + 8) * m + col] = sum[4];
	Csub[(row + 10) * m + col] = sum[5];
	Csub[(row + 12) * m + col] = sum[6];
	Csub[(row + 14) * m + col] = sum[7];
    }
}

__kernel void matrixMulKernelSimple (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k,
    __local data_t *As,
    __local data_t *Bs
) {
    //Block row and column
    int blockRow = get_group_id(1);
    int blockCol = get_group_id(0);

    //Thread row and column within Csub
    int row = get_local_id(1);
    int col = get_local_id(0);
	
    //Each thread computes one element of Csub
    data_t Cvalue = 0.0;

    //Load Asub and Bsub from device memory to shared memory
    As[row * BLOCK_SIZE + col] = A[blockRow * BLOCK_SIZE + col];
    Bs[row * BLOCK_SIZE + col] = B[row * k + blockCol];

    //Synchronize to make sure that the sub-matrices are loaded before the computation starts
    barrier(CLK_LOCAL_MEM_FENCE);
	
    //Multiply As and Bs
    #ifdef NVIDIA
      #pragma unroll
    #endif
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        Cvalue += As[row * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + col];
    }
    C[blockRow * m + blockCol] = Cvalue;
}
