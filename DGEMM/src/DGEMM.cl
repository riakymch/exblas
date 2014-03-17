
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  // For double precision numbers
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

//Data type used for input data fetches
typedef double data_t;

//Thread block size
#define BLOCK_SIZE 16

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
    int blockRow = get_group_id(1);
    int blockCol = get_group_id(0);

    //Each thread block computes one sub-matrix of C
    __global data_t* Csub = &C[m * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    //Thread row and column within Csub
    int row = get_local_id(1);
    int col = get_local_id(0);
    
    //Each thread computes one element of Csub
    data_t Cvalue = 0.0;

    //Loop over all sub-matrices of A and B to compute Csub
    for (int i = 0; i < (m / BLOCK_SIZE); ++i) {
	//Sub-matrix Asub of A
        __global data_t* Asub = &A[m * BLOCK_SIZE * blockRow + BLOCK_SIZE * i];

	//Sub-matrix Bsub of B
	__global data_t* Bsub = &B[k * BLOCK_SIZE * i + BLOCK_SIZE * blockCol];

	//Load Asub and Bsub from device memory to shared memory
	As[row * BLOCK_SIZE + col] = Asub[row * m + col];
	Bs[row * BLOCK_SIZE + col] = Bsub[row * k + col];

	//Synchronize to make sure that the sub-matrices are loaded before the computation starts
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//Multiply Asub and Bsub
        #ifdef NVIDIA
           #pragma unroll
        #endif
	for (int i = 0; i < BLOCK_SIZE; ++i) {
	    fma(As[row * BLOCK_SIZE + i], Bs[i * BLOCK_SIZE + col], Cvalue);
	}
	Csub[row * m + col] = Cvalue;
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
