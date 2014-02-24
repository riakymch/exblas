
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  // For double precision numbers
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  // For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

//Data type used for input data fetches
typedef double data_t;

//Thread block size
#define BLOCK_SIZE 16

__kernel void matrixMulKernel (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int Cwidth,
    int Cheight,
    int Bwidth
) {
    __local data_t As[BLOCK_SIZE][BLOCK_SIZE];
    __local data_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    //Block row and column
    int blockRow = get_group_id(1);
    int blockCol = get_group_id(0);

    //Each thread block computes one sub-matrix of C
    __global data_t* Csub = &C[Cwidth * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    //Thread row and column within Csub
    int row = get_local_id(1);
    int col = get_local_id(0);
    
    //Each thread computes one element of Csub
    data_t Cvalue = 0.0;

    //Loop over all sub-matrices of A and B to compute Csub
    for (int m = 0; m < (Cwidth / BLOCK_SIZE); ++m) {
	//Sub-matrix Asub of A
        __global data_t* Asub = &A[Cwidth * BLOCK_SIZE * blockRow + BLOCK_SIZE * m];

	//Sub-matrix Bsub of B
	__global data_t* Bsub = &B[Bwidth * BLOCK_SIZE * m + BLOCK_SIZE * blockCol];

	//Load Asub and Bsub from device memory to shared memory
	As[row][col] = Asub[row * Cwidth + col];
	Bs[row][col] = Bsub[row * Bwidth + col];

	//Synchronize to make sure that the sub-matrices are loaded before the computation starts
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//Multiply Asub and Bsub
	for (int i = 0; i < BLOCK_SIZE; ++i) {
	    Cvalue += As[row][i] * Bs[i][col];
	   
            //Synchronize to make sure that the preceding computation is done before loading 
            //two new sub-matrices of A and B in the next iteration
	    //barrier(CLK_LOCAL_MEM_FENCE);
	}
	Csub[row * Cwidth + col] = Cvalue;
    }
}

__kernel void matrixMulKernelSimple (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int Cwidth,
    int Cheight,
    int Bwidth
) {
    __local data_t As[BLOCK_SIZE][BLOCK_SIZE];
    __local data_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    //Block row and column
    int blockRow = get_group_id(1);
    int blockCol = get_group_id(0);

    //Thread row and column within Csub
    int row = get_local_id(1);
    int col = get_local_id(0);
	
    //Each thread computes one element of Csub
    data_t Cvalue = 0.0;

    //Load Asub and Bsub from device memory to shared memory
    As[row][col] = A[blockRow * BLOCK_SIZE + col];
    Bs[row][col] = B[row * Cheight + blockCol];

    //Synchronize to make sure that the sub-matrices are loaded before the computation starts
    barrier(CLK_LOCAL_MEM_FENCE);
	
    //Multiply As and Bs
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        Cvalue += As[row][i] * Bs[i][col];
    }
    C[blockRow * Cheight + blockCol] = Cvalue;
}
