
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
    __local data_t *As,
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

/*__kernel void matrixMulKernel (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k,
    __local data_t *Bwrk
) {
    int l, j;
    int i = get_group_id(0);
    int iloc = get_local_id(0);
    int nloc = get_local_size(0);

    data_t Awrk[1024];
    data_t tmp;

    if (i < m) {
	for (l = 0; l < m; l++)
 	    Awrk[l] = A[i * m + l];

        for (j = 0; j < m; j++) {
  	    for (l = iloc; l < m; l+=nloc)
 	         Bwrk[l] = B[l * m + j];
            barrier(CLK_LOCAL_MEM_FENCE);
	    tmp = 0.0;
            for (l = 0; l < m; l++)
		tmp += Awrk[l] * Bwrk[l];
	    C[i * m + j] = tmp;
        }
    }
}*/

/*void saxpy(float a, float *b, float *c)
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
    int k,
    __local data_t *bs
) {
    const int inx = get_local_id(0);
    const int iny = get_local_id(1);
    const int ibx = get_group_id(0) * BLOCK_SIZE;
    const int iby = get_group_id(1) * BLOCK_SIZE;
    const int id = inx + iny * BLOCK_SIZE;
	
    //Load Asub and Bsub from device memory to shared memory
    A += ibx + id;
    B += iby + inx + iny * k;
    C += ibx + id + iby * m;
    const data_t *Blas = B + k * m;

    data_t c[BLOCK_SIZE] = {0.0};

    do {
	for (int i = 0; i < BLOCK_SIZE; i += 4)
	    Bs[(i + iny) * BLOCK_SIZE + inx] = B[i * m];
        barrier(CLK_LOCAL_MEM_FENCE);

    	for (int i = 0; i < BLOCK_SIZE; i++, A += m)
	    axpy(A[0], &bs[i * BLOCK_SIZE], c);
    
        B += BLOCK_SIZE * k;	
        barrier(CLK_LOCAL_MEM_FENCE);
    } while (B < Blast);

    for (int i = 0; i < BLOCK_SIZE; i++, C += m)
        C[0] += c[i];
}*/
