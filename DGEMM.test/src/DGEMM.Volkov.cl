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
    c[0] = fma(a, b[0], c[0]);
    c[1] = fma(a, b[1], c[1]);
    c[2] = fma(a, b[2], c[2]);
    c[3] = fma(a, b[3], c[3]);
    c[4] = fma(a, b[4], c[4]);
    c[5] = fma(a, b[5], c[5]);
    c[6] = fma(a, b[6], c[6]);
    c[7] = fma(a, b[7], c[7]);
    c[8] = fma(a, b[8], c[8]);
    c[9] = fma(a, b[9], c[9]);
    c[10] = fma(a, b[10], c[10]);
    c[11] = fma(a, b[11], c[11]);
    c[12] = fma(a, b[12], c[12]);
    c[13] = fma(a, b[13], c[13]);
    c[14] = fma(a, b[14], c[14]);
    c[15] = fma(a, b[15], c[15]);
}

__kernel void matrixMulKernel (
    __global data_t* C,
    __global data_t* A,
    __global data_t* B,
    int m,
    int n,
    int k
) {
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);

    int M_wia = M_WG / M_DIMA;
    int K_wia = K_WG / K_DIMA;
    int K_wib = K_WG / K_DIMB;
    int N_wib = N_WG / N_DIMB;

    data_t C_pm[M_WI * N_WI] = {0.0};

    __local data_t A_lm[M_WG * K_WG];
    __local data_t B_lm[K_WG * N_WG];

    for (int p_wg = 0; p_wg < k - K_WG; p_wg += K_WG) {
	//load M_wia x K_wia elements of A into A_lm
    	for (int j = 0; j < M_wia; j++)
    	    for (int i = 0; i < K_wia; i++)
		A_lm[tx + ty] = 

	//load K_wib x N_wib elements of B into B_lm
    	for (int i = 0; i < M_wia; i++)
    	    for (int j = 0; j < K_wia; j++)
		B_lm[tx ] = 

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p_wi = 0; p_Wi < K_WG - K_WI; p_wi += K_WI) {
            saxpy(A[0], &bs[i][0], c);
	}

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < M_wi; i++)
    	for (int j = 0; j < N_wi; j++)
            C[0] = C_pm[i + j];
}

