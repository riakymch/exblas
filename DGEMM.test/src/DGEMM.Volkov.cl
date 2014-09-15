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

    //Load Asub and Bsub from device memory to shared memory
    A += ibx + id;
    B += inx + mul24(iby + iny, k);
    C += ibx + id + mul24(iby, m);

    //TODO: replace this with computable statements
    int M_wia = 2;
    int K_wia = 8;
    int K_wib = 8;
    int N_wib = 8;
    data_t C_pm[M_wia][N_wia] = {0.0};

    __local data_t A_lm[M_WG][K_WG];
    __local data_t B_lm[K_WG][N_WG];

    for (int p_wg = 0; p_wg < k - K_WG; j += K_WG) {
    	for (int i = 0; i < M_wia; i++)
    	    for (int j = 0; j < K_wia; j++)
		A_lm[tx ][] = 
        barrier(CLK_LOCAL_MEM_FENCE);

        #ifdef NVIDIA
            #pragma unroll
        #endif
        for (int i = 0; i < 16; i++, A += m)
            saxpy(A[0], &bs[i][0], c);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < M_wia; i++)
    	for (int j = 0; j < N_wia; j++)
            C[0] = C_pm[i][j];
}

