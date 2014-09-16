
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

    //__local data_t A_lm[M_WG * K_WG];
    __local data_t B_lm[K_WG * N_WG];

    for (int p_wg = 0; p_wg < k - K_WG; p_wg += K_WG) {
	/*//load M_wia x K_wia elements of A into A_lm
    	for (int i = 0; i < M_wia; i++)
    	    for (int j = 0; j < K_wia; j++)
		A_lm[tx + i * K_DIMA + (ty + j * M_DIMA) * K_WG] = A[p_wg + k * M_WG * by + tx + i * K_DIMA + (ty + j * M_DIMA) * k];*/

	//load K_wib x N_wib elements of B into B_lm
    	for (int i = 0; i < K_wib; i++)
    	    for (int j = 0; j < N_wib; j++)
		B_lm[tx + i * N_DIMB + (ty + j * K_DIMB) * N_WG] = B[bx * N_WG + n * p_wg + tx + i * N_DIMB + (ty + j * K_DIMB) * n];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p_wi = 0; p_wi < K_WG - K_WI; p_wi += K_WI) {
	    data_t A_pm[M_WI * K_WI];
	    data_t B_pm[K_WI * N_WI];

	    //load M_WI x K_WI elements of A_lm into A_pm
    	    for (int i = 0; i < M_WI; i++)
    		for (int j = 0; j < K_WI; j++)
	    	    //A_pm[j + i * K_WI] = A_lm[p_wi + j + (ty + i * M_DIMA) * K_WG];
		    A_pm[j + i * K_WI] = A[p_wg + p_wi + j + k * M_WG * by + (ty + i * M_DIMA) * k];

	    //load K_WI x N_WI elements of B_lm into B_pm
    	    for (int i = 0; i < K_WI; i++)
    		for (int j = 0; j < N_WI; j++)
	    	    B_pm[j + i * N_WI] = B_lm[(p_wi + i) * N_WG + tx + j * N_DIMB];

    	    for (int i = 0; i < M_WI; i++)
    		for (int j = 0; j < N_WI; j++) {
		    C_pm[j + i * N_WI] += A_pm[i * K_WI] * B_pm[j];
		    C_pm[j + i * N_WI] += A_pm[1 + i * K_WI] * B_pm[j + N_WI];
		}
	}

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < M_WI; i++)
    	for (int j = 0; j < N_WI; j++)
            C[bx * N_WG + tx + i * N_DIMC + n * M_WG * by + (ty + j * M_DIMC) * n] = C_pm[j + i * N_WI];
}

