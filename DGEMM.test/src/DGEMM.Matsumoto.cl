
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

    int K_wib = K_WG / K_DIMB;
    int N_wib = N_WG / N_DIMB;

    double2 C_pm[M_WI] = {0.0};

    __local data_t B_lm[K_WG * N_WG];

    for (int p_wg = 0; p_wg <= k - K_WG; p_wg += K_WG) {
	//load K_wib x N_wib elements of B into B_lm
    	for (int i = 0; i < K_wib; i++)
    	    for (int j = 0; j < N_wib; j++)
		B_lm[tx + j * N_DIMB + (ty + i * K_DIMB) * N_WG] = B[bx * N_WG + n * p_wg + tx + j * N_DIMB + (ty + i * K_DIMB) * n];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p_wi = 0; p_wi <= K_WG - K_WI; p_wi += K_WI) {
	    double2 A_pm[M_WI];
	    double2 B_pm[K_WI];

	    //load M_WI x K_WI elements of A_lm into A_pm
	    A_pm[0] = vload2(0, &A[p_wg + p_wi + k * M_WG * by + ty * k]);
	    A_pm[1] = vload2(0, &A[p_wg + p_wi + k * M_WG * by + (ty + M_DIMA) * k]);
	    A_pm[2] = vload2(0, &A[p_wg + p_wi + k * M_WG * by + (ty + 2 * M_DIMA) * k]);
	    A_pm[3] = vload2(0, &A[p_wg + p_wi + k * M_WG * by + (ty + 3 * M_DIMA) * k]);
	    A_pm[4] = vload2(0, &A[p_wg + p_wi + k * M_WG * by + (ty + 4 * M_DIMA) * k]);
	    A_pm[5] = vload2(0, &A[p_wg + p_wi + k * M_WG * by + (ty + 5 * M_DIMA) * k]);

	    //load K_WI x N_WI elements of B_lm into B_pm
	    B_pm[0].x = B[p_wi * N_WG + tx];
	    B_pm[0].y = B[(p_wi + 1) * N_WG + tx];
	    B_pm[1].x = B[p_wi * N_WG + tx + N_DIMB];
	    B_pm[1].y = B[(p_wi + 1) * N_WG + tx + N_DIMB];

	    //compute
	    C_pm[0].x = fma(A_pm[0].x, B_pm[0].x, C_pm[0].x);
	    C_pm[0].x = fma(A_pm[0].y, B_pm[0].y, C_pm[0].x);
	    C_pm[0].y = fma(A_pm[0].x, B_pm[1].x, C_pm[0].y);
	    C_pm[0].y = fma(A_pm[0].y, B_pm[1].y, C_pm[0].y);
	    C_pm[1].x = fma(A_pm[1].x, B_pm[0].x, C_pm[1].x);
	    C_pm[1].x = fma(A_pm[1].y, B_pm[0].y, C_pm[1].x);
	    C_pm[1].y = fma(A_pm[1].x, B_pm[1].x, C_pm[1].y);
	    C_pm[1].y = fma(A_pm[1].y, B_pm[1].y, C_pm[1].y);
	    C_pm[2].x = fma(A_pm[2].x, B_pm[0].x, C_pm[2].x);
	    C_pm[2].x = fma(A_pm[2].y, B_pm[0].y, C_pm[2].x);
	    C_pm[2].y = fma(A_pm[2].x, B_pm[1].x, C_pm[2].y);
	    C_pm[2].y = fma(A_pm[2].y, B_pm[1].y, C_pm[2].y);
	    C_pm[3].x = fma(A_pm[3].x, B_pm[0].x, C_pm[3].x);
	    C_pm[3].x = fma(A_pm[3].y, B_pm[0].y, C_pm[3].x);
	    C_pm[3].y = fma(A_pm[3].x, B_pm[1].x, C_pm[3].y);
	    C_pm[3].y = fma(A_pm[3].y, B_pm[1].y, C_pm[3].y);
	    C_pm[4].x = fma(A_pm[4].x, B_pm[0].x, C_pm[4].x);
	    C_pm[4].x = fma(A_pm[4].y, B_pm[0].y, C_pm[4].x);
	    C_pm[4].y = fma(A_pm[4].x, B_pm[1].x, C_pm[4].y);
	    C_pm[4].y = fma(A_pm[4].y, B_pm[1].y, C_pm[4].y);
	    C_pm[5].x = fma(A_pm[5].x, B_pm[0].x, C_pm[5].x);
	    C_pm[5].x = fma(A_pm[5].y, B_pm[0].y, C_pm[5].x);
	    C_pm[5].y = fma(A_pm[5].x, B_pm[1].x, C_pm[5].y);
	    C_pm[5].y = fma(A_pm[5].y, B_pm[1].y, C_pm[5].y);
	}

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < M_WI; i++) {
        C[bx * N_WG + tx + n * M_WG * by + (ty + i * M_DIMC) * n] = C_pm[i].x;
        C[bx * N_WG + tx + N_DIMC + n * M_WG * by + (ty + i * M_DIMC) * n] = C_pm[i].y;
    }
}

