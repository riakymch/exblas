#include <ostream>
#include <stdio.h>
#include <cmath>
#include "DGEMM.hpp"

/*
 * Naive implementation of DGEMM for comparision only; it is much easy to port than the BLAS implementation
 */
void DGEMMCPU(
    double *C,
    const double *A,
    const double *B,
    const uint m,
    const uint n,
    const uint k
) {
    for(uint i = 0; i < m; i++) {
        for(uint j = 0; j < n; j++) {
            for(uint l = 0; l < k; l++) {
                C[j * m + i] += A[l * m + i] * B[j * k + l];
            }
	    //printf("%.4g\t", output[i * z + j]);
        }
	//printf("\n");
    }
}

int compare(
    const double *ref_dgemm,
    const double *dgemm,
    const uint length,
    const double epsilon
) {
    double error = 0.0;

    for(uint i = 0; i < length; ++i) 
    {
        double diff = ref_dgemm[i] - dgemm[i];
        error += pow(abs(diff), 2);
    }

    error = ::sqrt((double) error);
    printf("error = %.8g\n", error);

    return error < epsilon;
}

void printMatrix(
    const double *A,
    const uint m,
    const uint n
){
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < n; j++) {
	     printf("%.4g\t", A[i * m + j]);
	}
	printf("\n");
    } 
}
