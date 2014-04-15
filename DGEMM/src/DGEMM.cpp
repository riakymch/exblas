#include <ostream>
#include <stdio.h>
#include <cmath>
#include "DGEMM.hpp"

/*
 * Naive implementation of DGEMM for comparision only; it is much easy to port than the BLAS implementation
 */
void matrixMultiplicationCPUReference(
    double *output,
    double *input0,
    double *input1,
    const uint y,
    const uint x,
    const uint z
) {
    for(cl_uint i = 0; i < y; i++) {
        for(cl_uint j = 0; j < z; j++) {
            for(cl_uint k = 0; k < x; k++) {
                output[i * z + j] += (input0[i * x + k] * input1[k * z + j]);
            }
	    //printf("%.4g\t", output[i * z + j]);
        }
	//printf("\n");
    }
}

int compare(
    const double *refData,
    const double *data,
    const int length,
    const double epsilon
) {
    double error = 0.0;

    for(int i = 0; i < length; ++i) 
    {
        double diff = refData[i] - data[i];
        error += pow(abs(diff), 2);
    }

    error = ::sqrt((double) error);
    printf("error = %.8g\n", error);

    return error < epsilon;
}

void printMatrix(
    double *A,
    int m,
    int n
){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
	     printf("%.4g\t", A[i * m + j]);
	}
	printf("\n");
    } 
}
