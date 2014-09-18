
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
    for(uint j = 0; j < n; j++)
        for(uint i = 0; i < m; i++) {
            double sum = 0.0;
            for(uint l = 0; l < k; l++)
                sum += A[i * k + l] * B[l * n + j];
            C[i * n + j] = sum;
        }
}

extern "C" bool compare(
    const double *ref_dgemm,
    const double *dgemm,
    const uint length,
    const double epsilon
) {
    double norm = 0.0;

    for(uint i = 0; i < length; i++)
        norm += pow(abs(ref_dgemm[i] - dgemm[i]), 2);
    norm = ::sqrt(norm);
    printf("Norm = %.17g\n", norm);

    return norm < epsilon ? true : false;
}

extern "C" bool compareDGEMMWithMPFR(
    const double *dgemm,
    const double *h_a,
    const double *h_b,
    const uint m,
    const uint n,
    const uint k,
    const double epsilon
) {
    double *dgemm_mpfr;
    mpfr_t sum, ddot, op1;

    dgemm_mpfr = (double *) malloc(m * n * sizeof(double));

    mpfr_init2(op1, 64);
    mpfr_init2(ddot, 128);
    mpfr_init2(sum, 4196);
    mpfr_set_d(ddot, 0.0, MPFR_RNDN);

    //Produce a result matrix of DGEMM using MPFR
    for(uint i = 0; i < m; i++) {
        for(uint j = 0; j < n; j++) {
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            for(uint l = 0; l < k; l++) {
                mpfr_set_d(op1, h_a[i * k + l], MPFR_RNDN);
                mpfr_mul_d(ddot, op1, h_b[l * n + j], MPFR_RNDN);
                mpfr_add(sum, sum, ddot, MPFR_RNDN);
            }
            dgemm_mpfr[i * n + j] = mpfr_get_d(sum, MPFR_RNDD);
        }
    }

    double norm = 0.0;
    //Compare the GPU and MPFR results
    for (uint i = 0; i < m * n; i++)
        norm += pow(abs(dgemm[i] - dgemm_mpfr[i]), 2);
    norm = ::sqrt(norm);
    printf("Compared to MPFR. Norm = %.17g\n", norm);

    free(dgemm_mpfr);
    mpfr_free_cache();

    return norm < epsilon ? true : false;
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

