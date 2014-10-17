
#include "TRSV.hpp"

/*
 * Naive implementation of TRSV for comparision only; it is much easy to port than the BLAS implementation
 */
extern "C" int TRSVUNN(
    const double *u,
    const double *b,
    double *x,
    const int n
) {
    double s;

    for(int i = n-1; i >= 0; i--) {
        s = b[i];
        for(int j = i+1; j < n; j++)
            s = s - u[i * n + j] * x[j];
        x[i] = s / u[i * (n + 1)];
    }

    return 1;
}

extern "C" bool compare(
    const double *trsv_cpu,
    const double *trsv_gpu,
    const uint n,
    const double epsilon
) {
    double norm = 0.0;

    for(uint i = 0; i < n; i++)
        norm += pow(abs(trsv_cpu[i] - trsv_gpu[i]), 2);
    norm = ::sqrt(norm);
    printf("Norm = %.15g\n", norm);

    return norm < epsilon ? true : false;
}


extern "C" bool compareTRSVUNNToMPFR(
    const double *u,
    const double *b,
    const double *trsv,
    const int n,
    const double epsilon
) {
    double *trsv_mpfr;
    mpfr_t sum, dot, div, op1;

    trsv_mpfr = (double *) malloc(n * sizeof(double));

    mpfr_init2(op1, 64);
    mpfr_init2(dot, 128);
    mpfr_init2(div, 128);
    mpfr_init2(sum, 4196);

    //Produce a result matrix of TRSV using MPFR
    for(int i = n-1; i >= 0; i--) {
        mpfr_set_d(sum, b[i], MPFR_RNDN);
        for(int j = i+1; j < n; j++) {
            mpfr_set_d(op1, u[i * n + j], MPFR_RNDN);
            mpfr_mul_d(dot, op1, trsv_mpfr[j], MPFR_RNDN);
            mpfr_sub(sum, sum, dot, MPFR_RNDN);
        }
        mpfr_div_d(div, sum, u[i * (n + 1)], MPFR_RNDN);
        trsv_mpfr[i] = mpfr_get_d(div, MPFR_RNDD);
    }

    double norm = 0.0;
    //Compare the GPU and MPFR results
    for (int i = 0; i < n; i++)
        norm += pow(abs(trsv[i] - trsv_mpfr[i]), 2);
    norm = ::sqrt(norm);
    printf("Compared to MPFR. Norm = %.17g\n", norm);

    free(trsv_mpfr);
    mpfr_free_cache();

    return norm < epsilon ? true : false;
}

extern "C" void printMatrix(
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

extern "C" void printVector(
    const double *a,
    const uint n
){
    for (uint i = 0; i < n; i++)
        printf("%.4g\t", a[i]);
    printf("\n");
}

