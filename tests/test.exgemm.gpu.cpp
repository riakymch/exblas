/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "blas3.hpp"
#include "common.hpp"

#include <iostream>
#include <limits>
#include <string.h>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#ifdef EXBLAS_VS_MPFR
#include <cstddef>
#include <mpfr.h>

extern "C" void printVector(
    const uint n,
    const double *a
){
    printf("x = [");
    for (uint i = 0; i < n; i++)
        printf("%.4g, ", a[i]);
    printf("]\n");
}

extern "C" void printMatrix(
    const int iscolumnwise,
    const uint m,
    const uint n,
    const double *A,
    const uint lda
){
    printf("a = [");
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < n; j++)
            if (iscolumnwise)
                printf("%.4g, ", A[j * lda + i]);
            else
                printf("%.4g, ", A[i * lda + j]);
        printf(";\n");
    }
    printf("]\n");
}

static double exgemmVsMPFR(const bool iscolumnwise, double *exgemm, uint m, uint n, uint k, double alpha, double *a, uint lda, double *b, uint ldb, double beta, double*c, uint ldc) {
    double *exgemm_mpfr;
    mpfr_t sum, dot, op1;

    exgemm_mpfr = (double *) malloc(m * n * sizeof(double));

    mpfr_init2(op1, 64);
    mpfr_init2(dot, 192);
    mpfr_init2(sum, 2098);

    //Produce a result matrix of DGEMM using MPFR
    for(uint i = 0; i < m; i++) {
        for(uint j = 0; j < n; j++) {
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            if (iscolumnwise) {
                for(uint l = 0; l < k; l++) {
                    mpfr_set_d(op1, a[l * lda + i], MPFR_RNDN);
                    mpfr_mul_d(dot, op1, b[j * ldb + l], MPFR_RNDN);
                    mpfr_add(sum, sum, dot, MPFR_RNDN);
                }
                //exgemm_mpfr[j * ldc + i] = mpfr_get_d(sum, MPFR_RNDD);
                exgemm_mpfr[j * ldc + i] = c[j * ldc + i] + mpfr_get_d(sum, MPFR_RNDD);
            } else {
                for(uint l = 0; l < k; l++) {
                    mpfr_set_d(op1, a[i * lda + l], MPFR_RNDN);
                    mpfr_mul_d(dot, op1, b[l * ldb + j], MPFR_RNDN);
                    mpfr_add(sum, sum, dot, MPFR_RNDN);
                }
                //exgemm_mpfr[i * ldc + j] = mpfr_get_d(sum, MPFR_RNDD);
                exgemm_mpfr[i * ldc + j] = c[i * ldc + j] + mpfr_get_d(sum, MPFR_RNDD);
            }
        }
    }
    //printVector(m, exgemm);
    //printVector(m, exgemm_mpfr);
    /*printMatrix(iscolumnwise, m, k, a, lda);
    printMatrix(iscolumnwise, k, n, b, ldb);
    printMatrix(iscolumnwise, m, n, c, ldc);*/

    //Compare the GPU and MPFR results
#if 0
    //Frobenius Norm
    double norm = 0.0, val = 0.0;
    for (uint i = 0; i < m * n; i++) {
        norm += pow(exgemm[i] - exgemm_mpfr[i], 2);
        val += pow(exgemm_mpfr[i], 2);
    }
    norm = ::sqrt(norm) / ::sqrt(val);
#else
    //Inf norm -- maximum absolute row sum norm
    double norm = 0.0, val = 0.0;
    for(uint i = 0; i < m; i++) {
        double rowsum = 0.0, valrowsum = 0.0;
        for(uint j = 0; j < n; j++) {
            if (iscolumnwise) {
                rowsum += fabs(exgemm[j * ldc + i] - exgemm_mpfr[j * ldc + i]);
                valrowsum += fabs(exgemm_mpfr[j * ldc + i]);
            } else {
                rowsum += fabs(exgemm[i * ldc + j] - exgemm_mpfr[i * ldc + j]);
                valrowsum += fabs(exgemm_mpfr[i * ldc + j]);
            }
        }
        val = std::max(val, valrowsum);
        norm = std::max(norm, rowsum);
    }
    norm = norm / val;
#endif

    free(exgemm_mpfr);
    mpfr_free_cache();

    return norm;
}

#else
static double exgemmVsSuperacc(const bool iscolumnwise, double *exgemm, uint m, uint n, double *superacc, uint ldc) {
#if 0
    //Frobenius Norm
    double norm = 0.0, val = 0.0;
    for (uint i = 0; i < m * n; i++) {
        norm += pow(exgemm[i] - superacc[i], 2);
        val += pow(superacc[i], 2);
    }
    norm = ::sqrt(norm) / ::sqrt(val);
#else
    //Inf norm -- maximum absolute row sum norm
    double norm = 0.0, val = 0.0;
    for(uint i = 0; i < m; i++) {
        double rowsum = 0.0, valrowsum = 0.0;
        for(uint j = 0; j < n; j++) {
            if (iscolumnwise) {
                rowsum += fabs(exgemm[j * ldc + i] - superacc[j * ldc + i]);
                valrowsum += fabs(superacc[j * ldc + i]);
            } else {
                rowsum += fabs(exgemm[i * ldc + j] - superacc[i * ldc + j]);
                valrowsum += fabs(superacc[i * ldc + j]);
            }
        }
        val = std::max(val, valrowsum);
        norm = std::max(norm, rowsum);
    }
    norm = norm / val;
#endif

    return norm;
}
#endif

static inline void copyMatrix(const bool iscolumnwise, const uint m, const uint n, double* c, const uint ldc, double* c_orig){
    for(uint i = 0; i < m; i++)
        for(uint j = 0; j < n; j++)
            if (iscolumnwise)
                c[j * ldc + i] = c_orig[j * ldc + i];
            else
                c[i * ldc + j] = c_orig[i * ldc + j];
}


int main(int argc, char *argv[]) {
    int m = 64, n = 64, k = 64;
    double alpha = 1.0, beta = 1.0;
    bool lognormal = false;

    if(argc > 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }
    //bool iscolumnwise = true;
    //int lda = m, ldb = k, ldc = m;
    bool iscolumnwise = false;
    int lda = k, ldb = n, ldc = n;
    if(argc > 6) {
        if(argv[6][0] == 'n') {
            lognormal = true;
        }
    }

    int range = 1;
    int emax = 0;
    double mean = 1., stddev = 1.;
    if(lognormal) {
        stddev = strtod(argv[4], 0);
        mean = strtod(argv[5], 0);
    }
    else {
        if(argc > 4) {
            range = atoi(argv[4]);
        }
        if(argc > 5) {
            emax = atoi(argv[5]);
        }
    }

    double eps = 1e-15;
    double *a, *b, *c, *c_orig;
    int err = posix_memalign((void **) &a, 64, m * k * sizeof(double));
    err &= posix_memalign((void **) &b, 64, k * n * sizeof(double));
    err &= posix_memalign((void **) &c, 64, m * n * sizeof(double));
    err &= posix_memalign((void **) &c_orig, 64, m * n * sizeof(double));
    if ((!a) || (!b) || (!c) || (!c_orig) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");
    if(lognormal) {
        init_lognormal_matrix(iscolumnwise, m, k, a, lda, mean, stddev);
        init_lognormal_matrix(iscolumnwise, k, n, b, ldb, mean, stddev);
        init_lognormal_matrix(iscolumnwise, m, n, c, ldc, mean, stddev);
    } else if ((argc > 6) && (argv[6][0] == 'i')) {
        init_ill_cond(m * k, a, range);
        init_ill_cond(k * n, b, range);
        init_ill_cond(m * n, c, range);
    } else {
        if(range == 1){
            init_naive(m * k, a);
            init_naive(k * n, b);
            init_naive(m * n, c);
        } else {
            init_fpuniform_matrix(iscolumnwise, m, k, a, lda, range, emax);
            init_fpuniform_matrix(iscolumnwise, k, n, b, ldb, range, emax);
            init_fpuniform_matrix(iscolumnwise, m, n, c, ldc, range, emax);
        }
    }
    copyMatrix(iscolumnwise, m, n, c_orig, ldc, c);

    fprintf(stderr, "%d %d %d ", m, n, k);

    if(lognormal) {
        fprintf(stderr, "%f ", stddev);
    } else {
        fprintf(stderr, "%d ", range);
    }

    bool is_pass = true;
    double *superacc;
    double norm;
    err = posix_memalign((void **) &superacc, 64, m * n * sizeof(double));
    if ((!superacc) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");
    copyMatrix(iscolumnwise, m, n, superacc, ldc, c);

    exgemm('N', 'N', m, n, k, alpha, a, lda, b, ldb, beta, superacc, ldc, 1);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(iscolumnwise, superacc, m, n, k, alpha, a, lda, b, ldb, beta, c_orig, ldc);
    printf("Superacc error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif

    copyMatrix(iscolumnwise, m, n, c, ldc, c_orig);
    exgemm('N', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 3);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(iscolumnwise, c, m, n, k, alpha, a, lda, b, ldb, beta, c_orig, ldc);
#else
    norm = exgemmVsSuperacc(iscolumnwise, c, m, n, superacc, ldc);
#endif
    printf("FPE3 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyMatrix(iscolumnwise, m, n, c, ldc, c_orig);
    exgemm('N', 'N', m, n, k, alpha, a, k, b, n, beta, c, n, 4);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(iscolumnwise, c, m, n, k, alpha, a, lda, b, ldb, beta, c_orig, ldc);
#else
    norm = exgemmVsSuperacc(iscolumnwise, c, m, n, superacc, ldc);
#endif
    printf("FPE4 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyMatrix(iscolumnwise, m, n, c, ldc, c_orig);
    exgemm('N', 'N', m, n, k, alpha, a, k, b, n, beta, c, n, 6);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(iscolumnwise, c, m, n, k, alpha, a, lda, b, ldb, beta, c_orig, ldc);
#else
    norm = exgemmVsSuperacc(iscolumnwise, c, m, n, superacc, ldc);
#endif
    printf("FPE6 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyMatrix(iscolumnwise, m, n, c, ldc, c_orig);
    exgemm('N', 'N', m, n, k, alpha, a, k, b, n, beta, c, n, 8);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(iscolumnwise, c, m, n, k, alpha, a, lda, b, ldb, beta, c_orig, ldc);
#else
    norm = exgemmVsSuperacc(iscolumnwise, c, m, n, superacc, ldc);
#endif
    printf("FPE8 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyMatrix(iscolumnwise, m, n, c, ldc, c_orig);
    exgemm('N', 'N', m, n, k, alpha, a, k, b, n, beta, c, n, 4, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(iscolumnwise, c, m, n, k, alpha, a, lda, b, ldb, beta, c_orig, ldc);
#else
    norm = exgemmVsSuperacc(iscolumnwise, c, m, n, superacc, ldc);
#endif
    printf("FPE4EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyMatrix(iscolumnwise, m, n, c, ldc, c_orig);
    exgemm('N', 'N', m, n, k, alpha, a, k, b, n, beta, c, n, 6, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(iscolumnwise, c, m, n, k, alpha, a, lda, b, ldb, beta, c_orig, ldc);
#else
    norm = exgemmVsSuperacc(iscolumnwise, c, m, n, superacc, ldc);
#endif
    printf("FPE6EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyMatrix(iscolumnwise, m, n, c, ldc, c_orig);
    exgemm('N', 'N', m, n, k, alpha, a, k, b, n, beta, c, n, 8, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemmVsMPFR(iscolumnwise, c, m, n, k, alpha, a, lda, b, ldb, beta, c_orig, ldc);
#else
    norm = exgemmVsSuperacc(iscolumnwise, c, m, n, superacc, ldc);
#endif
    printf("FPE8EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
    fprintf(stderr, "\n");

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}

