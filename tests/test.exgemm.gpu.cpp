/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie 
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

double exgemmVsMPFR(double *exgemm, uint m, uint n, uint k, double alpha, double *a, uint lda, double *b, uint ldb) {
    double *exgemm_mpfr;
    mpfr_t sum, dot, op1;

    exgemm_mpfr = (double *) malloc(m * n * sizeof(double));

    mpfr_init2(op1, 64);
    mpfr_init2(dot, 128);
    mpfr_init2(sum, 2098);
    mpfr_set_d(dot, 0.0, MPFR_RNDN);

    //Produce a result matrix of DGEMM using MPFR
    for(uint i = 0; i < m; i++) {
        for(uint j = 0; j < n; j++) {
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            for(uint l = 0; l < k; l++) {
                mpfr_set_d(op1, a[i * k + l], MPFR_RNDN);
                mpfr_mul_d(dot, op1, b[l * n + j], MPFR_RNDN);
                mpfr_add(sum, sum, dot, MPFR_RNDN);
            }
            exgemm_mpfr[i * n + j] = mpfr_get_d(sum, MPFR_RNDD);
        }
    }

    double norm = 0.0;
    //Compare the GPU and MPFR results
    for (uint i = 0; i < m * n; i++)
        norm += pow(abs(exgemm[i] - exgemm_mpfr[i]), 2);
    norm = ::sqrt(norm);
    //printf("Compared to MPFR. Norm = %.17g\n", norm);

    free(exgemm_mpfr);
    mpfr_free_cache();

    return norm;
}
#else
double exgemmVsSuperacc(double *exgemm, double *superacc, uint m, uint n, uint k) {
    double norm = 0.0;
    for (uint i = 0; i < m * n; i++)
        norm += pow(abs(superacc[i] - exgemm[i]), 2);
    norm = ::sqrt(norm);
    //printf("Compared to results with superaccs only. Norm = %.17g\n", norm);

    return norm;
}
#endif


int main(int argc, char *argv[]) {
    int m = 64, n = 64, k = 64;
    bool lognormal = false;
    if(argc > 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }
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

    double *a, *b, *c;
    posix_memalign((void **) &a, 64, m * k * sizeof(double));
    posix_memalign((void **) &b, 64, k * n * sizeof(double));
    posix_memalign((void **) &c, 64, m * n * sizeof(double));
    if ((!a) || (!b) || (!c))
        fprintf(stderr, "Cannot allocate memory for the main array\n");
    if(lognormal) {
        init_lognormal(a, m * k, mean, stddev);
        init_lognormal(b, k * n, mean, stddev);
        init_lognormal(c, m * n, mean, stddev);
    } else if ((argc > 6) && (argv[6][0] == 'i')) {
        init_ill_cond(a, m * k, range);
        init_ill_cond(b, k * n, range);
        init_ill_cond(c, m * n, range);
    } else {
        if(range == 1){
            init_naive(a, m * k);
            init_naive(b, k * n);
            init_naive(c, m * n);
        } else {
            init_fpuniform(a, m * k, range, emax);
            init_fpuniform(b, k * n, range, emax);
            init_fpuniform(c, m * n, range, emax);
        }
    }

    fprintf(stderr, "%d \t %d \t %d\n", m, n, k);

    bool is_pass = true;
    double eps = 1e-16;
    double *superacc;
    posix_memalign((void **) &superacc, 64, m * n * sizeof(double));

    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, superacc, n, 1);
#ifdef EXBLAS_VS_MPFR
    if (exgemmVsMPFR(superacc, m, n, k, 1.0, a, k, b, n) > eps) {
        is_pass = false;
    }
#endif

    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 3);
#ifdef EXBLAS_VS_MPFR
    if (exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n) > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 4);
#ifdef EXBLAS_VS_MPFR
    if (exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n) > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 8);
#ifdef EXBLAS_VS_MPFR
    if (exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n) > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 4, true);
#ifdef EXBLAS_VS_MPFR
    if (exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n) > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 6, true);
#ifdef EXBLAS_VS_MPFR
    if (exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n) > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif

    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 8, true);
#ifdef EXBLAS_VS_MPFR
    if (exgemmVsMPFR(c, m, n, k, 1.0, a, k, b, n) > eps) {
        is_pass = false;
    }
#else
    if (exgemmVsSuperacc(c, superacc, m, n, k) > eps) {
        is_pass = false;
    }
#endif
    fprintf(stderr, "\n");

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}

