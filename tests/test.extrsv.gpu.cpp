/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "blas2.hpp"
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


static void copyVector(uint n, double *x, const double *y) {
    for (uint i = 0; i < n; i++)
        x[i] = y[i];
}

#ifdef EXBLAS_VS_MPFR
#include <cstddef>
#include <mpfr.h>

static double extrsvVsMPFR(char uplo, const double *extrsv, int n, const double *a, uint lda, const double *x, uint incx) {
    mpfr_t sum, dot;

    double *extrsv_mpfr = (double *) malloc(n * sizeof(double));
    copyVector(n, extrsv_mpfr, x);

    mpfr_init2(dot, 128);
    mpfr_init2(sum, 2098);

    //Produce a result matrix of TRSV using MPFR
    if (uplo == 'L') {
        for(int i = 0; i < n; i++) {
            // sum += a[i,j] * x[j], j < i
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            for(int j = 0; j < i; j++) {
                mpfr_set_d(dot, a[j * n + i], MPFR_RNDN);
                mpfr_mul_d(dot, dot, -extrsv_mpfr[j], MPFR_RNDN);
                mpfr_add(sum, sum, dot, MPFR_RNDN);
            }
            mpfr_add_d(sum, sum, extrsv_mpfr[i], MPFR_RNDN);
            mpfr_div_d(sum, sum, a[i * (n + 1)], MPFR_RNDN);
            extrsv_mpfr[i] = mpfr_get_d(sum, MPFR_RNDN);
        }
    } else if (uplo == 'U') {
        for(int i = n-1; i >= 0; i--) {
            // sum += a[i,j] * x[j], j < i
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            for(int j = i+1; j < n; j++) {
                mpfr_set_d(dot, a[j * n + i], MPFR_RNDN);
                mpfr_mul_d(dot, dot, -extrsv_mpfr[j], MPFR_RNDN);
                mpfr_add(sum, sum, dot, MPFR_RNDN);
            }
            mpfr_add_d(sum, sum, extrsv_mpfr[i], MPFR_RNDN);
            mpfr_div_d(sum, sum, a[i * (n + 1)], MPFR_RNDN);
            extrsv_mpfr[i] = mpfr_get_d(sum, MPFR_RNDN);
        }
    }

    //compare the GPU and MPFR results
#if 0
    //L2 norm
    double nrm = 0.0, val = 0.0;
    for(uint i = 0; i < n; i++) {
        nrm += pow(fabs(extrsv[i] - extrsv_mpfr[i]), 2);
        val += pow(fabs(extrsv_mpfr[i]), 2);
    }
    nrm = ::sqrt(nrm) / ::sqrt(val);
#else
    //Inf norm
    double nrm = 0.0, val = 0.0;
    for(int i = 0; i < n; i++) {
        val = std::max(val, fabs(extrsv_mpfr[i]));
        nrm = std::max(nrm, fabs(extrsv[i] - extrsv_mpfr[i]));
        //printf("%.16g\t", fabs(extrsv[i] - extrsv_mpfr[i]));
    }
    nrm = nrm / val;
#endif

    free(extrsv_mpfr);
    mpfr_free_cache();

    return nrm;
}

#else
static double extrsvVsSuperacc(uint n, double *extrsv, double *superacc) {
    double nrm = 0.0, val = 0.0;
    for (uint i = 0; i < n; i++) {
        nrm += pow(fabs(extrsv[i] - superacc[i]), 2);
        val += pow(fabs(superacc[i]), 2);
    }
    nrm = ::sqrt(nrm) / ::sqrt(val);

    return nrm;
}
#endif


int main(int argc, char *argv[]) {
    char uplo = 'U';
    char transa = 'N';
    char diag = 'U';
    uint n = 64;
    bool lognormal = false;
    if(argc > 1)
        uplo = argv[1][0];
    if(argc > 2)
        transa = argv[2][0];
    if(argc > 3)
        diag = argv[3][0];
    if(argc > 4)
        n = atoi(argv[4]);
    if(argc > 7) {
        if(argv[7][0] == 'n') {
            lognormal = true;
        }
    }

    int range = 1;
    int emax = 0;
    double mean = 1., stddev = 1.;
    if(lognormal) {
        stddev = strtod(argv[5], 0);
        mean = strtod(argv[6], 0);
    }
    else {
        if(argc > 5) {
            range = atoi(argv[5]);
        }
        if(argc > 6) {
            emax = atoi(argv[6]);
        }
    }

    double eps = 1e-13;
    double *a, *x, *xorig;
    int err = posix_memalign((void **) &a, 64, n * n * sizeof(double));
    err &= posix_memalign((void **) &x, 64, n * sizeof(double));
    err &= posix_memalign((void **) &xorig, 64, n * sizeof(double));
    if ((!a) || (!x) || (!xorig) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

    if(lognormal) {
        printf("init_lognormal_tr_matrix\n");
        init_lognormal_tr_matrix(uplo, diag, n, a, mean, stddev);
        init_lognormal(n, xorig, mean, stddev);
    } else if ((argc > 7) && (argv[7][0] == 'i')) {
        printf("init_ill_cond\n");
        init_ill_cond(n * n, a, range);
        init_ill_cond(n, xorig, range);
    } else {
        printf("init_fpuniform_tr_matrix\n");
        init_fpuniform_tr_matrix(uplo, diag, n, a, range, emax);
        init_fpuniform(n, xorig, range, emax);
    }
    copyVector(n, x, xorig);

    fprintf(stderr, "%d ", n);

    if(lognormal) {
        fprintf(stderr, "%f ", stddev);
    } else {
        fprintf(stderr, "%d ", range);
    }

    bool is_pass = true;
    double *superacc;
    double norm;
    err = posix_memalign((void **) &superacc, 64, n * sizeof(double));
    if ((!superacc) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

	// DTRSV on GPU
    copyVector(n, superacc, xorig);
    extrsv(uplo, transa, diag, n, a, n, 0, superacc, 1, 0, 1);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, superacc, n, a, n, xorig, 1);
    printf("DTRSV error = %.16g\n", norm);
#endif

    copyVector(n, superacc, xorig);
    extrsv(uplo, transa, diag, n, a, n, 0, superacc, 1, 0, 0);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, superacc, n, a, n, xorig, 1);
    printf("Superacc error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv(uplo, transa, diag, n, a, n, 0, x, 1, 0, 3);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
#else
    norm = extrsvVsSuperacc(n, x, superacc);
#endif
    printf("FPE3 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector(n, x, xorig);
    extrsv(uplo, transa, diag, n, a, n, 0, x, 1, 0, 4);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
#else
    norm = extrsvVsSuperacc(n, x, superacc);
#endif
    printf("FPE4 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector(n, x, xorig);
    extrsv(uplo, transa, diag, n, a, n, 0, x, 1, 0, 8);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
#else
    norm = extrsvVsSuperacc(n, x, superacc);
#endif
    printf("FPE8 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector(n, x, xorig);
    extrsv(uplo, transa, diag, n, a, n, 0, x, 1, 0, 4, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
#else
    norm = extrsvVsSuperacc(n, x, superacc);
#endif
    printf("FPE4EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector(n, x, xorig);
    extrsv(uplo, transa, diag, n, a, n, 0, x, 1, 0, 6, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
#else
    norm = extrsvVsSuperacc(n, x, superacc);
#endif
    printf("FPE6EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector(n, x, xorig);
    extrsv(uplo, transa, diag, n, a, n, 0, x, 1, 0, 8, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(uplo, x, n, a, n, xorig, 1);
#else
    norm = extrsvVsSuperacc(n, x, superacc);
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

