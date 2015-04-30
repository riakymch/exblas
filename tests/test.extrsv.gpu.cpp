/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie 
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


static void copyVector(uint n, double *x, double *y) {
    for (uint i = 0; i < n; i++)
        x[i] = y[i];
}

#ifdef EXBLAS_VS_MPFR
#include <cstddef>
#include <mpfr.h>

static double extrsvVsMPFR(double *extrsv, uint n, double *a, uint lda, double *x, uint incx) {
    double *extrsv_mpfr;
    mpfr_t sum, dot, div, op1;

    extrsv_mpfr = (double *) malloc(n * sizeof(double));
    copyVector(n, extrsv_mpfr, x);

    mpfr_init2(op1, 64);
    mpfr_init2(dot, 128);
    mpfr_init2(div, 128);
    mpfr_init2(sum, 2098);

    //Produce a result matrix of TRSV using MPFR
    for(uint i = 0; i < n; i++) {
        mpfr_set_d(sum, extrsv_mpfr[i], MPFR_RNDN);
        for(uint j = 0; j < i; j++) {
            mpfr_set_d(op1, a[j * n + i], MPFR_RNDN);
            mpfr_mul_d(dot, op1, extrsv_mpfr[j], MPFR_RNDN);
            mpfr_sub(sum, sum, dot, MPFR_RNDN);
        }
        mpfr_div_d(div, sum, a[i * (n + 1)], MPFR_RNDN);
        extrsv_mpfr[i] = mpfr_get_d(div, MPFR_RNDN);
    }

    //Compare the GPU and MPFR results
    //L2 norm
    double norm = 0.0, val = 0.0;
    for(uint i = 0; i < n; i++) {
        norm += pow(fabs(extrsv[i] - extrsv_mpfr[i]), 2);
        val += pow(fabs(extrsv_mpfr[i]), 2);
    }
    norm = ::sqrt(norm) / ::sqrt(val);

    //free(extrsv_mpfr);
    mpfr_free_cache();

    return norm;
}
#else
static double extrsvVsSuperacc(uint n, double *extrsv, double *superacc) {
    double norm = 0.0, val = 0.0;
    for (uint i = 0; i < n; i++) {
        norm += pow(fabs(extrsv[i] - superacc[i]), 2);
        val += pow(fabs(superacc[i]), 2);
    }
    norm = ::sqrt(norm) / ::sqrt(val);

    return norm;
}
#endif


int main(int argc, char *argv[]) {
    int n = 64;
    bool lognormal = false;
    if(argc > 1)
        n = atoi(argv[1]);
    if(argc > 4) {
        if(argv[4][0] == 'n') {
            lognormal = true;
        }
    }

    int range = 1;
    int emax = 0;
    double mean = 1., stddev = 1.;
    if(lognormal) {
        stddev = strtod(argv[2], 0);
        mean = strtod(argv[3], 0);
    }
    else {
        if(argc > 2) {
            range = atoi(argv[2]);
        }
        if(argc > 3) {
            emax = atoi(argv[3]);
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
        init_lognormal(a, n * n, mean, stddev);
        init_lognormal(x, n, mean, stddev);
    } else if ((argc > 6) && (argv[6][0] == 'i')) {
        init_ill_cond(a, n * n, range);
        init_ill_cond(x, n, range);
    } else {
        if(range == 1){
            init_naive(a, n * n);
            init_naive(x, n);
        } else {
            init_fpuniform(a, n * n, range, emax);
            init_fpuniform(x, n, range, emax);
        }
    }
    copyVector(n, xorig, x);

    fprintf(stderr, "%d x %d\n", n, n);

    bool is_pass = true;
    double *superacc;
    double norm;
    err = posix_memalign((void **) &superacc, 64, n * sizeof(double));
    if ((!superacc) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

    copyVector(n, superacc, x);
    extrsv('L', 'N', 'N', n, a, n, superacc, 1, 0);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(superacc, n, a, n, xorig, 1);
    printf("Superacc norm = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif

    extrsv('L', 'N', 'N', n, a, n, x, 1, 3);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("FPE3 norm = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv('L', 'N', 'N', n, a, n, x, 1, 4);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("FPE4 norm = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv('L', 'N', 'N', n, a, n, x, 1, 8);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("FPE8 norm = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv('L', 'N', 'N', n, a, n, x, 1, 4, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("FPE4EE norm = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv('L', 'N', 'N', n, a, n, x, 1, 6, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("FPE6EE norm = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
        is_pass = false;
    }
#endif

    copyVector(n, x, xorig);
    extrsv('L', 'N', 'N', n, a, n, x, 1, 8, true);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("FPE8EE norm = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#else
    if (extrsvVsSuperacc(n, x, superacc) > eps) {
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

