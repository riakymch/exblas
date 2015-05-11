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
    mpfr_t sum, dot, div, op1, op2;

    extrsv_mpfr = (double *) malloc(n * sizeof(double));
    copyVector(n, extrsv_mpfr, x);

    mpfr_init2(op1, 64);
    mpfr_init2(op2, 64);
    mpfr_init2(dot, 128);
    mpfr_init2(div, 196);
    mpfr_init2(sum, 2098);

    //Produce a result matrix of TRSV using MPFR
    for(uint i = 0; i < n; i++) {
        // sum += a[i,j] * x[j], j < i
        mpfr_set_d(sum, 0.0, MPFR_RNDN);
        for(uint j = 0; j < i; j++) {
            mpfr_set_d(op1, a[j * n + i], MPFR_RNDN);
            mpfr_set_d(op2, extrsv_mpfr[j], MPFR_RNDN);
            mpfr_mul(dot, op1, op2, MPFR_RNDN);
            mpfr_sub(sum, sum, dot, MPFR_RNDN);
        }
        // sum = b[i] - sum
        mpfr_set_d(op1, extrsv_mpfr[i], MPFR_RNDN);
        mpfr_add(sum, op1, sum, MPFR_RNDN);
        // x[i] = sum / a[i,i]
        mpfr_set_d(op1, a[i * (n + 1)], MPFR_RNDN);
        mpfr_div(div, sum, op1, MPFR_RNDN);
        extrsv_mpfr[i] = mpfr_get_d(div, MPFR_RNDN);
    }

    //naive trsv
    double *trsvn;
    trsvn = (double *) malloc(n * sizeof(double));
    copyVector(n, trsvn, x);
    for (uint i = 0; i < n; i++) {
        double sum = 0.0;
        for(uint j = 0; j < i; j++)
            sum -= a[j * n + i] * trsvn[j];
        trsvn[i] = (sum + trsvn[i]) / a[i * (n + 1)];
    }

    //compare the GPU and MPFR results
#if 0
    //L2 norm
    double norm = 0.0, val = 0.0;
    for(uint i = 0; i < n; i++) {
        norm += pow(fabs(extrsv[i] - extrsv_mpfr[i]), 2);
        val += pow(fabs(extrsv_mpfr[i]), 2);
    }
    printf("val = %.16g\n", val);
    printf("norm = %.16g\n", norm);
    printf("\n\n");
    norm = ::sqrt(norm) / ::sqrt(val);
#else
    //Inf norm
    double norm = 0.0, val = 0.0, norm1 = 0.0;
    for(uint i = 0; i < n; i++) {
        val = std::max(val, fabs(extrsv_mpfr[i]));
        norm = std::max(norm, fabs(extrsv[i] - extrsv_mpfr[i]));
        norm1 = std::max(norm1, fabs(trsvn[i] - extrsv_mpfr[i]));
    }
    /*printf("val = %.16g\n", val);
    printf("norm = %.16g\n", norm);
    printf("norm1 = %.16g\n", norm1);
    printf("\n\n");*/
    norm = norm / val;
    norm1 = norm1 / val;
#endif

    // test ||b - A * extrsv||
    double *extrsv_mpfr1 = (double *) malloc(n * sizeof(double));
    double *extrsv1 = (double *) malloc(n * sizeof(double));
    double *trsvn1 = (double *) malloc(n * sizeof(double));
    for(uint i = 0; i < n; i++) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        mpfr_set_d(sum, 0.0, MPFR_RNDN);
        for(uint j = 0; j < n; j++) {
            mpfr_set_d(op1, a[j * n + i], MPFR_RNDN);
            mpfr_set_d(op2, extrsv_mpfr[j], MPFR_RNDN);
            mpfr_mul(dot, op1, op2, MPFR_RNDN);
            mpfr_add(sum, sum, dot, MPFR_RNDN);
            sum1 += a[j * n + i] * extrsv[j];
            sum2 += a[j * n + i] * trsvn[j];
        }
        extrsv_mpfr1[i] = mpfr_get_d(sum, MPFR_RNDN);
        extrsv1[i] = sum1;
        trsvn1[i] = sum2;
    }

    double norm02 = 0.0, val0 = 0.0, val2 = 0.0, norm12 = 0.0;
    for(uint i = 0; i < n; i++) {
        val0 = std::max(val0, fabs(x[i]));
        val2 = std::max(val2, fabs(x[i] - extrsv_mpfr1[i]));
        norm02 = std::max(norm02, fabs(x[i] - extrsv1[i]));
        norm12 = std::max(norm12, fabs(x[i] - trsvn1[i]));
    }
    /*printf("val0 = %.16g\n", val0);
    printf("val_res_mpfr = %.16g\n", val2 / val0);
    printf("val_res_extrsv = %.16g\n", norm02 / val0);
    printf("val_res_trsvn = %.16g\n", norm12 / val0);
    printf("\n\n");*/

    /*free(extrsv_mpfr);
    free(extrsv_mpfr1);
    free(extrsv);
    free(extrsv1);
    free(trsvn);
    free(trsvn1);*/
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
    uint n = 64;
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
        init_lognormal_matrix('L', 'N', a, n, mean, stddev);
        init_lognormal(x, n, mean, stddev);
    } else if ((argc > 6) && (argv[6][0] == 'i')) {
        init_ill_cond(a, n * n, range);
        init_ill_cond(x, n, range);
    } else {
        init_fpuniform_matrix('L', 'N', a, n, range, emax);
        init_fpuniform(x, n, range, emax);
    }
    copyVector(n, xorig, x);

    /*for(uint i = 0; i < n; i++)
        printf("%.16g\t", x[i]);
    printf("\n\n");*/

    fprintf(stderr, "%d x %d\n", n, n);

    bool is_pass = true;
    double *superacc;
    double norm;
    err = posix_memalign((void **) &superacc, 64, n * sizeof(double));
    if ((!superacc) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

    copyVector(n, superacc, x);
    extrsv('L', 'N', 'N', n, a, n, superacc, 1, 3);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(superacc, n, a, n, xorig, 1);
    printf("Superacc error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif

    extrsv('L', 'N', 'N', n, a, n, x, 1, 3);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("FPE3 error = %.16g\n", norm);
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
    printf("FPE4 error = %.16g\n", norm);
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
    printf("FPE8 error = %.16g\n", norm);
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
    printf("FPE4EE error = %.16g\n", norm);
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
    printf("FPE6EE error = %.16g\n", norm);
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
    printf("FPE8EE error = %.16g\n", norm);
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

