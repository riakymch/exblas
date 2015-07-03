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
#if 1
    // Compare to the results from Matlab
    FILE *pFilex;
    size_t resx;
    //pFilex = fopen("matrices/x_test_trsv_gemv_e13_64.bin", "rb");
    //pFilex = fopen("matrices/x_test_trsv_final_64.bin", "rb");
    //pFilex = fopen("matrices/x_test_trsv_e21_64.bin", "rb");
    //pFilex = fopen("matrices/x_test_trsv_e40_64.bin", "rb");
    pFilex = fopen("matrices/unn/xe_unn_100_3.08e+07.bin", "rb");
    if (pFilex == NULL) {
        fprintf(stderr, "Cannot open files to read matrix and vector\n");
        exit(1);
    }

    double *xmatlab = (double *) malloc(n * sizeof(double));
    resx = fread(xmatlab, sizeof(double), n, pFilex);
    if (resx != n) {
        fprintf(stderr, "Cannot read matrix and vector from files\n");
        exit(1);
    }
    fclose(pFilex);

    for(uint i = 0; i < n; i++)
        printf("%.16g\t", xmatlab[i]);
    printf("\n\n");

    for(uint i = 0; i < n; i++)
        printf("%.16g\t", extrsv[i]);
    printf("\n\n");

    //Inf norm
    double nrm2 = 0.0, val2 = 0.0;
    for(uint i = 0; i < n; i++) {
        val2 = std::max(val2, fabs(xmatlab[i]));
        nrm2 = std::max(nrm2, fabs(extrsv[i] - xmatlab[i]));
        if (fabs(extrsv[i] - xmatlab[i]) != 0.0)
            printf("\n %d \t", i);
        printf("%.16g\t", fabs(extrsv[i] - xmatlab[i]));
    }
    printf("\n\n");
    printf("ExTRSV vs Matlab = %.16g \t %.16g\n", nrm2, val2);
    nrm2 = nrm2 / val2;
    printf("ExTRSV vs Matlab = %.16g\n", nrm2);

    return nrm2;
#else

    mpfr_t sum, dot;

    double *extrsv_mpfr = (double *) malloc(n * sizeof(double));
    copyVector(n, extrsv_mpfr, x);

    mpfr_init2(dot, 128);
    mpfr_init2(sum, 2098);

    //Produce a result matrix of TRSV using MPFR
#if 0
    for(uint i = 0; i < n; i++) {
        // sum += a[i,j] * x[j], j < i
        mpfr_set_d(sum, 0.0, MPFR_RNDN);
        for(uint j = 0; j < i; j++) {
            mpfr_set_d(dot, a[j * n + i], MPFR_RNDN);
            mpfr_mul_d(dot, dot, -extrsv_mpfr[j], MPFR_RNDN);
            mpfr_add(sum, sum, dot, MPFR_RNDN);
        }
        mpfr_add_d(sum, sum, extrsv_mpfr[i], MPFR_RNDN);
        mpfr_div_d(sum, sum, a[i * (n + 1)], MPFR_RNDN);
        extrsv_mpfr[i] = mpfr_get_d(sum, MPFR_RNDN);
    }
#else
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
#endif

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
    for(uint i = 0; i < n; i++) {
        val = std::max(val, fabs(extrsv_mpfr[i]));
        nrm = std::max(nrm, fabs(extrsv[i] - extrsv_mpfr[i]));
        //printf("%.16g\t", fabs(extrsv[i] - extrsv_mpfr[i]));
    }
    nrm = nrm / val;
    printf("\n\n");
#endif

    // test ||b - A * extrsv||
    /*double *extrsv_mpfr1 = (double *) malloc(n * sizeof(double));
    double *extrsv1 = (double *) malloc(n * sizeof(double));
    for(uint i = 0; i < n; i++) {
        double sum1 = 0.0;
        mpfr_set_d(sum, 0.0, MPFR_RNDN);
        for(uint j = 0; j < n; j++) {
            mpfr_set_d(op1, a[j * n + i], MPFR_RNDN);
            mpfr_set_d(op2, extrsv_mpfr[j], MPFR_RNDN);
            mpfr_mul(dot, op1, op2, MPFR_RNDN);
            mpfr_add(sum, sum, dot, MPFR_RNDN);
            sum1 += a[j * n + i] * extrsv[j];
        }
        extrsv_mpfr1[i] = mpfr_get_d(sum, MPFR_RNDN);
        extrsv1[i] = sum1;
    }
    double norm02 = 0.0, val0 = 0.0, val2 = 0.0, norm12 = 0.0;
    for(uint i = 0; i < n; i++) {
        val0 = std::max(val0, fabs(x[i]));
        val2 = std::max(val2, fabs(x[i] - extrsv_mpfr1[i]));
        norm02 = std::max(norm02, fabs(x[i] - extrsv1[i]));
    }
    printf("val0 = %.16g\n", val0);
    printf("val_res_mpfr = %.16g\n", val2 / val0);
    printf("val_res_extrsv = %.16g\n", norm02 / val0);
    printf("\n\n");
    free(extrsv_mpfr1);
    free(extrsv1);
    */

    free(extrsv_mpfr);
    mpfr_free_cache();

    return nrm;
#endif
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

#if 1
    //Reading matrix A and vector b from files
    FILE *pFileA, *pFileb;
    size_t resA, resb;
    //pFileA = fopen("matrices/A_lnn_64_9.76e+08.bin", "rb");
    //pFileb = fopen("matrices/b_lnn_64_9.76e+08.bin", "rb");
    //pFileA = fopen("matrices/A_lnn_64_9.30e+13.bin", "rb");
    //pFileb = fopen("matrices/b_lnn_64_9.30e+13.bin", "rb");
    //pFileA = fopen("matrices/A_lnn_64_9.53e+21.bin", "rb");
    //pFileb = fopen("matrices/b_lnn_64_9.53e+21.bin", "rb");
    //pFileA = fopen("matrices/A_lnn_64_7.58e+40.bin", "rb");
    //pFileb = fopen("matrices/b_lnn_64_7.58e+40.bin", "rb");
    pFileA = fopen("matrices/unn/A_unn_100_3.08e+07.bin", "rb");
    pFileb = fopen("matrices/unn/b_unn_100_3.08e+07.bin", "rb");
    if ((pFileA == NULL) || (pFileb == NULL)) {
        fprintf(stderr, "Cannot open files to read matrix and vector\n");
        exit(1);
    }

    resA = fread(a, sizeof(double), n * n, pFileA);
    resb = fread(xorig, sizeof(double), n, pFileb);
    if ((resA != n * n) || (resb != n)) {
        fprintf(stderr, "Cannot read matrix and vector from files\n");
        exit(1);
    }

    fclose(pFileA);
    fclose(pFileb);
#else
    if(lognormal) {
        init_lognormal_matrix('L', 'N', a, n, mean, stddev);
        init_lognormal(xorig, n, mean, stddev);
    } else if ((argc > 6) && (argv[6][0] == 'i')) {
        init_ill_cond(a, n * n, range);
        init_ill_cond(xorig, n, range);
    } else {
        init_fpuniform_matrix('L', 'N', a, n, range, emax);
        init_fpuniform(xorig, n, range, emax);
    }
#endif
    copyVector(n, x, xorig);

    fprintf(stderr, "%d x %d\n", n, n);

    bool is_pass = true;
    double *superacc;
    double norm;
    err = posix_memalign((void **) &superacc, 64, n * sizeof(double));
    if ((!superacc) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

    copyVector(n, superacc, xorig);
    extrsv('U', 'N', 'N', n, a, n, superacc, 1, 0);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(superacc, n, a, n, xorig, 1);
    printf("Superacc error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif

    /*copyVector(n, x, xorig);
    extrsv('L', 'N', 'N', n, a, n, x, 1, 1);
#ifdef EXBLAS_VS_MPFR
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("FPE IR error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif*/

    copyVector(n, x, xorig);
    extrsv('U', 'N', 'N', n, a, n, x, 1, 3);
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
    exit(0);

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

