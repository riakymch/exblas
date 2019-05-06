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

// matrix is stored in column-major order
static double exgemvVsMPFR(const bool iscolumnwise, char trans, const double *exgemv, int m, int n, double alpha, const double *a, uint lda, const double *x, uint incx, double beta, const double *y, uint incy) {
    mpfr_t sum, dot;

    double *exgemv_mpfr = (double *) malloc((trans == 'T' ? n : m) * sizeof(double));

    mpfr_init2(dot, 128);
    mpfr_init2(sum, 2098);

	if (trans == 'T') {
        for(int j = 0; j < n; j++) {
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
        	for(int i = 0; i < m; i++) {
                if (iscolumnwise)
                    mpfr_set_d(dot, a[j * lda + i], MPFR_RNDN);
                else
                    mpfr_set_d(dot, a[i * lda + j], MPFR_RNDN);
                mpfr_mul_d(dot, dot, alpha, MPFR_RNDN);
                mpfr_mul_d(dot, dot, x[i], MPFR_RNDN);
                mpfr_add(sum, sum, dot, MPFR_RNDN);
            }
            mpfr_set_d(dot, y[j], MPFR_RNDN);
            mpfr_mul_d(dot, dot, beta, MPFR_RNDN);
            mpfr_add(sum, sum, dot, MPFR_RNDN);
            exgemv_mpfr[j] = mpfr_get_d(sum, MPFR_RNDN);
        }
	} else {
        for(int i = 0; i < m; i++) {
            mpfr_set_d(sum, 0.0, MPFR_RNDN);
            for(int j = 0; j < n; j++) {
                if (iscolumnwise)
                    mpfr_set_d(dot, a[j * lda + i], MPFR_RNDN);
                else
                    mpfr_set_d(dot, a[i * lda + j], MPFR_RNDN);
                mpfr_mul_d(dot, dot, alpha, MPFR_RNDN);
                mpfr_mul_d(dot, dot, x[j], MPFR_RNDN);
                mpfr_add(sum, sum, dot, MPFR_RNDN);
            }
            mpfr_set_d(dot, y[i], MPFR_RNDN);
            mpfr_mul_d(dot, dot, beta, MPFR_RNDN);
            mpfr_add(sum, sum, dot, MPFR_RNDN);
            exgemv_mpfr[i] = mpfr_get_d(sum, MPFR_RNDN);
        }
    }

    //compare the GPU and MPFR results
#if 0
    //L2 norm
    double nrm = 0.0, val = 0.0;
    for(uint i = 0; i < n; i++) {
        nrm += pow(fabs(exgemv[i] - exgemv_mpfr[i]), 2);
        val += pow(fabs(exgemv_mpfr[i]), 2);
    }
    nrm = ::sqrt(nrm) / ::sqrt(val);
#else
    //Inf norm
    m = trans == 'T' ? n : m;
    double nrm = 0.0, val = 0.0;
    for(int i = 0; i < m; i++) {
        val = std::max(val, fabs(exgemv_mpfr[i]));
        nrm = std::max(nrm, fabs(exgemv[i] - exgemv_mpfr[i]));
    }
    nrm = nrm / val;
#endif

    free(exgemv_mpfr);
    mpfr_free_cache();

    return nrm;
}

#else
static double exgemvVsSuperacc(uint m, double *exgemv, double *superacc) {
    double nrm = 0.0, val = 0.0;
    for (uint i = 0; i < m; i++) {
        nrm += pow(fabs(exgemv[i] - superacc[i]), 2);
        val += pow(fabs(superacc[i]), 2);
    }
    nrm = ::sqrt(nrm) / ::sqrt(val);

    return nrm;
}
#endif

static void copyVector(uint n, double *x, const double *y) {
    for (uint i = 0; i < n; i++)
        x[i] = y[i];
}


int main(int argc, char *argv[]) {
    char trans = 'N';
    uint m = 256, n = 256;
    bool iscolumnwise = true;
    bool lognormal = false;

    if(argc > 1)
        trans = argv[1][0];
    if(argc > 2)
        m = atoi(argv[2]);
    if(argc > 3)
        n = atoi(argv[3]);
    if(argc > 6) {
        if(argv[6][0] == 'n') {
            lognormal = true;
        }
    }
    int lda = m;

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
    double alpha = 1.0, beta = 1.0;
    double *a, *x, *y, *yorig;
    int err = posix_memalign((void **) &a, 64, m * n * sizeof(double));
    err &= posix_memalign((void **) &x, 64, ((trans == 'T') ? m : n) * sizeof(double));
    err &= posix_memalign((void **) &y, 64, ((trans == 'T') ? n : m) * sizeof(double));
    err &= posix_memalign((void **) &yorig, 64, ((trans == 'T') ? n : m) * sizeof(double));
    if ((!a) || (!x) || (!y) || (!yorig) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

    if(lognormal) {
        printf("init_lognormal_matrix\n");
        init_lognormal_matrix(iscolumnwise, m, n, a, lda, mean, stddev);
        init_lognormal((trans == 'T') ? m : n, x, mean, stddev);
        init_lognormal((trans == 'T') ? n : m, yorig, mean, stddev);
    } else if ((argc > 6) && (argv[6][0] == 'i')) {
        printf("init_ill_cond\n");
        init_ill_cond(m * n, a, range);
        init_ill_cond((trans == 'T') ? m : n, x, range);
        init_ill_cond((trans == 'T') ? n : m, yorig, range);
    } else {
        printf("init_fpuniform_matrix\n");
        init_fpuniform_matrix(iscolumnwise, m, n, a, lda, range, emax);
        init_fpuniform((trans == 'T') ? m : n, x, range, emax);
        init_fpuniform((trans == 'T') ? n : m, yorig, range, emax);
    }
    copyVector((trans == 'T') ? n : m, y, yorig);

    fprintf(stderr, "%d %d ", m, n);

    if(lognormal) {
        fprintf(stderr, "%f ", stddev);
    } else {
        fprintf(stderr, "%d ", range);
    }

    bool is_pass = true;
    double *superacc;
    double norm;
    err &= posix_memalign((void **) &superacc, 64, ((trans == 'T') ? n : m) * sizeof(double));
    if ((!superacc) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");

    // DGEMV on GPU
    copyVector((trans == 'T') ? n : m, superacc, yorig);
    exgemv(trans, m, n, alpha, a, lda, 0, x, 1, 0, beta, superacc, 1, 0, 1);
#ifdef EXBLAS_VS_MPFR
    norm = exgemvVsMPFR(iscolumnwise, trans, superacc, m, n, alpha, a, lda, x, 1, beta, yorig, 1);
    printf("DGEMV error = %.16g\n", norm);
#endif

    copyVector((trans == 'T') ? n : m, superacc, yorig);
    exgemv(trans, m, n, alpha, a, lda, 0, x, 1, 0, beta, superacc, 1, 0, 0);
#ifdef EXBLAS_VS_MPFR
    norm = exgemvVsMPFR(iscolumnwise, trans, superacc, m, n, alpha, a, lda, x, 1, beta, yorig, 1);
    printf("Superacc error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }
#endif

    copyVector((trans == 'T') ? n : m, y, yorig);
    exgemv(trans, m, n, alpha, a, lda, 0, x, 1, 0, beta, y, 1, 0, 3);
#ifdef EXBLAS_VS_MPFR
    norm = exgemvVsMPFR(iscolumnwise, trans, y, m, n, alpha, a, lda, x, 1, beta, yorig, 1);
#else
    norm = exgemvVsSuperacc((trans == 'T') ? n : m, y, superacc);
#endif
    printf("FPE3 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector((trans == 'T') ? n : m, y, yorig);
    exgemv(trans, m, n, alpha, a, lda, 0, x, 1, 0, beta, y, 1, 0, 4);
#ifdef EXBLAS_VS_MPFR
    norm = exgemvVsMPFR(iscolumnwise, trans, y, m, n, alpha, a, lda, x, 1, beta, yorig, 1);
#else
    norm = exgemvVsSuperacc((trans == 'T') ? n : m, y, superacc);
#endif
    printf("FPE4 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector((trans == 'T') ? n : m, y, yorig);
    exgemv(trans, m, n, alpha, a, lda, 0, x, 1, 0, beta, y, 1, 0, 8);
#ifdef EXBLAS_VS_MPFR
    norm = exgemvVsMPFR(iscolumnwise, trans, y, m, n, alpha, a, lda, x, 1, beta, yorig, 1);
#else
    norm = exgemvVsSuperacc((trans == 'T') ? n : m, y, superacc);
#endif
    printf("FPE8 error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector((trans == 'T') ? n : m, y, yorig);
    exgemv(trans, m, n, alpha, a, lda, 0, x, 1, 0, beta, y, 1, 0, 4, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemvVsMPFR(iscolumnwise, trans, y, m, n, alpha, a, lda, x, 1, beta, yorig, 1);
#else
    norm = exgemvVsSuperacc((trans == 'T') ? n : m, y, superacc);
#endif
    printf("FPE4EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector((trans == 'T') ? n : m, y, yorig);
    exgemv(trans, m, n, alpha, a, lda, 0, x, 1, 0, beta, y, 1, 0, 6, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemvVsMPFR(iscolumnwise, trans, y, m, n, alpha, a, lda, x, 1, beta, yorig, 1);
#else
    norm = exgemvVsSuperacc((trans == 'T') ? n : m, y, superacc);
#endif
    printf("FPE6EE error = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    copyVector((trans == 'T') ? n : m, y, yorig);
    exgemv(trans, m, n, alpha, a, lda, 0, x, 1, 0, beta, y, 1, 0, 8, true);
#ifdef EXBLAS_VS_MPFR
    norm = exgemvVsMPFR(iscolumnwise, trans, y, m, n, alpha, a, lda, x, 1, beta, yorig, 1);
#else
    norm = exgemvVsSuperacc((trans == 'T') ? n : m, y, superacc);
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

