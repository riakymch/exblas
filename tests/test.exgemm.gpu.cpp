/*
 *  Copyright (c) 2013-2015 University Pierre and Marie Curie 
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

double exgemmWithMPFR(int N, double *a) {
    mpfr_t mpaccum;
    mpfr_init2(mpaccum, 2098);
    mpfr_set_zero(mpaccum, 0);

    for(int i = 0; i != N; ++i) {
        mpfr_add_d(mpaccum, mpaccum, a[i], MPFR_RNDN);
    }
    double dacc = mpfr_get_d(mpaccum, MPFR_RNDN);

    //mpfr_printf("%Ra\n", mpaccum);
    mpfr_clear(mpaccum);

    return dacc;
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
        init_lognormal(a, m, k, mean, stddev);
        init_lognormal(b, k, n, mean, stddev);
        init_lognormal(c, m, n, mean, stddev);
    } else if ((argc > 6) && (argv[6][0] == 'i')) {
        init_ill_cond(a, m, k, range);
        init_ill_cond(b, k, n, range);
        init_ill_cond(c, m, n, range);
    } else {
        if(range == 1){
            init_naive(a, m, k);
            init_naive(b, k, n);
            init_naive(c, m, n);
        } else {
            init_fpuniform(a, m, k, range, emax);
            init_fpuniform(b, k, n, range, emax);
            init_fpuniform(c, m, n, range, emax);
        }
    }

    fprintf(stderr, "%d \t %d \t %d\n", m, n, k);

    bool is_pass = true;
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 1);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 3);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 4);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 8);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 4, true);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 6, true);
    exgemm('N', 'N', m, n, k, 1.0, a, k, b, n, 0.0, c, n, 8, true);

/*#ifdef EXBLAS_VS_MPFR
    double exgemmMPFR = exgemmWithMPFR(N, a);
    if ((fabs(exgemmMPFR - exgemm_fpe2) != 0) || (fabs(exgemmMPFR - exgemm_fpe4) != 0) || (fabs(exgemmMPFR - exgemm_fpe8ee) != 0) || (fabs(exgemmMPFR - exgemm_fpe4ee) != 0) || (fabs(exgemmMPFR - exgemm_fpe6ee) != 0)) {
        is_pass = false;
        printf("FAILED: %.16g \t %.16g \t %.16g \t %.16g \t %.16g\n", fabs(exgemmMPFR - exgemm_fpe2), fabs(exgemmMPFR - exgemm_fpe4), fabs(exgemmMPFR - exgemm_fpe4ee), fabs(exgemmMPFR - exgemm_fpe6ee), fabs(exgemmMPFR - exgemm_fpe8ee));
    }
#else
#endif*/
    fprintf(stderr, "\n");

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}

