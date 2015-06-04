/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include <string.h>
#include <math.h>
#include <mpfr.h>

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

double TwoProd(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

double TwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}

/*
double OddRoundSumNonnegative(double th, double tl) {
    union {
        double d;
        long l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

int Normalize(long *accumulator, int lda, int *imin, int *imax) {
    long carry_in = accumulator[*imin * lda] >> digits;
    accumulator[*imin * lda] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i * lda] += carry_in;
        long carry_out = accumulator[i * lda] >> digits;    // Arithmetic shift
        accumulator[i * lda] -= (carry_out << digits);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax * lda] += carry_in << digits;

    return carry_in < 0;
}

double Round(long *accumulator, int lda) {
    int imin = 0;
    int imax = 38;
    int negative = Normalize(accumulator, lda, &imin, &imax);

    //Find leading word
    int i;
    //Skip zeroes
    for (i = imax; accumulator[i * lda] == 0 && i >= imin; --i) {
    }
    if (negative) {
        //Skip ones
        for (; (accumulator[i * lda] & ((1l << digits) - 1)) == ((1l << digits) - 1) && i >= imin; --i) {
        }
    }
    if (i < 0)
        //TODO: should we preserve sign of zero?
        return 0.0;

    long hiword = negative ? ((1l << digits) - 1) - accumulator[i * lda] : accumulator[i * lda];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - f_words) * digits);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long) rint(rounded);
    double mid = ldexp((double) hiword, (i - f_words) * digits);

    //Compute sticky
    long sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1l << digits) - accumulator[j * lda] : accumulator[j * lda];

    long loword = negative ? (1l << digits) - accumulator[(i - 1) * lda] : accumulator[(i - 1) * lda];
    loword |= !!sticky;
    double lo = ldexp((double) loword, (i - 1 - f_words) * digits);

    //Now add3(hi, mid, lo)
    //No overlap, we have already normalized
    if (mid != 0)
        lo = OddRoundSumNonnegative(mid, lo);

    //Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}
*/

double OddRoundSum(double *fpe) {
    double th, tl;
    union {
        double d;
        long l;
    } thdb;

    th = TwoSum(fpe[1], fpe[0], &tl);

    // round to odd th if tl is not zero
    if (tl != 0.0) {
        thdb.d = th;
        // if the mantissa of th is odd, there is nothing to do
        if (!(thdb.l & 1)) {
            // choose the rounding direction
            // depending on the signs of th and tl
            if ((tl > 0.0) ^ (th < 0.0))
                thdb.l++;
            else
                thdb.l--;
            thdb.d = th;
        }
    }

    // final addition rounder to nearest
    return fpe[2] + th;
}

static void __extrsv(uint n, double *a, double *x) {
    double r, s, z;
    for (uint i = 0; i < n; i++) {
        double fpe[3] = {0.0};

        for(uint j = 0; j < i; j++) {
            r = TwoProd(a[j * n + i], -x[j], &s);

            fpe[2] = TwoSum(fpe[2], r, &z);
            if (z != 0.0) {
                fpe[1] = TwoSum(fpe[1], z, &z);
                if (z != 0.0) {
                    fpe[0] = TwoSum(fpe[0], z, &z);
                }
            }

            if (s != 0.0) {
                fpe[2] = TwoSum(fpe[2], s, &z);
                if (z != 0.0) {
                    fpe[1] = TwoSum(fpe[1], z, &z);
                    if (z != 0.0) {
                        fpe[0] = TwoSum(fpe[0], z, &z);
                    }
                }
            }
        }

        fpe[2] = TwoSum(fpe[2], x[i], &z);
        if (z != 0.0) {
            fpe[1] = TwoSum(fpe[1], z, &z);
            if (z != 0.0) {
                fpe[0] = TwoSum(fpe[0], z, &z);
            }
        }

        double sum = OddRoundSum(fpe);
        x[i] = sum / a[i * (n + 1)];
    }
}

static double extrsvVsMPFR(double *extrsv, uint n, double *a, uint lda, double *x, uint incx) {
    // Compare to the results from Matlab
    FILE *pFilex;
    size_t resx;
    pFilex = fopen("../matrices/x_test_trsv_64.bin", "rb");
    //pFilex = fopen("x_test_trsv_64_final.bin", "rb");
    //pFilex = fopen("x_test_gemv_64.bin", "rb");
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

    /*for(uint i = 0; i < n; i++)
        printf("%.16g\t", extrsv[i]);
    printf("\n\n");*/

    printf("err = [");
    for(uint i = 0; i < 63; i++) {
        printf("%.16g,\t", extrsv[i]);
        //if ((i+1) % 2 == 0)
        //    printf(";");
    }
    printf("];");
    printf("\n\n");

    //Inf norm
    double nrm2 = 0.0, val2 = 0.0;
    for(uint i = 0; i < n; i++) {
        val2 = val2 > fabs(xmatlab[i]) ? val2 : fabs(xmatlab[i]);
        nrm2 = nrm2 > fabs(extrsv[i] - xmatlab[i]) ? nrm2 : fabs(extrsv[i] - xmatlab[i]);
        //if (fabs(extrsv[i] - xmatlab[i]) != 0.0)
        //    printf("\n %d \t", i);
        printf("%.16g\t", fabs(extrsv[i] - xmatlab[i]));
    }
    printf("\n\n");
    printf("ExTRSV vs Matlab = %.16g \t %.16g\n", nrm2, val2);
    nrm2 = nrm2 / val2;
    printf("ExTRSV vs Matlab = %.16g\n", nrm2);

    return nrm2;

    mpfr_t sum, dot, div, op1, op2;

    double *extrsv_mpfr = (double *) malloc(n * sizeof(double));
    copyVector(n, extrsv_mpfr, x);

    mpfr_init2(op1, 53);
    mpfr_init2(op2, 53);
    mpfr_init2(dot, 106);
    mpfr_init2(div, 2098);
    mpfr_init2(sum, 2098);

    //Produce a result matrix of TRSV using MPFR
    for(uint i = 0; i < n; i++) {
        // sum += a[i,j] * x[j], j < i
        mpfr_set_d(sum, extrsv_mpfr[i], MPFR_RNDN);
        for(uint j = 0; j < i; j++) {
            mpfr_set_d(op1, a[j * n + i], MPFR_RNDN);
            mpfr_set_d(op2, extrsv_mpfr[j], MPFR_RNDN);
            mpfr_mul(dot, op1, op2, MPFR_RNDN);
            mpfr_sub(sum, sum, dot, MPFR_RNDN);
        }
        // x[i] = sum / a[i,i]
        mpfr_set_d(op1, a[i * (n + 1)], MPFR_RNDN);
        mpfr_div(div, sum, op1, MPFR_RNDN);
        extrsv_mpfr[i] = mpfr_get_d(div, MPFR_RNDN);
    }
    for(uint i = 0; i < n; i++) {
        printf("%.16g\t", extrsv_mpfr[i]);
    }
    printf("\n\n");

    //naive trsv
    double *trsvn = (double *) malloc(n * sizeof(double));
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
    double nrm = 0.0, val = 0.0;
    for(uint i = 0; i < n; i++) {
        nrm += pow(fabs(extrsv[i] - extrsv_mpfr[i]), 2);
        val += pow(fabs(extrsv_mpfr[i]), 2);
    }
    nrm = ::sqrt(nrm) / ::sqrt(val);
#else
    //Inf norm
    double nrm = 0.0, val = 0.0, nrm1 = 0.0;
    for(uint i = 0; i < n; i++) {
        val = val > fabs(extrsv_mpfr[i]) ? val : fabs(extrsv_mpfr[i]);
        nrm = nrm > fabs(extrsv[i] - extrsv_mpfr[i]) ? nrm : fabs(extrsv[i] - extrsv_mpfr[i]);
        nrm1 = nrm1 > fabs(trsvn[i] - extrsv_mpfr[i]) ? nrm1 : fabs(trsvn[i] - extrsv_mpfr[i]);
    }
    nrm = nrm / val;
    nrm1 = nrm1 / val;
    //printf("nrm1 = %.16g\t", nrm1);
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
}


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
    pFileA = fopen("/home/iakymchuk/workspace/exblas/tests/matrices/A_lnn_64_9.76e+08.bin", "rb");
    pFileb = fopen("/home/iakymchuk/workspace/exblas/tests/matrices/b_lnn_64_9.76e+08.bin", "rb");
    //pFileA = fopen("matrices/A_lnn_64_9.30e+13.bin", "rb");
    //pFileb = fopen("matrices/b_lnn_64_9.30e+13.bin", "rb");
    //pFileA = fopen("matrices/A_lnn_64_9.53e+21.bin", "rb");
    //pFileb = fopen("matrices/b_lnn_64_9.53e+21.bin", "rb");
    //pFileA = fopen("matrices/A_lnn_64_7.58e+40.bin", "rb");
    //pFileb = fopen("matrices/b_lnn_64_7.58e+40.bin", "rb");
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
    double norm;

    __extrsv(n, a, x);
    norm = extrsvVsMPFR(x, n, a, n, xorig, 1);
    printf("ExTRSV on CPU\nError = %.16g\n", norm);
    if (norm > eps) {
        is_pass = false;
    }

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}

