
#include <cmath>
#include <gmp.h>
#include <mpfr.h>
#include "common.hpp"

////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
inline double randDouble(int emin, int emax, int neg_ratio) {
    // Uniform mantissa
    double x = double(rand()) / double(RAND_MAX * .99) + 1.;
    // Uniform exponent
    int e = (rand() % (emax - emin)) + emin;
    // Sign
    if(neg_ratio > 1 && rand() % neg_ratio == 0)
        x = -x;

    return ldexp(x, e);
}

double min(double arr[], int size) {
    assert(arr != NULL);
    assert(size >= 0);

    if ((arr == NULL) || (size <= 0))
       return NAN;

    double val = DBL_MAX; 
    for (int i = 0; i < size; i++)
        if (val > arr[i])
            val = arr[i];

    return val;
}

void init_fpuniform(double *a, const uint n, const int range, const int emax)
{
    //Generate numbers on several bins starting from emax
    for(uint i = 0; i != n; ++i) {
        //a[i] = randDouble(emax-range, emax, 1);
        a[i] = randDouble(0, range, 1);
    }
    /*//Generate numbers on an interval [1, 2]
    for(uint i = 0; i != n; ++i) {
        a[i] = 1.0 + double(rand()) / double(RAND_MAX);
    }*/
}

/*
 * uint lower triangular
 */
void init_fpuniform_lu_matrix(double *a, const uint n, const int range, const int emax)
{
    //Generate numbers on several bins starting from emax
    for(uint j = 0; j < n; ++j)
        for(uint i = 0; i < n; ++i)
            if (j < i)
                a[j * n + i] = randDouble(0, range, 1);
            else if (j == i)
                a[i * (n + 1)] = randDouble(0, range, 1) * 100;
            else
                a[j * n + i] = 0.0;
}

/*
 * non-uint upper triangular
 */
void init_fpuniform_un_matrix(double *a, const uint n, const int range, const int emax)
{
    //Generate numbers on several bins starting from emax
    for(uint i = 0; i < n; ++i)
        for(uint j = 0; j < n; ++j)
            if (j >= i)
                a[i * n + j] = randDouble(0, range, 1);
            else
                a[i * n + j] = 0.0;
}

inline void linspace(double *b, double x, double y, uint n) {
    double val;
    double h = (y - x) / (n - 1);

    val = x;
    for (uint i = 0; i < n; ++i) {
        b[i] = val;
        val += h;
    }
}

inline double TwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}

inline void VecSum(double *p, double *t, uint n) {
    double s[n];

    s[0] = p[0];
    for (uint i = 1; i < n; i++)
        s[i] = TwoSum(s[i-1], p[i], &t[i-1]);
    t[n-1] = s[n-1];
}

extern "C" void generate_ill_cond_system(
    int islower,
    double *A,
    double *b,
    uint n,
    const double c
){
    double *U, *tmp1, *tmp2;
    U = (double *) calloc(n * n, sizeof(double));
    tmp1 = (double *) calloc(n, sizeof(double));

    uint p = (n + 1) / 2; // n = 2*p - 1
    tmp2 = (double *) calloc(p, sizeof(double));

    // Si n est impair, on se ramene au cas pair
    if (!(n % 2)) {
        A[(n + 1) * (n - 1)] = 1.0;
        b[n] = 1.0;
        //n = n - 1;
    }

    // On commence par generer A(1:p, 1:n) et b(1:p)
    //D = diag((ones(1,p) - 2*rand(1,p)) .* linspace(pow(10,-c), pow(10, c), p));
    init_fpuniform(tmp1, p, 1, 1);
    for (uint i = 0; i < p; i++)
        tmp1[i] = 1.0 - 2 * tmp1[i];
    linspace(tmp2, pow(10,-c), pow(10,c), p);

    //U = triu((1 - 2*rand(p)) .* (10.^round(c*(1 - 2*rand(p)))), 1);
    init_fpuniform(U, n * n, 1, 1);
    for (uint i = 0; i < p; i++)
        for (uint j = 0; j < p; j++)
            if (islower) {
                if (j < i)
                    U[j * n + i] = (1 - 2 * U[j * n + i]) * (pow(10,round(c * (1 - 2 * U[j * n + i]))));
            } else {
                if (j > i)
                    U[i * n + j] = (1 - 2 * U[i * n + j]) * (pow(10,round(c * (1 - 2 * U[i * n + j]))));
            }

    for (uint i = 0; i < p; i++)
        for (uint j = 0; j < p; j++) {
            if (islower) {
                if (j < i)
                    A[j * n + i] = U[j * n + i];
            } else {
                if (j > i)
                    A[i * n + j] = U[i * n + j];
            }
            if (i == j)
                A[i * (n + 1)] = tmp1[i] * tmp2[i];
        }

    // A l'aide de l'algorithme VecSum, on calcule A(1:p, p+1:n) et b(1:p)
    // de maniere a ce que l'on aie exactement sum(A(i:)) = b(i)
    if (islower)
        // assume column-wise storage, which is actually is
        for (uint i = 0; i < p; i++) {
            VecSum(&A[i * n], tmp1, p);
            for (uint j = 0; j < p-1; j++)
                A[j * n + p + i] = -tmp1[j];
            b[i] = tmp1[p-1];
        }
    else
        // assume row-wise storage, which is actually is
        for (uint i = 0; i < p; i++) {
            VecSum(&A[i * n], tmp1, p);
            for (uint j = 0; j < p-1; j++)
                A[i * n + p + j] = -tmp1[j];
            b[i] = tmp1[p-1];
        }
    // On genere maintenant A(p+1:n,p+1:n) et b(p+1:n)
    // A(p+1:n,p+1:n) est generee aleatoirement avec des coefficient 
    // compris entre -1 et 1
    //A(p+1:n,p+1:n) = triu(ones(p-1) - 2*rand(p-1));
    for (uint i = p; i < n-1; i++)
        for (uint j = p; j < n-1; j++)
            if (islower) {
                if (j <= i)
                    A[j * n + i] = 1 - 2 * U[i * n + j];
            } else {
                if (j >= i)
                    A[i * n + j] = 1 - 2 * U[i * n + j];
            }
    // b(p+1:n) est le resultat du produit A(p+1:n,p+1:n) * ones(p-1,1)
    // calcule avec une grande precision
    mpfr_t sum;
    mpfr_init2(sum, 256);
    for (uint i = p; i < n; i++) {
        mpfr_set_d(sum, 0.0, MPFR_RNDN);

        for (uint j = p+1; j < n; j++)
            mpfr_add_d(sum, sum, A[i * n + j], MPFR_RNDN);

        b[i] = mpfr_get_d(sum, MPFR_RNDN);
    }

    mpfr_free_cache();
    free(U);
    free(tmp1);
    free(tmp2);
}


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
extern "C" double KnuthTwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}

extern "C" double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

extern "C" void printMatrix(
    const int iscolumnwise,
    const double *A,
    const uint lda,
    const uint n
){
    printf("[ ");
    for (uint i = 0; i < n; i++) {
        for (uint j = 0; j < n; j++)
            if (iscolumnwise)
                printf("%.4g, ", A[j * lda + i]);
            else
                printf("%.4g, ", A[i * lda + j]);
        printf(";\n");
    }
    printf("]\n");
}

extern "C" void printVector(
    const double *a,
    const uint n
){
    for (uint i = 0; i < n; i++)
        printf("%.4g\t", a[i]);
    printf("\n");
}

