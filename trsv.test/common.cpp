
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

void generate_ill_cond_system(double *a, double *b, const int n, const double c) {

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

