#include "DDOT.hpp"

/*
 * Naive implementation of DDOT for comparision only; it is much easy to port than the BLAS implementation
 */
double DDOT_CPU(
    double *a,
    double *b,
    const unsigned int n
) {
    double res = 0.0;
    for(unsigned int i = 0; i < n; i++)
        res += a[i] * b[i];

    return res;
}
