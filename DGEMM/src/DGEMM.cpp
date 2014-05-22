#include <ostream>
#include <stdio.h>
#include <cmath>
#include "DGEMM.hpp"

/*
 * Naive implementation of DGEMM for comparision only; it is much easy to port than the BLAS implementation
 */
void DGEMMCPU(
    double *C,
    const double *A,
    const double *B,
    const uint m,
    const uint n,
    const uint k
) {
    for(uint i = 0; i < m; i++) {
        for(uint j = 0; j < n; j++) {
            for(uint l = 0; l < k; l++) {
                C[j * m + i] += A[l * m + i] * B[j * k + l];
            }
	    //printf("%.4g\t", output[i * z + j]);
        }
	//printf("\n");
    }
}

int compare(
    const double *ref_dgemm,
    const double *dgemm,
    const uint length,
    const double epsilon
) {
    double error = 0.0;

    for(uint i = 0; i < length; ++i) 
    {
        double diff = ref_dgemm[i] - dgemm[i];
        error += pow(abs(diff), 2);
    }

    error = ::sqrt((double) error);
    printf("error = %.8g\n", error);

    return error < epsilon;
}

extern "C" bool compareDGEMMWithMPFR(const double *dgemm, const double *h_a, const double *h_b, const uint m, const uint n, const uint k) {
  double *dgemm_mpfr;
  mpfr_t sum, ddot, op1;

  dgemm_mpfr = (double *) malloc(m * n * sizeof(double));

  mpfr_init2(op1, 64);
  mpfr_init2(ddot, 128);
  mpfr_init2(sum, 2098);
  mpfr_set_d(ddot, 0.0, MPFR_RNDN);

  //Produce a result matrix of DGEMM using MPFR
  for(uint i = 0; i < m; i++) {
      for(uint j = 0; j < n; j++) {
          mpfr_set_d(sum, 0.0, MPFR_RNDN);
          for(uint l = 0; l < k; l++) {
    		mpfr_set_d(op1, h_a[l * m + i], MPFR_RNDN);
		mpfr_mul_d(ddot, op1, h_b[j * k + l], MPFR_RNDN);
		mpfr_add(sum, sum, ddot, MPFR_RNDN);
          }
	  dgemm_mpfr[j * m + i] = mpfr_get_d(sum, MPFR_RNDD);
      }
  }

  bool dgemm_cmp = false;
  double norm = 0.0;
  //Compare the GPU and MPFR results
  for (uint i = 0; i < m * n; i++) {
      norm += pow(abs(dgemm[i] - dgemm_mpfr[i]), 2);
  }
  norm = ::sqrt(norm);
  printf("Compare to MPFR. Norm = %.17g\n", norm);
  if (norm < 1e-16)
      dgemm_cmp = true;

  free(dgemm_mpfr);
  mpfr_free_cache();

  return dgemm_cmp;
}

void printMatrix(
    const double *A,
    const uint m,
    const uint n
){
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < n; j++) {
	     printf("%.4g\t", A[i * m + j]);
	}
	printf("\n");
    } 
}

