#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "benchmark.h"

/*  oclblas_dbeg
 *
 *  Generates random numbers uniformly distributed between -0.5 and 0.5.

 *  Auxiliary routine for test program for Level 3 Blas.
 
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 */
double oclblas_dbeg(int *reset)
{
  if (*reset == TRUE) {
    srand(481);
    *reset = FALSE;
  }
  return (double)((double)rand()/RAND_MAX);
} /* End of dbeg function */

/*  oclblas_dmake
 *
 *  Generates values for an M by N matrix A.
 *  Stores the values in the array AA in the data structure required
 *  by the routine, with unwanted elements set to rogue value.

 *  TYPE is 'GE', 'SY' or 'TR'.

 *  Auxiliary routine for test program for Level 3 Blas.

 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 */
int
oclblas_dmake(
    char *type, char uplo, char diag, int m, 
    int n, double *a, int nmax, double *aa, int lda,
    int *reset, double transl
    )
{
  /* Local variables */
  static int i, j;
  static int gen, tri, sym;
  static int ibeg, iend;
  static int unit, lower, upper;

  /* Parameter adjustments */
  a -= 1 + nmax;
  --aa;

  /* Function Body */
  gen = strcmp(type, "GE") == 0;
  sym = strcmp(type, "SY") == 0;
  tri = strcmp(type, "TR") == 0;
  upper = (sym || tri) && (uplo == 'U');
  lower = (sym || tri) && (uplo == 'L');
  unit = tri && (diag == 'U');

  /* Generate data in array A. */
  for (j = 1; j <= n; ++j) {
    for (i = 1; i <= m; ++i) {
      if (gen || (upper && i <= j) || (lower && i >= j)) {
        a[i + j * nmax] = oclblas_dbeg(reset) + transl;
        if (i != j) {
          /* Set some elements to zero */
          if (n > 3 && j == n / 2) {
            a[i + j * nmax] = 0.;
          }
          if (sym) {
            a[j + i * nmax] = a[i + j * nmax];
          } else if (tri) {
            a[j + i * nmax] = 0.;
          }
        }
      }
    }
    if (tri) {
      a[j + j * nmax] += 1.;
    }
    if (unit) {
      a[j + j * nmax] = 1.;
    }
  }

  /* Store elements in array AS in data structure required by routine. */
  if (gen) {
    for (j = 1; j <= n; ++j) {
      for (i = 1; i <= m; ++i) {
        aa[i + (j - 1) * lda] = a[i + j * nmax];
      }
      for (i = m + 1; i <= lda; ++i) {
        aa[i + (j - 1) * lda] = -1e10;
      }
    }
  } else if (sym || tri) {
    for (j = 1; j <= n; ++j) {
      if (upper) {
        ibeg = 1;
        if (unit) {
          iend = j - 1;
        } else {
          iend = j;
        }
      } else {
        if (unit) {
          ibeg = j + 1;
        } else {
          ibeg = j;
        }
        iend = n;
      }
      for (i = 1; i <= ibeg-1; ++i) {
        aa[i + (j - 1) * lda] = -1e10;
      }
      for (i = ibeg; i <= iend; ++i) {
        aa[i + (j - 1) * lda] = a[i + j * nmax];
      }
      for (i = iend + 1; i <= lda; ++i) {
        aa[i + (j - 1) * lda] = -1e10;
      }
    }
  }
  return 0;
} /* End of oclblas_dmake function */

void oclblas_dprint(int m, int n, double *a, int lda, char *label)
{
  printf("##### START PRINTING %dx%d MATRIX %s #####\n", m, n, label);
  int i, j;
  for (i = 0; i < m; i++) {
    printf("%3d:", i);
    for (j = 0; j < n; j++) {
      if (a[i + j*lda] == -1e10)
        printf(" %8s", "-");
      else
        printf(" %8.4lf", a[i + j*lda]);
    }
    printf("\n");
  }
  printf("##### END PRINTING %dx%d MATRIX %s #######\n", m, n, label);
}
