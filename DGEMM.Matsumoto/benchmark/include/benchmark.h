#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __cplusplus
extern "C" {
#endif

#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))

#define TRUE  (1)
#define FALSE (0)

#define D_NEG_ONE (-1)
#define S_NEG_ONE (-1)

#define D_THRESHOLD (1e-5)
#define S_THRESHOLD (1e-3)

#ifndef FORTRAN_NAME
#if defined(ADD_)
#define FORTRAN_NAME(lcname, UCNAME)  lcname##_
#elif defined(NOCHANGE)
#define FORTRAN_NAME(lcname, UCNAME)  lcname
#elif defined(UPCASE)
#define FORTRAN_NAME(lcname, UCNAME)  UCNAME
#else
#error Define one of ADD_, NOCHANGE, or UPCASE for how Fortran functions are name mangled.
#endif
#endif

#define blasf77_daxpy      FORTRAN_NAME( daxpy,  DAXPY  )
#define blasf77_saxpy      FORTRAN_NAME( saxpy,  SAXPY  )
#define lapackf77_dlange   FORTRAN_NAME( dlange, DLANGE )
#define lapackf77_dlarnv   FORTRAN_NAME( dlarnv, DLARNV )
#define lapackf77_slange   FORTRAN_NAME( slange, SLANGE )
#define lapackf77_slarnv   FORTRAN_NAME( slarnv, SLARNV )

#if !defined(BENCHMARK_ACML)
void blasf77_daxpy(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);
void blasf77_saxpy(const int *n, const float *alpha, const float *x, const int *incx, float *y, const int *incy);
double lapackf77_dlange(const char *norm, const int *m, const int *n, const double *A, const int *lda, double *work);
void lapackf77_dlarnv(const int *idist, int *iseed, const int *n, double *x);
float lapackf77_slange(const char *norm, const int *m, const int *n, const float *A, const int *lda, float *work);
void lapackf77_slarnv(const int *idist, int *iseed, const int *n, float *x);
#endif

double oclblas_get_current_time(void);

int oclblas_dmake(char *type, char uplo, char diag, int m, int n, double *a, int nmax, double *aa, int lda, int *reset, double transl);
void oclblas_dprint(int m, int n, double *a, int lda, char *label);

int oclblas_smake(char *type, char uplo, char diag, int m, int n, float *a, int nmax, float *aa, int lda, int *reset, float transl);
void oclblas_sprint(int m, int n, float *a, int lda, char *label);

#ifdef __cplusplus
}
#endif

#endif // ifndef BENCHMARK_H_
