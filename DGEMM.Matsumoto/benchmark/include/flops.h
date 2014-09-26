/* flops.h
 *
 * File originally provided by Univ. of Tennessee,
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @date 2010-12-20
 *
 * This file provide the flops formula for all Level 3 BLAS and some
 * Lapack routines.  Each macro uses the same size parameters as the
 * function associated and provide one formula for additions and one
 * for multiplications. Example to use these macros:
 *
 *    FLOPS_ZGEMM( m, n, k )
 *
 * All the formula are reported in the LAPACK Lawn 41:
 *     http://www.netlib.org/lapack/lawns/lawn41.ps
 */
#ifndef FLOPS_H_
#define FLOPS_H_

/*
 * Level 3 BLAS
 */
#define FMULS_GEMM(__m, __n, __k) ((__m) * (__n) * (__k))
#define FADDS_GEMM(__m, __n, __k) ((__m) * (__n) * (__k))

#define FMULS_SYMM(__side, __m, __n) ( ( (__side) == CblasLeft ) ? FMULS_GEMM((__m), (__m), (__n)) : FMULS_GEMM((__m), (__n), (__n)) )
#define FADDS_SYMM(__side, __m, __n) ( ( (__side) == CblasLeft ) ? FADDS_GEMM((__m), (__m), (__n)) : FADDS_GEMM((__m), (__n), (__n)) )

#define FMULS_SYRK(__k, __n) (0.5 * (__k) * (__n) * ((__n)+1))
#define FADDS_SYRK(__k, __n) (0.5 * (__k) * (__n) * ((__n)+1))

#define FMULS_SYR2K(__k, __n) ((__k) * (__n) * (__n)        )
#define FADDS_SYR2K(__k, __n) ((__k) * (__n) * (__n) + (__n))

#define FMULS_TRMM_2(__m, __n) (0.5 * (__n) * (__m) * ((__m)+1))
#define FADDS_TRMM_2(__m, __n) (0.5 * (__n) * (__m) * ((__m)-1))

#define FMULS_TRMM(__side, __m, __n) ( ( (__side) == 'L' ) ? FMULS_TRMM_2((__m), (__n)) : FMULS_TRMM_2((__n), (__m)) )
#define FADDS_TRMM(__side, __m, __n) ( ( (__side) == 'L' ) ? FADDS_TRMM_2((__m), (__n)) : FADDS_TRMM_2((__n), (__m)) )

#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FMULS_TRMM


#define FLOPS_DGEMM(__m, __n, __k) (     FMULS_GEMM((double)(__m), (double)(__n), (double)(__k)) +       FADDS_GEMM((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SGEMM(__m, __n, __k) (     FMULS_GEMM((double)(__m), (double)(__n), (double)(__k)) +       FADDS_GEMM((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_DSYMM(__side, __m, __n) (     FMULS_SYMM(__side, (double)(__m), (double)(__n)) +       FADDS_SYMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_SSYMM(__side, __m, __n) (     FMULS_SYMM(__side, (double)(__m), (double)(__n)) +       FADDS_SYMM(__side, (double)(__m), (double)(__n)) )

#define FLOPS_DSYRK(__k, __n) (     FMULS_SYRK((double)(__k), (double)(__n)) +       FADDS_SYRK((double)(__k), (double)(__n)) )
#define FLOPS_SSYRK(__k, __n) (     FMULS_SYRK((double)(__k), (double)(__n)) +       FADDS_SYRK((double)(__k), (double)(__n)) )

#define FLOPS_DSYR2K(__k, __n) (     FMULS_SYR2K((double)(__k), (double)(__n)) +       FADDS_SYR2K((double)(__k), (double)(__n)) )
#define FLOPS_SSYR2K(__k, __n) (     FMULS_SYR2K((double)(__k), (double)(__n)) +       FADDS_SYR2K((double)(__k), (double)(__n)) )

#define FLOPS_DTRMM(__side, __m, __n) (     FMULS_TRMM(__side, (double)(__m), (double)(__n)) +       FADDS_TRMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_STRMM(__side, __m, __n) (     FMULS_TRMM(__side, (double)(__m), (double)(__n)) +       FADDS_TRMM(__side, (double)(__m), (double)(__n)) )

#define FLOPS_DTRSM(__side, __m, __n) (     FMULS_TRSM(__side, (double)(__m), (double)(__n)) +       FADDS_TRSM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_STRSM(__side, __m, __n) (     FMULS_TRSM(__side, (double)(__m), (double)(__n)) +       FADDS_TRSM(__side, (double)(__m), (double)(__n)) )

#endif // #ifndef FLOPS_H_
