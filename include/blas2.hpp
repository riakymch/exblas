/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file blas2.hpp
 *  \brief Provides a set of Basic Linear Algera Subprograms level-3 to handle matrix-vector operations
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef BLAS2_HPP_
#define BLAS2_HPP_

// config from cmake
#include "config.h"

/**
 * \defgroup blas2 BLAS Level-2 Functions
 */

/**
 * \defgroup ExTRSV TRSV Functions
 * \ingroup blas2
 */

/**
 * \ingroup ExTRSV
 * \brief ExTRSV solves one of the systems of equations
 *
 *      A*x = b,   or   A**T*x = b,
 *
 *  using our multi-level reproducible and accurate algorithm, assuming that
 *  a matrix and a vector are composed of real numbers.
 *
 *  If fpe < 3, it relies on superaccumulators only. Otherwise, it relies on 
 *  floating-point expansions of size FPE with superaccumulators when needed
 *
 * \param uplo 'U' or 'L' an upper or a lower triangular matrix A
 * \param transa 'T' or 'N' a transpose or a non-transpose matrix A
 * \param diag 'U' or 'N' a unit or non-unit triangular matrix A
 * \param n size of matrix A
 * \param a matrix A
 * \param lda leading dimension of A
 * \param x vector
 * \param incx the increment for the elements of a
 * \param fpe size of floating-point expansion
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return matrix C contains the reproducible and accurate result of the matrix product
 */
int extrsv(char uplo, char transa, char diag, int n, double *a, int lda, double *x, int incx, int fpe, bool early_exit = false);

#endif // BLAS2_HPP_

