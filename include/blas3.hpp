/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file blas3.hpp
 *  \brief Provides a set of Basic Linear Algera Subprograms level-3 to work with matrices
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef BLAS3_HPP_
#define BLAS3_HPP_

// config from cmake
#include "config.h"

/**
 * \defgroup blas3 BLAS Level-3 Functions
 */

/**
 * \defgroup ExGEMM GEMM Functions
 * \ingroup blas3
 */

/**
 * \ingroup ExGEMM
 * \brief ExGEMM computes the matrix-matrix multiplication (three matrices are composed of real numbers) 
 *     using our multi-level reproducible and accurate algorithm.
 *
 *     If fpe < 2, it relies on superaccumulators only. Otherwise, it relies on floating-point expansions
 *     of size FPE with superaccumulators when needed
 *
 * \param transa 'T' or 'N' -- transpose or non-transpose matrix A
 * \param transb 'T' or 'N' -- transpose or non-transpose matrix B
 * \param m nb of rows of matrix C
 * \param n nb of columns of matrix C
 * \param k nb of rows in matrix B
 * \param alpha scalar
 * \param a matrix A
 * \param lda leading dimension of A
 * \param b matrix B
 * \param ldb leading dimension of B
 * \param beta scalar
 * \param c matrix C
 * \param ldc leading dimension of C
 * \param fpe size of floating-point expansion
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return matrix C contains the reproducible and accurate result of the matrix product
 */
int exgemm(char transa, char transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc, int fpe, bool early_exit = false);

#endif // BLAS3_HPP_
