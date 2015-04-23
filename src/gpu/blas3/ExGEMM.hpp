/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file gpu/blas3/ExGEMM.hpp
 *  \brief Provides a set of gemm routines
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef EXGEMM_HPP_
#define EXGEMM_HPP_

#include "common.hpp"


/**
 * \ingroup ExGEMM
 * \brief Executes on GPU parallel matrix-matrix multiplication (C := beta * C + alpha * op(A) * op(B), where op(X) = X or op(X) = X^T)
 *     built on top of our multi-level reproducible and accurate algorithm that relies upon floating-point expansions in conjuction
 *     with superaccumulators. For internal use
 *
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
 * \param program_file path to the file with kernels
 * \return matrix C contains the reproducible and accurate result of the matrix product
 */
int runExGEMM(int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc, int fpe, const char* program_file);

#endif // EXGEMM_HPP_
