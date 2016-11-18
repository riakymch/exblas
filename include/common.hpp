/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file common.hpp
 *  \brief A set of common definitions and functions
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <cstdio>
#include <cstdlib>

/**
 * \defgroup common Common Definitions and Functions
 */

/**
 * \ingroup common
 * \brief Maximum exponent
 */
int constexpr e_bits = 1023;    // 319

/**
 * \ingroup common
 * \brief Maximum exponent + the number of bits in signigicant
 */
int constexpr f_bits = 1023 + 52;   // 300

/**
 * \ingroup common
 * \brief Number of limbs in superaccumulator
 */
int constexpr bin_count = 39;


/**
 * \ingroup common
 * \brief Generates a random number for log-uniform distribution
 *
 * \return The generated number
 */
double randDoubleUniform();

/**
 * \ingroup common
 * \brief Generates a random number for uniform distribution
 *
 * \param emin minimum exponent
 * \param emax maximum exponenet
 * \param neg_ratio
 * \return The generated number
 */
double randDouble(int emin, int emax, int neg_ratio);

/**
 * \ingroup common
 * \brief Generates a real vector with uniform distribution of its ellements
 *
 * \param n vector size
 * \param a input/output vector
 * \param range dynamic range of generated elements
 * \param emax maximum exponent (emax + range < EMAX)
 */
void init_fpuniform(const int n, double *a, int range, int emax);

/**
 * \ingroup common
 * \brief Generates a real matrix with uniform distribution of its ellements
 *
 * \param iscolumnwise true|false column-major or row-major order
 * \param m nb of rows of matrix a
 * \param n nb of columns of matrix a
 * \param a input/output matrix
 * \param lda leading dimension of matrix a
 * \param range dynamic range of generated elements
 * \param emax maximum exponent (emax + range < EMAX)
 */
void init_fpuniform_matrix(const bool iscolumnwise, const int m, const int n, double *a, const int lda, const int range, const int emax);

/**
 * \ingroup common
 * \brief Generates a real lower/upper or unit/non-unit matrix with uniform distribution of its ellements
 *
 * \param uplo L|U lower or upper triangular matrix
 * \param diag U|N unit or non-unit diagonal
 * \param n matrix size
 * \param a input/output matrix
 * \param range dynamic range of generated elements
 * \param emax maximum exponent (emax + range < EMAX)
 */
void init_fpuniform_tr_matrix(const char uplo, const char diag, const int n, double *a, const int range, const int emax);

/**
 * \ingroup common
 * \brief Generates a real vector with log-uniform distribution of its ellements
 *
 * \param n vector size
 * \param a input/output vector
 * \param mean 
 * \param stddev
 */
void init_lognormal(const int n, double *a, double mean, double stddev);

/**
 * \ingroup common
 * \brief Generates a real matrix with log-uniform distribution of its ellements
 *
 * \param iscolumnwise true|false column-major or row-major order
 * \param m nb of rows of matrix a
 * \param n nb of columns of matrix a
 * \param a input/output matrix
 * \param lda leading dimension of matrix a
 * \param range dynamic range of generated elements
 * \param emax maximum exponent (emax + range < EMAX)
 */
void init_lognormal_matrix(const bool iscolumnwise, const int m, const int n, double *a, const int lda, const double mean, const double stddev);

/**
 * \ingroup common
 * \brief Generates a real lower/upper or unit/non-unit matrices with log-uniform distribution of its ellements
 *
 * \param uplo L|U lower or upper triangular matrix
 * \param diag U|N unit or non-unit diagonal
 * \param n matrix size
 * \param a input/output matrix
 * \param mean
 * \param stddev
 */
void init_lognormal_tr_matrix(const char uplo, const char diag, const int n, double *a, const double mean, const double stddev);

/**
 * \ingroup common
 * \brief Generates a real vector with ill-conditioned ellements using \n
 * Algorithm 6.1. Generation of extremely ill-conditioned dot products from \n
 * T.Ogita et al. Accurate sum and dot product, SIAM Journal on Scientific Computing, 
 * 26(6):1955-1988, 2005.
 *
 * \param n vector size
 * \param a input/output vector
 * \param c anticipated condition number
 */
void init_ill_cond(const int n, double *a, double c);

/**
 * \ingroup common
 * \brief Generates a real vector with all ellements set to 1.1
 *
 * \param n vector size
 * \param a input/output vector
 */
void init_naive(const int n, double *a);

#endif // COMMON_H
