/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file ExSUM.hpp
 *  \brief Provides a set of summation routines
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef EXSUM_HPP_
#define EXSUM_HPP_

#include "common.hpp"


/**
 * \ingroup ExSUM
 * \brief Executes parallel summation/reduction on elements of a real vector with our 
 *     multi-level reproducible and accurate algorithm that relies upon
 *     floating-point expansions in conjuction with superaccumulators or superaccumulators
 *     only
 *
 * \param N vector size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param fpe size of floating-point expansion
 * \param program_file path to the file with kernels
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
double runExSUM(int N, double *a, int inca, int fpe, const char* program_file);

#endif // EXSUM_HPP_
