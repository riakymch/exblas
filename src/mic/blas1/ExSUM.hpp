/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file mic/blas1/ExSUM.hpp
 *  \brief Provides a set of summation routines
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef DSUM_H_
#define DSUM_H_

#include "superaccumulator.hpp"
#include "ExSUM.MIC.hpp"
#include "ExSUM.FPE.hpp"
#include <omp.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_scheduler_init.h>
#include "common.hpp"


/**
 * \class TBBlongsum
 * \ingroup ExSUM
 * \brief This class is meant to be used in our multi-level reproducible and 
 *  accurate algorithm with superaccumulators only
 */
class TBBlongsum {
    double* a; /**< a real vector to sum */
public:
    Superaccumulator acc; /**< supperaccumulator */

    /**
     * The main function that performs summation of the vector's elelements into the 
     * superaccumulator
     */
    void operator()(tbb::blocked_range<size_t> const & r) {
        for(size_t i = r.begin(); i != r.end(); i += r.grainsize()) 
            acc.Accumulate(a[i]);
    }

    /** 
     * Construction that uses another object of TBBlongsum for initialization
     * \param x a TBBlongsum instance
     */
    TBBlongsum(TBBlongsum & x, tbb::split) : a(x.a), acc(e_bits, f_bits) {}

    /** 
     * Joins two superaccumulators of two different instances
     * \param y a TBBlongsum instance
     */
    void join(TBBlongsum & y) { acc.Accumulate(y.acc); }

    /** 
     * Construction that initiates a real vector to sum and a supperacccumulator
     * \param a a real vector
     */
    TBBlongsum(double a[]) :
        a(a), acc()
    {}
};

/**
 * \ingroup ExSUM
 * \brief Parallel summation computes the sum of elements of a real vector with our 
 *     multi-level reproducible and accurate algorithm that solely relies upon superaccumulators
 *
 * \param N vector size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param offset specifies position in the vector to start with 
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
double ExSUMSuperaccBack(int N, double *a, int inca, int offset);

/**
 * \ingroup ExSUM
 * \brief Parallel summation computes the sum of elements of a real vector with our 
 *     multi-level reproducible and accurate algorithm that solely relies upon 
 *     superaccumulators. This algorithm is based on TBB
 *
 * \param N vector size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param offset specifies position in the vector to start with 
 * TODO: not done for offset
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
double ExSUMSuperacc(int N, double *a, int inca, int offset);

/**
 * \ingroup ExSUM
 * \brief Parallel summation computes the sum of elements of a real vector with our 
 *     multi-level reproducible and accurate algorithm that relies upon 
 *     floating-point expansions of size CACHE and superaccumulators when needed
 *
 * \param N vector size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param offset specifies position in the vector to start with 
 * TODO: not done for inca and offset
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
template<typename CACHE> double ExSUMFPE(int N, double *a, int inca, int offset);

#endif // EXSUM_H_
