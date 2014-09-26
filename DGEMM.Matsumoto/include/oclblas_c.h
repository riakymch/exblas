/* Some parts of this file are based on clAmdBlas-complex.h of clAmdBlas
 *	Copyright (C) 2010 Advanced Micro Devices, Inc. All Rights Reserved.
 */
#ifndef OCLBLAS_C_H_
#define OCLBLAS_C_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef cl_float2 FloatComplex;

static __inline FloatComplex
floatComplex(float real, float imag) {
    FloatComplex z;
    z.s[0] = real;
    z.s[1] = imag;
    return z;
}

#ifndef CREAL
#define CREAL(v) ((v).s[0])
#endif
#ifndef CIMAG
#define CIMAG(v) ((v).s[1])
#endif

#ifdef __cplusplus
}
#endif

#endif // #ifndef OCLBLAS_COMPLEX_H_
