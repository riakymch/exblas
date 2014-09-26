#ifndef OCLBLAS_Z_H_
#define OCLBLAS_Z_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CREAL
#define CREAL(v) ((v).s[0])
#endif
#ifndef CIMAG
#define CIMAG(v) ((v).s[1])
#endif

#ifdef __cplusplus
}
#endif

#endif // #ifndef OCLBLAS_COMPLEX_Z_
