/* oclblas_xerbla.cpp
 *
 *  Purpose
 *  =======
 *
 *  XERBLA  is an error handler for the LAPACK routines.
 *  It is called by an LAPACK routine if an input parameter has an
 *  invalid value.  A message is printed and execution stops.
 *
 *  Installers may consider modifying the STOP statement in order to
 *  call system-specific exception-handling facilities.
 *
 *  Arguments
 *  =========
 *
 *  SRNAME  (input) CHARACTER*(*)
 *          The name of the routine which called XERBLA.
 *
 *  INFO    (input) INTEGER
 *          The position of the invalid parameter in the parameter list
 *          of the calling routine.
 *
 * =====================================================================
 */
#include <cstdio>

extern "C"
void oclblas_xerbla(const char *srname, void *vinfo)
{
   int *info = (int *)vinfo;
   fprintf(stdout, "Argument %d to routine %s was incorrect\n", *info, srname);
}
